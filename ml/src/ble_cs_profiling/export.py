"""Export a trained checkpoint to hardware-independent deployment formats.

Entry point: ``ble-cs export <checkpoint>`` (see ``ble-cs export --help``).

Output layout per checkpoint, one folder per precision::

    <output root>/<checkpoint>/fp32_full/model.{onnx,tflite}   float32 reference
                               int8_full/model.tflite          INT8 incl. input/output (MCUs)
                               int8_qdq/model.onnx             INT8 weights/activations, float I/O
                               results.json                    sizes, hashes, MAE (see results.py)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ble_cs_profiling import config as cfg
from ble_cs_profiling.checkpoint import load_checkpoint
from ble_cs_profiling.data import load_calibration_inputs, load_eval_set
from ble_cs_profiling.quantization import evaluate_tflite, quantize_full_int8
from ble_cs_profiling.results import record_export

FP32 = "fp32_full"
INT8_FULL = "int8_full"
INT8_QDQ = "int8_qdq"

ONNX_INPUT_NAME = "iq_input"
ONNX_OUTPUT_NAME = "distance_m"


@dataclass(frozen=True)
class ExportOptions:
    """Which formats to export and whether to evaluate them."""

    onnx: bool = True  # FP32 + INT8 QDQ ONNX (NPU and GPU toolchains)
    tflite: bool = True  # FP32 + full INT8 TFLite (TFLite Micro on MCUs)
    onnx_opset: int = 13  # see export_onnx() for why not newer
    evaluate: bool = True  # MAE of both TFLite models on the full eval set


def export_checkpoint(
    checkpoint_path: Path,
    output_root: Path,
    options: ExportOptions | None = None,
) -> dict[str, Path]:
    """Export one checkpoint to all requested formats and record the results.
    """
    options = options or ExportOptions()
    if not (options.onnx or options.tflite):
        raise ValueError("Nothing to export: enable onnx and/or tflite")

    model, checkpoint = load_checkpoint(checkpoint_path)
    input_shape = tuple(checkpoint.get("input_shape", [1, cfg.NUM_FEATURES]))
    if options.tflite and input_shape[0] != 1:
        raise ValueError("TFLite quantization and evaluation require batch size 1")

    name = checkpoint_name(checkpoint_path)
    run_dir = output_root / name
    print(f"Loaded checkpoint: {checkpoint_path} ({checkpoint['model_name']})")

    # Only the shape matters for tracing, the values do not end up in the exported graph.
    sample_input = torch.randn(*input_shape, dtype=torch.float32)
    # Loaded once, shared by ONNX and TFLite: both quantize on exactly the same samples.
    calibration_inputs = _calibration_inputs(checkpoint, input_shape)

    artifacts: dict[str, Path] = {}
    if options.onnx:
        artifacts |= export_onnx_models(model, sample_input, calibration_inputs, run_dir, options.onnx_opset)
    if options.tflite:
        artifacts |= export_tflite_models(model, sample_input, calibration_inputs, run_dir)

    maes = evaluate_tflite_models(artifacts, checkpoint, input_shape) if options.tflite and options.evaluate else {}
    record_export(run_dir, name.as_posix(), checkpoint, list(artifacts.values()), maes)
    return artifacts


def checkpoint_name(checkpoint_path: Path) -> Path:
    """Name of the output folder: the checkpoint's path inside the checkpoint dir, without ``.pt``.

    ``checkpoints/mobilenet/relu.pt`` becomes ``mobilenet/relu``. The registry name is not
    used on purpose: variants of one architecture share it and
    would overwrite each other. Checkpoints outside the checkpoint dir use their file name.
    """
    try:
        return checkpoint_path.resolve().relative_to(cfg.CHECKPOINT_DIR).with_suffix("")
    except ValueError:
        return Path(checkpoint_path.stem)


def export_onnx_models(
    model: nn.Module,
    sample_input: torch.Tensor,
    calibration_inputs: np.ndarray,
    run_dir: Path,
    opset_version: int,
) -> dict[str, Path]:
    """Export the FP32 ONNX model and its statically quantized INT8 QDQ version."""
    fp32_path = export_onnx(model, sample_input, _artifact_path(run_dir, FP32, ".onnx"), opset_version)
    int8_path = quantize_onnx_int8(fp32_path, _artifact_path(run_dir, INT8_QDQ, ".onnx"), calibration_inputs)
    return {"onnx_fp32": fp32_path, "onnx_int8_qdq": int8_path}


def export_tflite_models(
    model: nn.Module,
    sample_input: torch.Tensor,
    calibration_inputs: np.ndarray,
    run_dir: Path,
) -> dict[str, Path]:
    """Convert the model to FP32 TFLite, check it against PyTorch, then quantize to full INT8."""
    fp32_path, edge_model = export_tflite(model, sample_input, _artifact_path(run_dir, FP32, ".tflite"))
    _check_tflite_matches_pytorch(model, edge_model, sample_input)
    int8_path = quantize_full_int8(fp32_path, _artifact_path(run_dir, INT8_FULL, ".tflite"), calibration_inputs)
    return {"tflite_fp32": fp32_path, "tflite_int8_full": int8_path}


def evaluate_tflite_models(
    artifacts: dict[str, Path],
    checkpoint: dict[str, Any],
    input_shape: tuple[int, ...],
) -> dict[Path, float]:
    """Run every exported TFLite model on the full eval set and return its MAE per path."""
    feature_names = checkpoint.get("feature_names", list(cfg.FEATURES_RAW_IQ_ALL))
    eval_inputs, targets = load_eval_set(feature_names, checkpoint.get("target", cfg.TARGET))
    eval_inputs = eval_inputs.reshape((-1, *input_shape[1:]))
    tflite_paths = [path for path in artifacts.values() if path.suffix == ".tflite"]
    return {path: evaluate_tflite(path, eval_inputs, targets) for path in tflite_paths}


def export_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: Path,
    opset_version: int = 13,
) -> Path:
    """Export the FP32 model to ONNX (target: STM32N6 NPU via ST Edge AI, Jetson).

    Opset 13 and the legacy TorchScript exporter on purpose: ONNX ``HardSwish`` only exists
    from opset 14; at opset 13 it is lowered to HardSigmoid + Mul. The ST toolchain used
    initially rejected ONNX HardSwish on the NPU. Newer ST Edge AI versions list it as
    supported, so this may be revisited, see docs/onnx_export_stm32n6.md.
    """
    torch.onnx.export(
        model,
        (sample_input,),
        f=str(output_path),
        dynamo=False,
        export_params=True,
        opset_version=opset_version,
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        dynamic_axes={ONNX_INPUT_NAME: {0: "batch"}, ONNX_OUTPUT_NAME: {0: "batch"}},
        external_data=False,
    )
    _print_saved("ONNX FP32", output_path)
    return output_path


def quantize_onnx_int8(fp32_path: Path, output_path: Path, calibration_inputs: np.ndarray) -> Path:
    """Quantize an ONNX model statically to INT8 (QDQ format, per-channel weights, float I/O)."""
    from tempfile import TemporaryDirectory

    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if len(calibration_inputs) == 0 or not np.isfinite(calibration_inputs).all():
        raise ValueError("Calibration inputs must be nonempty and finite")

    with TemporaryDirectory() as temporary:
        prepared_path = Path(temporary) / "prepared.onnx"
        # ONNX Runtime's symbolic shape inference fails on the MobileNet conv shapes,
        # standard ONNX shape inference and graph optimization are sufficient.
        quant_pre_process(str(fp32_path), str(prepared_path), skip_symbolic_shape=True)
        quantize_static(
            str(prepared_path),
            str(output_path),
            _OnnxCalibrationReader(calibration_inputs),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    _print_saved("ONNX INT8 QDQ", output_path)
    return output_path


def export_tflite(model: nn.Module, sample_input: torch.Tensor, output_path: Path) -> tuple[Path, Any]:
    """Convert the PyTorch model to FP32 TFLite; returns the path and the in-memory edge model."""
    import litert_torch

    edge_model = litert_torch.convert(model, (sample_input,))
    edge_model.export(str(output_path))
    _print_saved("TFLite FP32", output_path)
    return output_path, edge_model


def _check_tflite_matches_pytorch(model: nn.Module, edge_model: Any, sample_input: torch.Tensor) -> None:
    """Print the largest output difference between PyTorch and the converted TFLite model."""
    with torch.inference_mode():
        torch_output = model(sample_input).cpu().numpy()
    edge_output = np.asarray(edge_model(sample_input))
    max_difference = float(np.max(np.abs(torch_output - edge_output)))
    print(f"TFLite vs. PyTorch, max. absolute difference: {max_difference:.6g}")


class _OnnxCalibrationReader:
    """Feeds the calibration samples one by one to ONNX Runtime's static quantization."""

    def __init__(self, calibration_inputs: np.ndarray) -> None:
        self._samples = iter(calibration_inputs)

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next sample as a batch of one, or None when all samples are used."""
        sample = next(self._samples, None)
        if sample is None:
            return None
        return {ONNX_INPUT_NAME: np.asarray(sample[None, ...], dtype=np.float32)}


def _calibration_inputs(checkpoint: dict[str, Any], input_shape: tuple[int, ...]) -> np.ndarray:
    """Load the calibration samples and reshape them to the model's input shape."""
    feature_names = checkpoint.get("feature_names", list(cfg.FEATURES_RAW_IQ_ALL))
    return load_calibration_inputs(feature_names).reshape((-1, *input_shape[1:]))


def _artifact_path(run_dir: Path, precision: str, suffix: str) -> Path:
    """Return ``<run_dir>/<precision>/model<suffix>`` and create its folder."""
    folder = run_dir / precision
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"model{suffix}"


def _print_saved(label: str, path: Path) -> None:
    """Print where a model file was written and how large it is."""
    print(f"Saved {label}: {path} ({path.stat().st_size / 1024:.1f} KiB)")
