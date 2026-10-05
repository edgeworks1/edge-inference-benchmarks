"""Load a trained checkpoint and export it to deployment formats.

Entry point: ``ble-cs export <checkpoint>`` (see ``ble-cs export --help``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ble_cs_profiling import config as cfg
from ble_cs_profiling.models import build_model
from ble_cs_profiling.quantization import evaluate_tflite, load_features, quantize_full_int8


@dataclass(frozen=True)
class ExportOptions:
    onnx: bool = True  # FP32 + statically quantized INT8 ONNX
    tflite: bool = True  # FP32 + full INT8 TFLite (integer-only incl. I/O for TFLM)
    onnx_opset: int = 13
    evaluate: bool = True  # MAE of both TFLite artifacts on the eval set


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from its metadata and load its trained weights."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nRun `ble-cs train` first."
        )
    
    # Export vermeidet GPU-abhängige Artefakte, daher immer auf CPU laden
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with PyTorch versions that predate weights_only.
        checkpoint = torch.load(path, map_location="cpu")

    required_keys = {"model_name", "model_kwargs", "model_state_dict"}
    missing = required_keys - checkpoint.keys()
    if missing:
        raise ValueError(f"Invalid checkpoint; missing keys: {sorted(missing)}")

    model = build_model(checkpoint["model_name"], checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.cpu().eval(), checkpoint


def export_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: Path,
    opset_version: int = 13,
) -> Path:
    """Export with legacy opset support and decomposed HardSwish at opset 13."""
    # The TorchScript exporter supports opset 13 directly, including lowering
    # HardSwish to HardSigmoid + Mul for older deployment runtimes.
    torch.onnx.export(
        model,
        (sample_input,),
        f=str(output_path),
        dynamo=False,
        export_params=True,
        opset_version=opset_version,
        input_names=["iq_input"],
        output_names=["distance_m"],
        dynamic_axes={"iq_input": {0: "batch"}, "distance_m": {0: "batch"}},
        external_data=False,
    )
    print(f"ONNX gespeichert:  {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    return output_path


def load_calibration_inputs(
    checkpoint: dict[str, Any],
    input_shape: tuple[int, ...],
) -> np.ndarray:
    """Select CALIBRATION_SAMPLES reproducible training samples for all deployment formats."""
    feature_names = checkpoint.get("feature_names", list(cfg.FEATURES_RAW_IQ_ALL))
    training_inputs = load_features(cfg.TRAIN_DATA_PATH, feature_names)
    if len(training_inputs) == 0:
        raise ValueError("Calibration training data must be nonempty")
    indices = np.random.default_rng(cfg.SEED).choice(
        len(training_inputs), min(cfg.CALIBRATION_SAMPLES, len(training_inputs)), replace=False,
    )
    return training_inputs[indices].reshape((-1, *input_shape[1:]))


def quantize_onnx_int8(
    fp32_path: Path, output_path: Path, calibration_inputs: np.ndarray,
) -> Path:
    """Export static INT8 QDQ weights/activations with floating-point I/O."""
    from tempfile import TemporaryDirectory

    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if len(calibration_inputs) == 0 or not np.isfinite(calibration_inputs).all():
        raise ValueError("Calibration inputs must be nonempty and finite")

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.samples = iter(calibration_inputs)

        def get_next(self) -> dict[str, np.ndarray] | None:
            sample = next(self.samples, None)
            if sample is None:
                return None
            return {"iq_input": np.asarray(sample[None, ...], dtype=np.float32)}

    with TemporaryDirectory() as temporary:
        prepared_path = Path(temporary) / "prepared.onnx"
        # ORT symbolic inference fails on the Dynamo-exported MobileNet Conv
        # shapes. Keep standard ONNX shape inference and graph optimization.
        quant_pre_process(str(fp32_path), str(prepared_path), skip_symbolic_shape=True)
        quantize_static(
            str(prepared_path), str(output_path), Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    print(f"ONNX INT8 gespeichert: {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    return output_path


def export_tflite(
    model: nn.Module,
    sample_input: tuple[torch.Tensor, ...],
    output_path: Path,
) -> tuple[Path, Any]:
    """Convert the PyTorch model to an FP32 LiteRT/TFLite model."""
    try:
        import litert_torch
    except ImportError as error:
        raise RuntimeError(
            "TFLite export requires litert_torch in the active environment."
        ) from error

    edge_model = litert_torch.convert(model, sample_input)
    edge_model.export(str(output_path))
    print(f"TFLite gespeichert: {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    return output_path, edge_model


def check_output(
    model: nn.Module,
    sample_input: tuple[torch.Tensor, ...],
    edge_model: Any,
) -> None:
    """Compare TFLite output numerically with its PyTorch source model."""
    with torch.inference_mode():
        torch_output = model(*sample_input).cpu().numpy() # Das * ist notwendig, um das Tupel zu entpacken und die einzelnen Tensoren als separate Argumente an das Modell zu übergeben.
    edge_output = np.asarray(edge_model(*sample_input))
    max_abs_diff = float(np.max(np.abs(torch_output - edge_output)))
    print(f"TFLite max. absolute Abweichung zu PyTorch: {max_abs_diff:.6g}")


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    options: ExportOptions | None = None,
) -> dict[str, Path]:
    """Export one checkpoint to all requested formats; returns ``{artifact: path}``."""
    options = options or ExportOptions()
    if not (options.onnx or options.tflite):
        raise ValueError("Nothing to export: enable onnx and/or tflite")

    model, checkpoint = load_checkpoint(checkpoint_path)
    model_name = checkpoint["model_name"]
    input_shape = tuple(checkpoint.get("input_shape", [1, cfg.NUM_FEATURES]))
    if options.tflite and input_shape[0] != 1:
        raise ValueError("TFLite PTQ/evaluation requires export batch size 1")
    sample_input = (torch.randn(*input_shape, dtype=torch.float32),)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint geladen: {checkpoint_path} ({model_name})")
    if "eval_mae" in checkpoint:
        print(f"Gespeicherter Eval-MAE: {checkpoint['eval_mae']:.4f} m")

    # Wird für Quantisierung benötigt; einmal laden, für ONNX und TFLite wiederverwenden
    calibration_inputs = load_calibration_inputs(checkpoint, input_shape)

    artifacts: dict[str, Path] = {}
    if options.onnx:
        artifacts["fp32_onnx"] = export_onnx(
            model, sample_input[0], output_dir / f"{model_name}_fp32.onnx", options.onnx_opset,
        )
        artifacts["int8_onnx"] = quantize_onnx_int8(
            artifacts["fp32_onnx"], output_dir / f"{model_name}_int8.onnx", calibration_inputs,
        )

    if options.tflite:
        fp32_path, edge_model = export_tflite(
            model, sample_input, output_dir / f"{model_name}_fp32.tflite",
        )
        check_output(model, sample_input, edge_model)
        artifacts["fp32"] = fp32_path

        artifacts["full_int8"] = quantize_full_int8(
            fp32_path, output_dir / f"{model_name}_full_int8.tflite", calibration_inputs,
        )

        if options.evaluate:
            feature_names = checkpoint.get("feature_names", list(cfg.FEATURES_RAW_IQ_ALL))
            target_name = checkpoint.get("target", cfg.TARGET)
            eval_inputs = load_features(cfg.EVAL_DATA_PATH, feature_names).reshape(
                (-1, *input_shape[1:])
            )
            targets = pd.read_parquet(cfg.EVAL_DATA_PATH, columns=[target_name])[target_name].to_numpy()
            for path in artifacts.values():
                if path.suffix == ".tflite":
                    evaluate_tflite(path, eval_inputs, targets)

    return artifacts
