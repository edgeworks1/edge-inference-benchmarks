"""Load a trained checkpoint and export it to deployment formats.

Change the constants in ``DEPLOYMENT CONFIGURATION`` and run
``python -m ble_cs_profiling.export``.
"""

from pathlib import Path
from typing import Any

import traceback

import numpy as np
import torch
from torch import nn

from ble_cs_profiling import config as cfg
from ble_cs_profiling.models import build_model
from ble_cs_profiling.quantization import evaluate_tflite, load_features, quantize_full_int8


# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================
CHECKPOINT_PATH = cfg.CHECKPOINT_DIR / "mobilenet" / "fp32_full.pt"
EXPORT_ONNX = True  # Export FP32 and statically quantized INT8 ONNX
EXPORT_TFLITE = True
ONNX_OPSET_VERSION = 13
EXPORT_FULL_INT8 = True  # Static PTQ to integer-only TFLite
CALIBRATION_SAMPLES = 1000
CALIBRATION_SEED = 42


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from its metadata and load its trained weights."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nRun ble_cs_profiling.train first or change "
            "CHECKPOINT_PATH in ble_cs_profiling/export.py."
        )

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
        opset_version=ONNX_OPSET_VERSION,
        input_names=["iq_input"],
        output_names=["distance_m"],
        dynamic_axes={"iq_input": {0: "batch"}, "distance_m": {0: "batch"}},
        external_data=False,
    )
    print(f"ONNX gespeichert:  {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    return output_path


def load_calibration_inputs(
    checkpoint: dict[str, Any], input_shape: tuple[int, ...],
) -> np.ndarray:
    """Select reproducible training samples for both deployment formats."""
    if CALIBRATION_SAMPLES < 1:
        raise ValueError("CALIBRATION_SAMPLES must be positive")
    feature_names = checkpoint.get("feature_names", list(cfg.FEATURES_RAW_IQ_ALL))
    training_inputs = load_features(cfg.TRAIN_DATA_PATH, feature_names)
    if len(training_inputs) == 0:
        raise ValueError("Calibration training data must be nonempty")
    indices = np.random.default_rng(CALIBRATION_SEED).choice(
        len(training_inputs), min(CALIBRATION_SAMPLES, len(training_inputs)), replace=False,
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
        torch_output = model(*sample_input).cpu().numpy()
    edge_output = np.asarray(edge_model(*sample_input))
    max_abs_diff = float(np.max(np.abs(torch_output - edge_output)))
    print(f"TFLite max. absolute Abweichung zu PyTorch: {max_abs_diff:.6g}")


def main() -> None:
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH)
    model_name = checkpoint["model_name"]
    input_shape = tuple(checkpoint.get("input_shape", [1, cfg.NUM_FEATURES]))
    sample_input = (torch.randn(*input_shape, dtype=torch.float32),)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint geladen: {CHECKPOINT_PATH}")
    if "eval_mae" in checkpoint:
        print(f"Gespeicherter Eval-MAE: {checkpoint['eval_mae']:.4f} m")

    if EXPORT_ONNX:
        try:
            onnx_path = export_onnx(
                model,
                sample_input[0],
                cfg.OUTPUT_DIR / f"{model_name}_fp32.onnx",
            )
            quantize_onnx_int8(
                onnx_path, cfg.OUTPUT_DIR / f"{model_name}_int8.onnx",
                load_calibration_inputs(checkpoint, input_shape),
            )
        except ModuleNotFoundError as error:
            print(
                f"ONNX-Export/Quantisierung unvollständig: Python-Paket '{error.name}' fehlt. "
                "Installiere 'onnx', 'onnxscript' und 'onnxruntime' in der aktiven Umgebung."
            )
        except Exception as error:
            print(f"ONNX-Export/Quantisierung fehlgeschlagen: {type(error).__name__}: {error}")
            traceback.print_exc()

    if EXPORT_TFLITE or EXPORT_FULL_INT8:
        try:
            fp32_path, edge_model = export_tflite(
                model,
                sample_input,
                cfg.OUTPUT_DIR / f"{model_name}_fp32.tflite",
            )
            check_output(model, sample_input, edge_model)
            if EXPORT_FULL_INT8:
                import pandas as pd

                feature_names = checkpoint.get("feature_names", list(cfg.FEATURES_RAW_IQ_ALL))
                target_name = checkpoint.get("target", cfg.TARGET)
                if input_shape[0] != 1:
                    raise ValueError("TFLite PTQ/evaluation requires export batch size 1")
                calibration_inputs = load_calibration_inputs(checkpoint, input_shape)
                int8_path = quantize_full_int8(
                    fp32_path, cfg.OUTPUT_DIR / f"{model_name}_full_int8.tflite", calibration_inputs,
                )
                eval_inputs = load_features(cfg.EVAL_DATA_PATH, feature_names).reshape(
                    (-1, *input_shape[1:])
                )
                targets = pd.read_parquet(cfg.EVAL_DATA_PATH, columns=[target_name])[target_name].to_numpy()
                evaluate_tflite(fp32_path, eval_inputs, targets)
                evaluate_tflite(int8_path, eval_inputs, targets)
        except Exception as error:
            raise RuntimeError(f"TFLite-Export/Quantisierung/Auswertung fehlgeschlagen: {error}") from error


if __name__ == "__main__":
    main()
