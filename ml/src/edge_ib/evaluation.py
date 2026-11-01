"""Evaluate an exported TFLite model by running the actual file with the LiteRT interpreter.

This measures exactly what will be deployed: a quantized model gets float samples
quantized to INT8 at its input and its INT8 prediction dequantized back to metres.
"""

from pathlib import Path
from typing import Any

import numpy as np

PROGRESS_INTERVAL = 10_000  # print progress every N samples


def evaluate_tflite(path: Path, inputs: np.ndarray, targets: np.ndarray) -> float:
    """Return the mean absolute error (in target units) of a TFLite model on ``inputs``."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(inputs) == 0 or len(inputs) != len(targets):
        raise ValueError("Evaluation requires equally sized, nonempty inputs and targets")

    interpreter, input_detail, output_detail = _load_single_io_interpreter(path)
    total_error = 0.0
    for index, (sample, target) in enumerate(zip(inputs, targets, strict=True), start=1):
        error = abs(_predict_one(interpreter, input_detail, output_detail, sample) - target)
        if not np.isfinite(error):
            raise ValueError("Non-finite prediction or target during evaluation")
        total_error += error
        if index % PROGRESS_INTERVAL == 0:
            print(f"Evaluating {path}: {index:,}/{len(inputs):,} samples", flush=True)

    mae = total_error / len(inputs)
    print(f"MAE on eval set [{path}]: {mae:.4f} m ({len(inputs):,} samples)", flush=True)
    return mae


def _load_single_io_interpreter(path: Path) -> tuple[Any, dict, dict]:
    """Load a TFLite model that has exactly one input and one output tensor."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("Evaluation requires exactly one input and one output")
    return interpreter, input_details[0], output_details[0]


def _predict_one(
    interpreter: Any, input_detail: dict, output_detail: dict, sample: np.ndarray
) -> float:
    """Run one float sample through the model and return its prediction as a float."""
    interpreter.set_tensor(input_detail["index"], _to_model_input(sample[None, ...], input_detail))
    interpreter.invoke()
    prediction = _to_float_output(interpreter.get_tensor(output_detail["index"]), output_detail)
    if prediction.size != 1:
        raise ValueError("The MAE requires exactly one scalar prediction per sample")
    return prediction.item()


def _to_model_input(value: np.ndarray, detail: dict) -> np.ndarray:
    """Quantize a float input (``round(value / scale) + zero_point``) for integer models."""
    if np.issubdtype(detail["dtype"], np.integer):
        scale, zero_point = detail["quantization"]
        if scale <= 0:
            raise ValueError("Integer model input without quantization scale")
        limits = np.iinfo(detail["dtype"])
        value = np.clip(np.rint(value / scale + zero_point), limits.min, limits.max)
    return value.astype(detail["dtype"])


def _to_float_output(value: np.ndarray, detail: dict) -> np.ndarray:
    """Dequantize an integer output (``(value - zero_point) * scale``) back to float."""
    value = value.astype(np.float64)
    if np.issubdtype(detail["dtype"], np.integer):
        scale, zero_point = detail["quantization"]
        if scale <= 0:
            raise ValueError("Integer model output without quantization scale")
        value = (value - zero_point) * scale
    return value
