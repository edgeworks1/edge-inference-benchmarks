"""Model-independent static INT8 PTQ and evaluation for single-input regressors."""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def quantize_full_int8(
    fp32_path: Path, output_path: Path, calibration_inputs: np.ndarray,
) -> Path:
    """Calibrate a LiteRT graph and reject float fallback, including float I/O."""
    from ai_edge_quantizer import quantizer, recipe

    if len(calibration_inputs) == 0 or not np.isfinite(calibration_inputs).all():
        raise ValueError("Calibration inputs must be nonempty and finite")

    quantization_recipe = recipe.static_wi8_ai8()
    value_ranges = _calibrate_value_ranges(fp32_path, quantization_recipe, calibration_inputs)
    result = quantizer.Quantizer(fp32_path, quantization_recipe).quantize(value_ranges)
    _write_with_int8_io(_inline_constant_buffers(result.quantized_model), output_path)

    print(f"TFLite full_int8 gespeichert: {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    return output_path


def _single_signature_input(model_path: Path) -> tuple[str, str]:
    """Return ``(signature, input_name)`` of a model with exactly one input and output."""
    from ai_edge_litert.interpreter import Interpreter

    signatures = Interpreter(model_path=str(model_path)).get_signature_list()
    if len(signatures) != 1:
        raise ValueError("PTQ currently requires exactly one model signature")
    signature, details = next(iter(signatures.items()))
    if len(details['inputs']) != 1 or len(details['outputs']) != 1:
        raise ValueError("PTQ currently requires one input and one output")
    return signature, details['inputs'][0]


def _calibrate_value_ranges(
    fp32_path: Path, quantization_recipe: Any, calibration_inputs: np.ndarray,
) -> dict:
    """Run the calibration samples through the FP32 model and record every tensor's range."""
    from ai_edge_quantizer import calibrator, recipe_manager
    from ai_edge_quantizer.utils import qsv_utils

    signature, input_name = _single_signature_input(fp32_path)
    manager = recipe_manager.RecipeManager()
    manager.load_quantization_recipe(quantization_recipe)
    # Global extrema are essential for scalar regression outputs: the default
    # moving average would collapse their range towards the average prediction.
    calibration = calibrator.Calibrator(
        fp32_path.read_bytes(), num_threads=4,
        qsv_update_func=qsv_utils.min_max_update,
    )
    calibration.calibrate({signature: [
        {input_name: sample[None, ...]} for sample in calibration_inputs
    ]}, manager)
    return calibration.get_model_qsvs()


def _inline_constant_buffers(model_bytes: bytes) -> bytes:
    """Repack the flatbuffer so that all constants live inside it.

    The interface rewriter cannot preserve external buffer offsets. Inline
    constants before rewriting so larger LiteRT exports remain valid.
    """
    import flatbuffers
    from ai_edge_litert.tools import flatbuffer_utils

    packed_model = flatbuffer_utils.read_model_from_bytearray(model_bytes)
    builder = flatbuffers.Builder(0)
    builder.Finish(packed_model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())


def _write_with_int8_io(model_bytes: bytes, output_path: Path) -> None:
    """Switch the model's float input/output to INT8, validate it, then write it."""
    from ai_edge_litert import _pywrap_modify_model_interface
    from ai_edge_litert import schema_py_generated as schema

    with tempfile.TemporaryDirectory() as temporary:
        intermediate = Path(temporary) / 'quantized.tflite'
        candidate = Path(temporary) / 'full_int8.tflite'
        intermediate.write_bytes(model_bytes)
        status = _pywrap_modify_model_interface.modify_model_interface(
            str(intermediate), str(candidate), schema.TensorType.INT8, schema.TensorType.INT8,
        )
        if status != 0:
            raise RuntimeError("Could not convert model input/output to INT8")
        validate_full_int8(candidate)
        output_path.write_bytes(candidate.read_bytes())


def validate_full_int8(path: Path) -> None:
    """Inspect every subgraph, not just the interpreter's primary graph."""
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter

    model = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    allowed = {schema.TensorType.INT8, schema.TensorType.INT32,
               schema.TensorType.INT64, schema.TensorType.BOOL}
    for graph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(graph_index)
        for index in range(graph.TensorsLength()):
            tensor = graph.Tensors(index)
            if tensor.Type() not in allowed:
                raise ValueError(f"Not full_int8: tensor {tensor.Name()!r} has type {tensor.Type()}")
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    for detail in interpreter.get_input_details() + interpreter.get_output_details():
        if detail['dtype'] != np.int8 or detail['quantization'][0] <= 0:
            raise ValueError("full_int8 requires quantized INT8 input and output")


def evaluate_tflite(path: Path, inputs: np.ndarray, targets: np.ndarray) -> float:
    """Compute global MAE in target units by executing the serialized artifact."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(inputs) == 0 or len(inputs) != len(targets):
        raise ValueError("Evaluation requires equally sized, nonempty inputs and targets")

    interpreter, input_detail, output_detail = _load_single_io_interpreter(path)
    total_error = 0.0
    for index, (sample, target) in enumerate(zip(inputs, targets, strict=True), start=1):
        prediction = _predict_one(interpreter, input_detail, output_detail, sample)
        error = abs(prediction - target)
        if not np.isfinite(error):
            raise ValueError("Non-finite prediction or target during evaluation")
        total_error += error
        if index % 10000 == 0:
            print(f"Eval {path}: {index:,}/{len(inputs):,} Samples", flush=True)

    mae = total_error / len(inputs)
    print(f"Globaler MAE auf Eval-Set [{path}]: {mae:.4f} m ({len(inputs):,} Samples)", flush=True)
    return mae


def _load_single_io_interpreter(path: Path) -> tuple[Any, dict, dict]:
    """Load a ``.tflite`` model with exactly one input and one output tensor."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("Evaluation currently requires one input and one scalar output")
    return interpreter, input_details[0], output_details[0]


def _predict_one(interpreter: Any, input_detail: dict, output_detail: dict, sample: np.ndarray) -> float:
    """Run one float sample through the model and return the prediction as float."""
    interpreter.set_tensor(input_detail['index'], _to_model_input(sample[None, ...], input_detail))
    interpreter.invoke()
    prediction = _to_float_output(interpreter.get_tensor(output_detail['index']), output_detail)
    if prediction.size != 1:
        raise ValueError("Global distance MAE requires one scalar prediction per sample")
    return prediction.item()


def _to_model_input(value: np.ndarray, detail: dict) -> np.ndarray:
    """Quantize a float input (``value / scale + zero_point``) if the model expects integers."""
    if np.issubdtype(detail['dtype'], np.integer):
        scale, zero = detail['quantization']
        if scale <= 0:
            raise ValueError("Missing input quantization scale")
        limits = np.iinfo(detail['dtype'])
        value = np.clip(np.rint(value / scale + zero), limits.min, limits.max)
    return value.astype(detail['dtype'])


def _to_float_output(value: np.ndarray, detail: dict) -> np.ndarray:
    """Dequantize an integer output (``(value - zero_point) * scale``) back to float."""
    value = value.astype(np.float64)
    if np.issubdtype(detail['dtype'], np.integer):
        scale, zero = detail['quantization']
        if scale <= 0:
            raise ValueError("Missing output quantization scale")
        value = (value - zero) * scale
    return value
