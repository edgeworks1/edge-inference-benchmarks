"""Static post-training quantization (PTQ) to INT8 for TFLite and ONNX models.

Both formats are calibrated on the same representative samples: they run through the
FP32 model, the observed value range of every tensor defines its INT8 scale and zero point.

- TFLite: full INT8, including input and output. This is what TFLite Micro on an MCU needs.
- ONNX:   INT8 weights and activations in QDQ format, float32 input and output.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# TFLite: full INT8
# ============================================================
def quantize_tflite_full_int8(
    fp32_path: Path, output_path: Path, calibration_inputs: np.ndarray
) -> Path:
    """Quantize an FP32 TFLite model to full INT8 and reject any remaining float tensor."""
    from ai_edge_quantizer import quantizer, recipe

    _check_calibration_inputs(calibration_inputs)
    int8_recipe = recipe.static_wi8_ai8()  # INT8 weights and INT8 activations everywhere
    value_ranges = _calibrate_tflite(fp32_path, int8_recipe, calibration_inputs)
    quantized = quantizer.Quantizer(fp32_path, int8_recipe).quantize(value_ranges).quantized_model
    _write_with_int8_io(_inline_constant_buffers(quantized), output_path)
    return output_path


def validate_full_int8(path: Path) -> None:
    """Raise if any tensor in any subgraph is float, or if input/output are not quantized INT8."""
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter

    # Integer types that may remain: INT32 biases, INT64/BOOL shape and index tensors
    allowed = {
        schema.TensorType.INT8,
        schema.TensorType.INT32,
        schema.TensorType.INT64,
        schema.TensorType.BOOL,
    }
    model = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    for graph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(graph_index)
        for tensor_index in range(graph.TensorsLength()):
            tensor = graph.Tensors(tensor_index)
            if tensor.Type() not in allowed:
                raise ValueError(
                    f"Not full INT8: tensor {tensor.Name()!r} has type {tensor.Type()}"
                )

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    for detail in interpreter.get_input_details() + interpreter.get_output_details():
        if detail["dtype"] != np.int8 or detail["quantization"][0] <= 0:
            raise ValueError("Full INT8 requires a quantized INT8 input and output")


def _calibrate_tflite(
    fp32_path: Path, quantization_recipe: Any, calibration_inputs: np.ndarray
) -> dict:
    """Run the calibration samples through the FP32 model and record every tensor's value range."""
    from ai_edge_quantizer import calibrator, recipe_manager
    from ai_edge_quantizer.utils import qsv_utils

    signature, input_name = _single_tflite_input(fp32_path)
    manager = recipe_manager.RecipeManager()
    manager.load_quantization_recipe(quantization_recipe)
    # Keep the global min/max: the default moving average would shrink the range of a
    # scalar regression output towards the average prediction and clip extreme distances.
    calibration = calibrator.Calibrator(
        fp32_path.read_bytes(),
        num_threads=4,
        qsv_update_func=qsv_utils.min_max_update,
    )
    samples = [{input_name: sample[None, ...]} for sample in calibration_inputs]
    calibration.calibrate({signature: samples}, manager)
    return calibration.get_model_qsvs()


def _single_tflite_input(model_path: Path) -> tuple[str, str]:
    """Return ``(signature name, input name)`` of a model with exactly one input and one output."""
    from ai_edge_litert.interpreter import Interpreter

    signatures = Interpreter(model_path=str(model_path)).get_signature_list()
    if len(signatures) != 1:
        raise ValueError("Quantization requires exactly one model signature")
    signature, details = next(iter(signatures.items()))
    if len(details["inputs"]) != 1 or len(details["outputs"]) != 1:
        raise ValueError("Quantization requires exactly one input and one output")
    return signature, details["inputs"][0]


def _inline_constant_buffers(model_bytes: bytes) -> bytes:
    """Repack the flatbuffer so that all constant buffers are stored inside it.

    The quantizer may store large constants after the flatbuffer and point to them by offset;
    the input/output rewriter in the next step cannot preserve such offsets.
    """
    import flatbuffers
    from ai_edge_litert.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model_from_bytearray(model_bytes)
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def _write_with_int8_io(model_bytes: bytes, output_path: Path) -> None:
    """Switch the model's float input and output to INT8, validate the result, then write it."""
    from ai_edge_litert import _pywrap_modify_model_interface
    from ai_edge_litert import schema_py_generated as schema

    # The rewriter only works on files, so it runs on temporary copies
    with tempfile.TemporaryDirectory() as temporary:
        float_io_path = Path(temporary) / "float_io.tflite"
        int8_io_path = Path(temporary) / "int8_io.tflite"
        float_io_path.write_bytes(model_bytes)
        status = _pywrap_modify_model_interface.modify_model_interface(
            str(float_io_path),
            str(int8_io_path),
            schema.TensorType.INT8,
            schema.TensorType.INT8,
        )
        if status != 0:
            raise RuntimeError("Could not convert the model input/output to INT8")
        validate_full_int8(int8_io_path)
        output_path.write_bytes(int8_io_path.read_bytes())


# ============================================================
# ONNX: INT8 QDQ
# ============================================================
def quantize_onnx_int8_qdq(
    fp32_path: Path, output_path: Path, calibration_inputs: np.ndarray
) -> Path:
    """Quantize an FP32 ONNX model statically to INT8 (QDQ format, per-channel weights, float I/O)."""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    _check_calibration_inputs(calibration_inputs)
    with tempfile.TemporaryDirectory() as temporary:
        prepared_path = Path(temporary) / "prepared.onnx"
        # ONNX Runtime's symbolic shape inference fails on the MobileNet conv shapes,
        # standard ONNX shape inference and graph optimization are sufficient.
        quant_pre_process(str(fp32_path), str(prepared_path), skip_symbolic_shape=True)
        quantize_static(
            str(prepared_path),
            str(output_path),
            _OnnxCalibrationReader(calibration_inputs, _onnx_input_name(fp32_path)),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    return output_path


class _OnnxCalibrationReader:
    """Feeds the calibration samples one by one to ONNX Runtime's static quantization."""

    def __init__(self, calibration_inputs: np.ndarray, input_name: str) -> None:
        self._samples = iter(calibration_inputs)
        self._input_name = input_name

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next sample as a batch of one, or None when all samples are used."""
        sample = next(self._samples, None)
        if sample is None:
            return None
        return {self._input_name: np.asarray(sample[None, ...], dtype=np.float32)}


def _onnx_input_name(model_path: Path) -> str:
    """Return the name of the single input of an ONNX model."""
    import onnx

    inputs = onnx.load(str(model_path), load_external_data=False).graph.input
    if len(inputs) != 1:
        raise ValueError("Quantization requires exactly one model input")
    return inputs[0].name


# ============================================================
# Shared
# ============================================================
def _check_calibration_inputs(calibration_inputs: np.ndarray) -> None:
    """Reject empty or non-finite calibration data: it would silently produce broken scales."""
    if len(calibration_inputs) == 0 or not np.isfinite(calibration_inputs).all():
        raise ValueError("Calibration inputs must be nonempty and finite")
