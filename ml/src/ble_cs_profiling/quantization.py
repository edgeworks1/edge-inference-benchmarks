"""Model-independent static INT8 PTQ and evaluation for single-input regressors."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def load_features(path: Path, feature_names: list[str]) -> np.ndarray:
    return pd.read_parquet(path, columns=feature_names).to_numpy(dtype=np.float32)


def quantize_full_int8(
    fp32_path: Path, output_path: Path, calibration_inputs: np.ndarray,
) -> Path:
    """Calibrate a LiteRT graph and reject float fallback, including float I/O."""
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_litert import _pywrap_modify_model_interface
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_quantizer import quantizer, recipe, calibrator, recipe_manager
    from ai_edge_quantizer.utils import qsv_utils

    if len(calibration_inputs) == 0 or not np.isfinite(calibration_inputs).all():
        raise ValueError("Calibration inputs must be nonempty and finite")
    interpreter = Interpreter(model_path=str(fp32_path))
    signatures = interpreter.get_signature_list()
    if len(signatures) != 1:
        raise ValueError("PTQ currently requires exactly one model signature")
    signature, details = next(iter(signatures.items()))
    if len(details['inputs']) != 1 or len(details['outputs']) != 1:
        raise ValueError("PTQ currently requires one input and one output")
    input_name = details['inputs'][0]
    quantization_recipe = recipe.static_wi8_ai8()
    qt = quantizer.Quantizer(fp32_path, quantization_recipe)
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
    result = qt.quantize(calibration.get_model_qsvs())
    with tempfile.TemporaryDirectory() as temporary:
        intermediate = Path(temporary) / 'quantized.tflite'
        candidate = Path(temporary) / 'full_int8.tflite'
        # The interface rewriter cannot preserve external buffer offsets. Inline
        # constants before rewriting so larger LiteRT exports remain valid.
        import flatbuffers
        from ai_edge_litert.tools import flatbuffer_utils

        packed_model = flatbuffer_utils.read_model_from_bytearray(result.quantized_model)
        builder = flatbuffers.Builder(0)
        builder.Finish(packed_model.Pack(builder), file_identifier=b'TFL3')
        intermediate.write_bytes(bytes(builder.Output()))
        status = _pywrap_modify_model_interface.modify_model_interface(
            str(intermediate), str(candidate), schema.TensorType.INT8, schema.TensorType.INT8,
        )
        if status != 0:
            raise RuntimeError("Could not convert model input/output to INT8")
        validate_full_int8(candidate)
        output_path.write_bytes(candidate.read_bytes())
    print(f"TFLite full_int8 gespeichert: {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    return output_path


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
    from ai_edge_litert.interpreter import Interpreter

    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(inputs) == 0 or len(inputs) != len(targets):
        raise ValueError("Evaluation requires equally sized, nonempty inputs and targets")
    interpreter = Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("Evaluation currently requires one input and one scalar output")
    inp, out = input_details[0], output_details[0]
    total_error = 0.0
    for index, (sample, target) in enumerate(zip(inputs, targets), start=1):
        value = sample[None, ...]
        if np.issubdtype(inp['dtype'], np.integer):
            scale, zero = inp['quantization']
            if scale <= 0:
                raise ValueError("Missing input quantization scale")
            limits = np.iinfo(inp['dtype'])
            value = np.clip(np.rint(value / scale + zero), limits.min, limits.max)
        interpreter.set_tensor(inp['index'], value.astype(inp['dtype']))
        interpreter.invoke()
        prediction = interpreter.get_tensor(out['index']).astype(np.float64)
        if np.issubdtype(out['dtype'], np.integer):
            scale, zero = out['quantization']
            if scale <= 0:
                raise ValueError("Missing output quantization scale")
            prediction = (prediction - zero) * scale
        if prediction.size != 1:
            raise ValueError("Global distance MAE requires one scalar prediction per sample")
        error = abs(prediction.item() - target)
        if not np.isfinite(error):
            raise ValueError("Non-finite prediction or target during evaluation")
        total_error += error
        if index % 10000 == 0:
            print(f"Eval {path.stem}: {index:,}/{len(inputs):,} Samples", flush=True)
    mae = total_error / len(inputs)
    print(f"Globaler MAE auf Eval-Set [{path.stem}]: {mae:.4f} m ({len(inputs):,} Samples)", flush=True)
    return mae
