"""Generate the C/C++ sources that embed a model and test inputs into the firmware.

A microcontroller has no file system, so the ``.tflite`` file becomes a constant byte array
that the linker places in flash. The emitted file and symbol names (``g_model_data``,
``TEST_INPUT_SAMPLES`` ...) are the contract with the firmware: they never change, so
swapping a model only recompiles one file.
"""

import hashlib
from pathlib import Path

import numpy as np

MODEL_SYMBOL = "g_model_data"
MODEL_FILE_STEM = "model_data"
SAMPLES_HEADER = "data.hpp"
BYTES_PER_LINE = 12


def write_model_sources(tflite_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Write ``model_data.h`` / ``model_data.cc`` with the model bytes and their sha256.

    The firmware prints the embedded sha256, which links every measurement on the device
    to exactly one exported model file.
    """
    if not tflite_path.is_file():
        raise FileNotFoundError(f"TFLite file not found: {tflite_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_bytes = tflite_path.read_bytes()
    size = len(model_bytes)

    header_path = output_dir / f"{MODEL_FILE_STEM}.h"
    header_path.write_text(
        "#pragma once\n\n"
        "#include <cstddef>\n"
        "#include <cstdint>\n\n"
        f"// Source: {tflite_path} ({size:,} bytes / {size / 1024:.2f} KiB)\n"
        f"extern const unsigned char {MODEL_SYMBOL}[];\n"
        f"extern const size_t {MODEL_SYMBOL}_len;\n"
        f"extern const char {MODEL_SYMBOL}_sha256[];\n"
    )

    source_path = output_dir / f"{MODEL_FILE_STEM}.cc"
    source_path.write_text(
        f'#include "{MODEL_FILE_STEM}.h"\n\n'
        "// const: placed in flash, not RAM.\n"
        "// alignas(16): TFLite Micro reads the flatbuffer in place.\n"
        f"alignas(16) const unsigned char {MODEL_SYMBOL}[] = {{\n"
        f"{_c_byte_rows(model_bytes)}"
        "\n};\n\n"
        f"const size_t {MODEL_SYMBOL}_len = {size};\n"
        f'const char {MODEL_SYMBOL}_sha256[] = "{hashlib.sha256(model_bytes).hexdigest()}";\n'
    )
    print(f"Generated model sources: {header_path}, {source_path} ({size / 1024:.2f} KiB)")
    return header_path, source_path


def write_test_samples(
    features: np.ndarray, targets: np.ndarray, output_dir: Path
) -> tuple[Path, Path]:
    """Write test inputs as ``data.hpp`` (firmware) and ``test_samples_<n>.npy`` (Jetson/PC).

    The ground truth is included so the firmware output can be checked by eye.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples, num_features = features.shape

    ground_truth = _c_floats(targets, decimals=4)
    header_path = output_dir / SAMPLES_HEADER
    header_path.write_text(
        "#pragma once\n\n"
        "#include <cstddef>\n\n"
        f"#define NUM_TEST_SAMPLES {num_samples}\n"
        f"#define NUM_INPUT_FEATURES {num_features}\n\n"
        "// const: placed in flash, not RAM\n"
        "alignas(16) const float TEST_INPUT_SAMPLES[NUM_TEST_SAMPLES][NUM_INPUT_FEATURES] = {\n"
        f"{_c_float_rows(features)}"
        "};\n\n"
        "// Ground truth for a quick plausibility check of the predictions\n"
        f"alignas(16) const float TEST_GROUND_TRUTH[NUM_TEST_SAMPLES] = {{{ground_truth}}};\n"
    )

    npy_path = output_dir / f"test_samples_{num_samples}.npy"
    np.save(npy_path, features)
    print(f"Generated test samples: {header_path}, {npy_path} (shape {features.shape})")
    return header_path, npy_path


def _c_byte_rows(data: bytes) -> str:
    """Format bytes as indented C hex literals, ``BYTES_PER_LINE`` per line."""
    lines = (
        "    " + ", ".join(f"0x{byte:02x}" for byte in data[start : start + BYTES_PER_LINE])
        for start in range(0, len(data), BYTES_PER_LINE)
    )
    return ",\n".join(lines)


def _c_float_rows(rows: np.ndarray) -> str:
    """Format a 2D array as C initializer rows ``{1.000000f, ...},``, one per line."""
    return "".join(f"    {{{_c_floats(row, decimals=6)}}},\n" for row in rows)


def _c_floats(values: np.ndarray, decimals: int) -> str:
    """Format values as comma-separated C float literals (``1.5000f``)."""
    return ", ".join(f"{value:.{decimals}f}f" for value in values)
