"""Generate C/C++ sources consumed by the firmware projects.

The emitted symbols (``g_model_data``, ``TEST_INPUT_SAMPLES`` ...) are the contract
with the firmware: file and variable names stay fixed so that swapping a model only
recompiles one translation unit (no ``west build --pristine`` needed).
"""

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_SYMBOL = "model_data"
MODEL_BASE_NAME = "model_data"
HEX_BYTES_PER_LINE = 12


def write_model_sources(tflite_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Embed a ``.tflite`` flatbuffer as ``model_data.h`` / ``model_data.cc``."""
    if not tflite_path.is_file():
        raise FileNotFoundError(f"TFLite-Datei nicht gefunden: {tflite_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = tflite_path.read_bytes()
    data_len = len(data)
    formatted_hex = ",\n".join(
        "    " + ", ".join(f"0x{byte:02x}" for byte in data[i:i + HEX_BYTES_PER_LINE])
        for i in range(0, data_len, HEX_BYTES_PER_LINE)
    )

    header_path = output_dir / f"{MODEL_BASE_NAME}.h"
    header_path.write_text(
        "#pragma once\n\n"
        "#include <cstddef>\n"
        "#include <cstdint>\n\n"
        f"// Quelle: {tflite_path.name} ({data_len:,} Bytes / {data_len / 1024:.2f} KB)\n"
        f"extern const unsigned char g_{MODEL_SYMBOL}[];\n"
        f"extern const size_t g_{MODEL_SYMBOL}_len;\n"
    )

    source_path = output_dir / f"{MODEL_BASE_NAME}.cc"
    source_path.write_text(
        f'#include "{MODEL_BASE_NAME}.h"\n\n'
        "// Liegt dank alignas(16) und const direkt im Flash (ROM)\n"
        f"alignas(16) const unsigned char g_{MODEL_SYMBOL}[] = {{\n"
        f"{formatted_hex}"
        "\n};\n\n"
        f"const size_t g_{MODEL_SYMBOL}_len = {data_len};\n"
    )
    print(f"Modell-Sources erzeugt: {header_path}, {source_path} ({data_len / 1024:.2f} KB)")
    return header_path, source_path


def write_test_samples(
    eval_path: Path,
    feature_names: list[str],
    target: str,
    output_dir: Path,
    num_samples: int = 10,
    seed: int = 34,
) -> tuple[Path, Path]:
    """Draw reproducible eval samples as ``data.hpp`` (MCU) and ``.npy`` (Jetson/PC)."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(eval_path, columns=[*feature_names, target])
    sampled = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
    samples = sampled[feature_names].to_numpy(dtype=np.float32)
    ground_truth = sampled[target].to_numpy(dtype=np.float32)

    rows = "".join(
        "    {" + ", ".join(f"{value:.6f}f" for value in row) + "},\n" for row in samples
    )
    gt_values = ", ".join(f"{value:.4f}f" for value in ground_truth)

    header_path = output_dir / "data.hpp"
    header_path.write_text(
        "#pragma once\n\n"
        "#include <cstddef>\n\n"
        f"#define NUM_TEST_SAMPLES {num_samples}\n"
        f"#define NUM_INPUT_FEATURES {samples.shape[1]}\n\n"
        "// Liegt dank const und alignas(16) sauber im Flash (ROM)\n"
        "alignas(16) const float TEST_INPUT_SAMPLES[NUM_TEST_SAMPLES][NUM_INPUT_FEATURES] = {\n"
        f"{rows}"
        "};\n\n"
        # Ground Truth zur schnellen Verifikation der Inferenz
        f"alignas(16) const float TEST_GROUND_TRUTH[NUM_TEST_SAMPLES] = {{{gt_values}}};\n"
    )

    npy_path = output_dir / f"test_samples_{num_samples}.npy"
    np.save(npy_path, samples)
    print(f"Test-Samples erzeugt: {header_path}, {npy_path} (Seed {seed}, Shape {samples.shape})")
    return header_path, npy_path
