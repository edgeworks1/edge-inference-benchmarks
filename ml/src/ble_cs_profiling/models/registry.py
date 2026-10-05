"""Central model registry shared by training and deployment.

Checkpoints only store the model name (+ kwargs), never the class itself.
``build_model(name)`` turns that name back into an architecture.
A variant is one line: same class, different preset arguments.
"""

from functools import partial
from typing import Any

from torch import nn

from ble_cs_profiling.models.mlp.edge_bench_mlp import EdgeBenchMLP
from ble_cs_profiling.models.mobilenet_v3.mobilenet_v3_micro_2d import MobileNetV3Micro2D
from ble_cs_profiling.models.mobilenet_v3.mobilenet_v3_small_2d import MobileNetV3Small2D
from ble_cs_profiling.models.mobilenet_v3.mobilenet_v3_small_2d_bad_simd import (
    MobileNetV3Small2DBadSimd,
)

# Alle Modelle bleiben registriert, sonst lassen sich ihre Checkpoints nicht mehr laden.
MODELS: dict[str, partial[nn.Module]] = {
    "mobilenet_v3_small_2d": partial(MobileNetV3Small2D),
    "mobilenet_v3_small_2d_relu": partial(MobileNetV3Small2D, block_activation="relu"),
    "mobilenet_v3_small_2d_bad_simd": partial(MobileNetV3Small2DBadSimd),
    "mobilenet_v3_micro_2d": partial(MobileNetV3Micro2D),
    "edge_bench_mlp": partial(EdgeBenchMLP),
}


def build_model(name: str, model_kwargs: dict[str, Any] | None = None) -> nn.Module:
    """Construct a registered model; ``model_kwargs`` override its preset arguments."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODELS))}")
    return MODELS[name](**(model_kwargs or {}))
