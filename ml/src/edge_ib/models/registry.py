"""Central model registry shared by training and export.

Checkpoints store only a model *name* and its keyword arguments, never the class itself.
``build_model(name)`` turns that name back into an architecture, so a checkpoint stays
loadable as long as its name is registered here.

Adding a model: implement it as an ``nn.Module`` and add one line to ``MODELS``.
A variant is also one line: the same class with different preset arguments.
"""

from functools import partial
from typing import Any

from torch import nn

from edge_ib.models.mlp.edge_bench_mlp import EdgeBenchMLP
from edge_ib.models.mobilenet_v3.mobilenet_v3_micro_2d import MobileNetV3Micro2D
from edge_ib.models.mobilenet_v3.mobilenet_v3_small_2d import MobileNetV3Small2D
from edge_ib.models.mobilenet_v3.mobilenet_v3_small_2d_bad_simd import (
    MobileNetV3Small2DBadSimd,
)

# Never remove an entry: checkpoints trained with it could no longer be loaded.
MODELS: dict[str, partial[nn.Module]] = {
    "mobilenet_v3_small_2d": partial(MobileNetV3Small2D),
    "mobilenet_v3_small_2d_relu": partial(MobileNetV3Small2D, block_activation="relu"),
    "mobilenet_v3_small_2d_bad_simd": partial(MobileNetV3Small2DBadSimd),
    "mobilenet_v3_micro_2d": partial(MobileNetV3Micro2D),
    "edge_bench_mlp": partial(EdgeBenchMLP),
}


def build_model(name: str, model_kwargs: dict[str, Any] | None = None) -> nn.Module:
    """Build a fresh, untrained instance of the registered model ``name``.

    ``model_kwargs`` override the registered preset arguments.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODELS))}")
    return MODELS[name](**(model_kwargs or {}))
