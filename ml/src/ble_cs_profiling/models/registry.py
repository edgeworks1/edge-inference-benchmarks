"""Central model registry shared by training and deployment."""

from collections.abc import Callable
from typing import Any

from torch import nn

from ble_cs_profiling.models.mobilenet_v3.mobilenet_v3_micro_2d import MobileNetV3Micro2D
from ble_cs_profiling.models.mobilenet_v3.mobilenet_v3_small_2d import MobileNetV3Small2D
from ble_cs_profiling.models.mobilenet_v3.mobilenet_v3_small_2d_bad_simd import (
    MobileNetV3Small2DBadSimd,
)


ModelBuilder = Callable[..., nn.Module]

# Alle Modelle bleiben registriert: Checkpoints speichern nur den Namen und
# lassen sich sonst nicht mehr laden.
MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "mobilenet_v3_small_2d": MobileNetV3Small2D,
    "mobilenet_v3_small_2d_relu": MobileNetV3Small2D,
    "mobilenet_v3_small_2d_bad_simd": MobileNetV3Small2DBadSimd,
    "mobilenet_v3_micro_2d": MobileNetV3Micro2D,
}

DEFAULT_MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "mobilenet_v3_small_2d": {},
    "mobilenet_v3_small_2d_relu": {"block_activation": "relu"},
    "mobilenet_v3_small_2d_bad_simd": {},
    "mobilenet_v3_micro_2d": {
        "image_height": 4,
        "image_width": 75,
        "num_outputs": 1,
    },
}


def build_model(name: str, model_kwargs: dict[str, Any] | None = None) -> nn.Module:
    """Construct a registered model, optionally overriding its defaults."""
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}") from error

    kwargs = DEFAULT_MODEL_KWARGS[name] | (model_kwargs or {})
    return builder(**kwargs)