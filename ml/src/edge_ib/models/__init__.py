"""Model architectures and the registry that training and export share."""

from edge_ib.models.registry import MODELS, build_model

__all__ = ["MODELS", "build_model"]
