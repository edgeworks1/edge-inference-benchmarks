"""Checkpoints: the hand-off between training and export.

A checkpoint stores the trained weights plus everything needed to rebuild and evaluate the
model later: registry name, constructor arguments, training settings, input shape and the
dataset schema. Export never needs the training code state of back then.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ble_cs_profiling import config as cfg
from ble_cs_profiling.models import build_model

FORMAT_VERSION = 1
REQUIRED_KEYS = {"model_name", "model_kwargs", "model_state_dict"}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    model_kwargs: dict[str, Any],
    eval_mae: float,
) -> None:
    """Save the trained weights together with all metadata needed to rebuild the model."""
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "model_state_dict": _cpu_state_dict(model),
        "run_config": _training_settings(),
        "eval_mae": eval_mae,
        "input_shape": [1, cfg.NUM_FEATURES],
        "feature_names": list(cfg.FEATURES_RAW_IQ_ALL),
        "target": cfg.TARGET,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path} ({path.stat().st_size / 1024:.1f} KiB)")


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from a checkpoint and load its weights; returns ``(model, checkpoint)``.

    The model is always loaded on the CPU in eval mode: export must not depend on a GPU.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}\nRun `ble-cs train` first.")

    # weights_only: refuses to unpickle arbitrary objects, only tensors and plain data
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing = REQUIRED_KEYS - checkpoint.keys()
    if missing:
        raise ValueError(f"Invalid checkpoint {path}, missing keys: {sorted(missing)}")

    model = build_model(checkpoint["model_name"], checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.cpu().eval(), checkpoint


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy all weights to the CPU so the checkpoint loads on machines without a GPU."""
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def _training_settings() -> dict[str, Any]:
    """Collect the fixed training settings from ``config.py`` for traceability."""
    return {
        "seed": cfg.SEED,
        "epochs": cfg.EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
        "learning_rate": cfg.LEARNING_RATE,
        "train_sample_rate": cfg.TRAIN_SAMPLE_RATE,
    }
