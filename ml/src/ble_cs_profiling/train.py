"""Generic training and evaluation runner for profiling models.

Entry point: ``ble-cs train`` (see ``ble-cs train --help``).
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Muss vor bestimmten CUDA/PyTorch-Imports gesetzt sein für reproduzierbare GEMM-Calls
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ble_cs_profiling import config as cfg
from ble_cs_profiling.data import load_eval_set, load_training_set
from ble_cs_profiling.models import build_model


@dataclass
class RunResult:
    model: nn.Module
    epoch_losses: list[float]
    mae: float
    predictions: np.ndarray


def set_seed(seed: int) -> None:
    """Set global seeds across all frameworks and configure deterministic backends."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass


def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Return training features/targets and evaluation features as tensors, eval targets as array."""
    train_features, train_targets = load_training_set()
    eval_features, eval_targets = load_eval_set(cfg.FEATURES_RAW_IQ_ALL, cfg.TARGET)
    x_train = torch.from_numpy(train_features)
    y_train = torch.from_numpy(train_targets).unsqueeze(1)
    x_eval = torch.from_numpy(eval_features)
    return x_train, y_train, x_eval, eval_targets


def run_model(model: nn.Module) -> RunResult:
    """Train and evaluate any model mapping flat IQ batches to ``(batch, 1)``.

    Runs on CUDA when available, otherwise on CPU.
    """
    # Seed vor DataLoader- und Trainingsstart nochmals synchronisieren
    set_seed(cfg.SEED)

    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_eval, y_eval = load_data()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.SEED),  # reproduzierbare Batch-Reihenfolge
    )
    eval_loader = DataLoader(TensorDataset(x_eval), batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = model.to(selected_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    epoch_losses: list[float] = []

    print(
        f"Starte Training von {model.__class__.__name__} auf {selected_device} "
        f"({cfg.NUM_FEATURES} Features)"
    )
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(selected_device)
            batch_y = batch_y.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(train_loader)
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch + 1:02d}/{cfg.EPOCHS:02d} - Loss: {mean_loss:.4f}")

    model.eval()
    prediction_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for (batch_x,) in eval_loader:
            prediction_batches.append(model(batch_x.to(selected_device)).cpu())
    predictions_np = torch.cat(prediction_batches).numpy().reshape(-1)
    mae = float(mean_absolute_error(y_eval, predictions_np))
    print(f"Globaler MAE auf Eval-Set: {mae:.4f} m")
    return RunResult(model, epoch_losses, mae, predictions_np)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    model_kwargs: dict[str, Any],
    mae: float,
) -> None:
    """Persist weights and all information required to rebuild the model."""
    checkpoint = {
        "format_version": 1,
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        # Zur Nachvollziehbarkeit, falls die Konstanten oben mal angepasst werden
        "run_config": {
            "seed": cfg.SEED,
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "learning_rate": cfg.LEARNING_RATE,
            "train_sample_rate": cfg.TRAIN_SAMPLE_RATE,
        },
        "eval_mae": mae,
        "input_shape": [1, cfg.NUM_FEATURES],
        "feature_names": list(cfg.FEATURES_RAW_IQ_ALL),
        "target": cfg.TARGET,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint gespeichert: {path} ({path.stat().st_size / 1024:.1f} KiB)")


def train(
    model_name: str,
    checkpoint_path: Path,
    model_kwargs: dict[str, Any] | None = None,
) -> RunResult:
    """Build a registered model, train/evaluate it and save a checkpoint."""
    model_kwargs = model_kwargs or {}
    # WICHTIG: Seed vor build_model() setzen, damit die Gewichtsinitialisierung deterministisch ist
    set_seed(cfg.SEED)

    model = build_model(model_name, model_kwargs)
    result = run_model(model)
    save_checkpoint(checkpoint_path, result.model, model_name, model_kwargs, result.mae)
    return result
