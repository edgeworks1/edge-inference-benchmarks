"""Train a registered model with the fixed settings from ``config.py`` and save a checkpoint.

Entry point: ``edge-ib train`` (see ``edge-ib train --help``). Training is deterministic:
the same model, data and settings give bit-identical weights on the same hardware.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path

# Set before importing torch: cuBLAS reads its workspace setting when CUDA initializes and
# needs it for deterministic matrix multiplications on the GPU. PYTHONHASHSEED is set in
# set_seed() from cfg.SEED.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edge_ib import config as cfg
from edge_ib.checkpoint import save_checkpoint
from edge_ib.data import load_eval_set, load_training_set
from edge_ib.models import build_model


@dataclass
class TrainingResult:
    """Trained model plus its loss curve and evaluation metrics."""

    model: nn.Module
    epoch_losses: list[float]
    mae: float
    predictions: np.ndarray


def train(model_name: str, checkpoint_path: Path) -> TrainingResult:
    """Build the registered model ``model_name``, train and evaluate it, save a checkpoint."""
    set_seed(cfg.SEED)  # before build_model(): makes the weight initialization deterministic
    model = build_model(model_name)

    result = fit_and_evaluate(model)
    # Variants are registry entries with preset arguments, so no extra model kwargs are stored
    save_checkpoint(checkpoint_path, result.model, model_name, {}, result.mae, result.epoch_losses)
    return result


def fit_and_evaluate(model: nn.Module) -> TrainingResult:
    """Train ``model`` on the training subset, then compute its MAE on the full eval set."""
    # Re-seed so that this function is also reproducible on its own
    set_seed(cfg.SEED)
    device = select_device()

    train_features, train_targets = load_training_set()
    eval_features, eval_targets = load_eval_set(cfg.FEATURES_RAW_IQ_ALL, cfg.TARGET)
    train_loader = make_train_loader(train_features, train_targets)

    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    print(f"Training {model.__class__.__name__} on {device} ({cfg.NUM_FEATURES} features)")
    epoch_losses = []
    for epoch in range(1, cfg.EPOCHS + 1):
        mean_loss = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch:02d}/{cfg.EPOCHS:02d} - Loss: {mean_loss:.4f}")

    predictions = predict(model, eval_features, device)
    mae = float(mean_absolute_error(eval_targets, predictions))
    print(f"MAE on eval set: {mae:.4f} m")
    return TrainingResult(model, epoch_losses, mae, predictions)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and switch PyTorch to deterministic algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cuDNN would otherwise pick the fastest (and possibly non-deterministic) kernel
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def select_device() -> torch.device:
    """Use the GPU if one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_train_loader(features: np.ndarray, targets: np.ndarray) -> DataLoader:
    """Wrap the training arrays in a shuffling DataLoader with a reproducible batch order."""
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets).unsqueeze(1))
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one pass over the training data and return the mean batch loss."""
    model.train()
    total_loss = 0.0
    for batch_features, batch_targets in loader:
        batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(batch_features), batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """Return the model's predictions for ``features`` as a flat array, batch by batch."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=cfg.BATCH_SIZE)
    with torch.inference_mode():
        batches = [model(batch.to(device)).cpu() for (batch,) in loader]
    return torch.cat(batches).numpy().reshape(-1)
