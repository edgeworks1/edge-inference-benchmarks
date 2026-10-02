"""Generic training and evaluation runner for profiling models.

Change the constants in ``SCRIPT CONFIGURATION`` and run
``python -m ble_cs_profiling.train``.
"""

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Muss vor bestimmten CUDA/PyTorch-Imports gesetzt sein für reproduzierbare GEMM-Calls
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ble_cs_profiling import config as cfg
from ble_cs_profiling.models import build_model


# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
MODEL_NAME = "mobilenet_v3_small_2d"
MODEL_KWARGS: dict[str, Any] = {}
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
TRAIN_SAMPLE_RATE = 0.1  # 0.1 = random 10 %, 1.0 = complete training set
SEED = 42
NUM_WORKERS = 0
DEVICE: str | None = None  # None = CUDA when available, otherwise CPU

CHECKPOINT_PATH = cfg.CHECKPOINT_DIR / f"{MODEL_NAME}_0.1.pt"


@dataclass(frozen=True)
class RunConfig:
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    train_sample_rate: float = 1.0
    seed: int = 42
    num_workers: int = 0


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


def seed_worker(worker_id: int) -> None:
    """Worker init function to guarantee reproducibility across DataLoader sub-processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(
    train_sample_rate: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Load IQ features and optionally draw a reproducible training subset."""
    if not 0 < train_sample_rate <= 1:
        raise ValueError("train_sample_rate must be greater than 0 and at most 1")

    train_df = pd.read_parquet(cfg.TRAIN_DATA_PATH)
    eval_df = pd.read_parquet(cfg.EVAL_DATA_PATH)
    complete_train_size = len(train_df)
    if train_sample_rate < 1.0:
        sample_size = max(1, round(complete_train_size * train_sample_rate))
        train_df = train_df.sample(n=sample_size, random_state=seed)

    print(
        f"Trainingsdaten: {len(train_df):,}/{complete_train_size:,} Samples "
        f"({len(train_df) / complete_train_size:.1%})"
    )
    x_train = torch.tensor(
        train_df[cfg.FEATURES_RAW_IQ_ALL].values, dtype=torch.float32
    )
    y_train = torch.tensor(
        train_df[cfg.TARGET].values, dtype=torch.float32
    ).unsqueeze(1)
    x_eval = torch.tensor(
        eval_df[cfg.FEATURES_RAW_IQ_ALL].values, dtype=torch.float32
    )
    y_eval = eval_df[cfg.TARGET].to_numpy()
    return x_train, y_train, x_eval, y_eval


def run_model(
    model: nn.Module,
    run_config: RunConfig | None = None,
    device: torch.device | str | None = None,
) -> RunResult:
    """Train and evaluate any model mapping flat IQ batches to ``(batch, 1)``."""
    settings = run_config or RunConfig()
    if settings.epochs < 1:
        raise ValueError("epochs must be at least 1")

    # Seed vor DataLoader- und Trainingsstart nochmals synchronisieren
    set_seed(settings.seed)

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    x_train, y_train, x_eval, y_eval = load_data(
        train_sample_rate=settings.train_sample_rate,
        seed=settings.seed,
    )

    train_generator = torch.Generator().manual_seed(settings.seed)
    eval_generator = torch.Generator().manual_seed(settings.seed)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    eval_loader = DataLoader(
        TensorDataset(x_eval),
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        generator=eval_generator,
        worker_init_fn=seed_worker,
    )

    model = model.to(selected_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    epoch_losses: list[float] = []

    print(
        f"Starte Training von {model.__class__.__name__} auf {selected_device} "
        f"({cfg.NUM_FEATURES} Features)"
    )
    for epoch in range(settings.epochs):
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
        print(f"Epoch {epoch + 1:02d}/{settings.epochs:02d} - Loss: {mean_loss:.4f}")

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
    settings: RunConfig,
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
        "run_config": asdict(settings),
        "eval_mae": mae,
        "input_shape": [1, cfg.NUM_FEATURES],
        "feature_names": list(cfg.FEATURES_RAW_IQ_ALL),
        "target": cfg.TARGET,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint gespeichert: {path} ({path.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    settings = RunConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        train_sample_rate=TRAIN_SAMPLE_RATE,
        seed=SEED,
        num_workers=NUM_WORKERS,
    )

    # WICHTIG: Seed vor build_model() setzen, damit die Gewichtsinitialisierung deterministisch ist
    set_seed(settings.seed)

    model = build_model(MODEL_NAME, MODEL_KWARGS)
    result = run_model(model, settings, DEVICE)

    save_checkpoint(
        CHECKPOINT_PATH,
        result.model,
        MODEL_NAME,
        MODEL_KWARGS,
        settings,
        result.mae,
    )


if __name__ == "__main__":
    main()