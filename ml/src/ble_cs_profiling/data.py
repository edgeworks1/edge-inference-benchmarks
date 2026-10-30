"""Dataset access: the only module that knows how the data is stored.

Training, export and evaluation get their arrays from here. To use the pipeline with
other data, this module (and the schema in ``config.py``) is what you replace.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ble_cs_profiling import config as cfg


def load_features(path: Path, feature_names: list[str]) -> np.ndarray:
    """Load the model input columns of a Parquet file as a float32 array ``(samples, features)``."""
    return pd.read_parquet(path, columns=feature_names).to_numpy(dtype=np.float32)


def load_targets(path: Path, target: str) -> np.ndarray:
    """Load the regression target of a Parquet file as a float64 array ``(samples,)``."""
    return pd.read_parquet(path, columns=[target])[target].to_numpy()


def load_training_set() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, targets)`` of a reproducible random subset of the training data.

    The subset size is ``TRAIN_SAMPLE_RATE`` of the training set, drawn with ``SEED``.
    """
    columns = [*cfg.FEATURES_RAW_IQ_ALL, cfg.TARGET]
    train_df = pd.read_parquet(cfg.TRAIN_DATA_PATH, columns=columns)
    complete_size = len(train_df)
    if cfg.TRAIN_SAMPLE_RATE < 1.0:
        subset_size = max(1, round(complete_size * cfg.TRAIN_SAMPLE_RATE))
        train_df = train_df.sample(n=subset_size, random_state=cfg.SEED)

    print(
        f"Training data: {len(train_df):,}/{complete_size:,} samples "
        f"({len(train_df) / complete_size:.1%})"
    )
    features = train_df[cfg.FEATURES_RAW_IQ_ALL].to_numpy(dtype=np.float32)
    targets = train_df[cfg.TARGET].to_numpy(dtype=np.float32)
    return features, targets


def load_eval_set(feature_names: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, targets)`` of the complete evaluation set."""
    return load_features(cfg.EVAL_DATA_PATH, feature_names), load_targets(
        cfg.EVAL_DATA_PATH, target
    )


def draw_eval_samples(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, targets)`` of ``num_samples`` reproducible random eval samples.

    Used as fixed test inputs on the device, so that every model sees the same inputs.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    eval_df = pd.read_parquet(cfg.EVAL_DATA_PATH, columns=[*cfg.FEATURES_RAW_IQ_ALL, cfg.TARGET])
    sampled = eval_df.sample(n=num_samples, random_state=seed)
    return sampled[cfg.FEATURES_RAW_IQ_ALL].to_numpy(dtype=np.float32), sampled[
        cfg.TARGET
    ].to_numpy(dtype=np.float32)


def load_calibration_inputs(feature_names: list[str]) -> np.ndarray:
    """Draw ``CALIBRATION_SAMPLES`` reproducible training samples for INT8 calibration.

    Quantization measures the value range of every tensor on these samples, so they
    must be representative of real inputs. All export formats use the same samples.
    """
    training_inputs = load_features(cfg.TRAIN_DATA_PATH, feature_names)
    if len(training_inputs) == 0:
        raise ValueError("Calibration needs a nonempty training set")
    sample_count = min(cfg.CALIBRATION_SAMPLES, len(training_inputs))
    indices = np.random.default_rng(cfg.SEED).choice(
        len(training_inputs), sample_count, replace=False
    )
    return training_inputs[indices]
