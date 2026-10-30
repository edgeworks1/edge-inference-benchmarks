"""Central project configuration: paths, dataset schema and fixed training settings.

Every path can be overridden with an environment variable (used by Docker and CI):

    BLE_CS_PROJECT_ROOT    base folder for data/, outputs/ and checkpoints/
    BLE_CS_DATA_DIR        datasets (Parquet)
    BLE_CS_OUTPUT_DIR      exported artifacts (ONNX, TFLite, C sources, results)
    BLE_CS_CHECKPOINT_DIR  trained PyTorch checkpoints

Example: ``BLE_CS_DATA_DIR=/app/data/sample`` trains on the small public data sample.
"""

import os
from pathlib import Path


# ============================================================
# Paths (independent of the current working directory)
# ============================================================
def _find_project_root() -> Path:
    """Return ``$BLE_CS_PROJECT_ROOT`` or the nearest parent folder containing ``pyproject.toml``.

    Installed as a wheel (e.g. in the Docker image) there is no source tree to search,
    so the current working directory is the fallback.
    """
    if root := os.environ.get("BLE_CS_PROJECT_ROOT"):
        return Path(root).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _dir_from_env(variable: str, default: Path) -> Path:
    """Return the folder named by environment ``variable``, or ``default`` if it is unset."""
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT = _find_project_root()

DATA_DIR = _dir_from_env("BLE_CS_DATA_DIR", PROJECT_ROOT / "data")
TRAIN_DATA_PATH = DATA_DIR / "combined_training_reduced.parquet"
EVAL_DATA_PATH = DATA_DIR / "combined_eval_reduced.parquet"

# Folders are created when something is written, not at import time.
OUTPUT_DIR = _dir_from_env("BLE_CS_OUTPUT_DIR", PROJECT_ROOT / "outputs")
CHECKPOINT_DIR = _dir_from_env("BLE_CS_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")


# ============================================================
# Dataset schema
# ============================================================
TARGET = "GT_Distance_m"  # ground-truth distance in metres (regression target)

# Raw IQ samples of one channel sounding procedure; only indices 4..78 (inclusive) are used.
VALID_IQ_INDICES = range(4, 79)

IQ_LOCAL_I = [f"I_local_{i}" for i in VALID_IQ_INDICES]
IQ_LOCAL_Q = [f"Q_local_{i}" for i in VALID_IQ_INDICES]
IQ_REMOTE_I = [f"I_remote_{i}" for i in VALID_IQ_INDICES]
IQ_REMOTE_Q = [f"Q_remote_{i}" for i in VALID_IQ_INDICES]

# Model input: 4 rows x 75 IQ samples = 300 features. The order matters, the 2D models
# reshape it back into a 4 x 75 image.
FEATURES_RAW_IQ_ALL = IQ_LOCAL_I + IQ_LOCAL_Q + IQ_REMOTE_I + IQ_REMOTE_Q
NUM_FEATURES = len(FEATURES_RAW_IQ_ALL)


# ============================================================
# Fixed training settings
# ============================================================
# Deliberately not configurable from the CLI: training only has to be fast and
# reproducible (training twice gives bit-identical weights); the focus of this
# project is deployment, not accuracy. Every value is stored in the checkpoint.
SEED = 42
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
TRAIN_SAMPLE_RATE = 0.1  # train on a random 10 % of the training set

# Representative training samples for INT8 calibration (ONNX and TFLite), drawn with
# SEED: every export calibrates on exactly the same data.
CALIBRATION_SAMPLES = 1000
