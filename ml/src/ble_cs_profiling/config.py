"""
Zentrale Konfigurationsdatei für das Edge AI Profiling Projekt.

Alle Pfade lassen sich per Umgebungsvariable überschreiben (z.B. in Docker/CI):

    BLE_CS_PROJECT_ROOT   Basisordner für data/, outputs/, checkpoints/
    BLE_CS_DATA_DIR       Datensätze (Parquet)
    BLE_CS_OUTPUT_DIR     Exportierte Artefakte (ONNX, TFLite, C-Arrays)
    BLE_CS_CHECKPOINT_DIR Trainierte PyTorch-Checkpoints
"""
import os
from pathlib import Path


# ============================================================
# 📂 PFAD-MANAGEMENT (unabhängig vom Startordner)
# ============================================================
def _find_project_root() -> Path:
    """Env-Variable, sonst der nächste Ordner oberhalb dieses Pakets mit pyproject.toml."""
    if root := os.environ.get("BLE_CS_PROJECT_ROOT"):
        return Path(root).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Als Wheel installiert (z.B. Docker) gibt es keinen Source-Tree: Env-Variable setzen.
    return Path.cwd()


def _dir_from_env(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT = _find_project_root()

# Daten-Verzeichnisse
DATA_DIR = _dir_from_env("BLE_CS_DATA_DIR", PROJECT_ROOT / "data")
TRAIN_DATA_PATH = DATA_DIR / "combined_training_reduced.parquet"
EVAL_DATA_PATH = DATA_DIR / "combined_eval_reduced.parquet"

# Output- & Artefakt-Verzeichnisse (ONNX, TFLite, C-Arrays, CSV-Logs).
# Ordner werden erst beim Schreiben angelegt, nicht schon beim Import.
OUTPUT_DIR = _dir_from_env("BLE_CS_OUTPUT_DIR", PROJECT_ROOT / "outputs")

# Trainierte PyTorch-Checkpoints als Schnittstelle zwischen Training und Export
CHECKPOINT_DIR = _dir_from_env("BLE_CS_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")


# ============================================================
# 🎯 TARGET & METADATEN
# ============================================================
TARGET = "GT_Distance_m"


# ============================================================
# 📡 FEATURE-DEFINITIONEN (Reine IQ-Samples)
# ============================================================
VALID_IQ_INDICES = range(4, 79)  # Indizes 4 bis 78 inklusive

IQ_LOCAL_I = [f"I_local_{i}" for i in VALID_IQ_INDICES]
IQ_LOCAL_Q = [f"Q_local_{i}" for i in VALID_IQ_INDICES]
IQ_REMOTE_I = [f"I_remote_{i}" for i in VALID_IQ_INDICES]
IQ_REMOTE_Q = [f"Q_remote_{i}" for i in VALID_IQ_INDICES]

# Gesamte Feature-Liste für Training & Inferenz (300 Dimensionen: 75 * 4)
FEATURES_RAW_IQ_ALL = IQ_LOCAL_I + IQ_LOCAL_Q + IQ_REMOTE_I + IQ_REMOTE_Q
NUM_FEATURES = len(FEATURES_RAW_IQ_ALL)


# ============================================================
# 🏋️ FESTE TRAININGS-EINSTELLUNGEN
# ============================================================
# Absichtlich nicht per CLI änderbar: Training soll nur schnell und reproduzierbar
# sein (2x trainieren = 2x gleiches Ergebnis), der Fokus liegt auf dem Deployment.
SEED = 42
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
TRAIN_SAMPLE_RATE = 0.1  # 0.1 = zufällige 10 % der Trainingsdaten

# Repräsentative Trainings-Samples für die INT8-Kalibrierung (ONNX + TFLite),
# gezogen mit SEED: jeder Export kalibriert auf exakt denselben Daten.
CALIBRATION_SAMPLES = 1000
