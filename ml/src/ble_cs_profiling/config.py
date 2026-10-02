"""
Zentrale Konfigurationsdatei für das Edge AI Profiling Projekt.
"""
from pathlib import Path

# ============================================================
# 📂 PFAD-MANAGEMENT (Projektweit relativ aufgelöst)
# ============================================================
# Ermittelt den Ordner, in dem config.py liegt, als Ankerpunkt
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent if CONFIG_DIR.name in ["src", "pipeline"] else CONFIG_DIR

# Daten-Verzeichnisse
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "combined_training_reduced.parquet"
EVAL_DATA_PATH = DATA_DIR / "combined_eval_reduced.parquet"

# Output- & Artefakt-Verzeichnisse (ONNX, TFLite, C-Arrays, CSV-Logs)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trainierte PyTorch-Checkpoints als Schnittstelle zwischen Training und Export
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


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
