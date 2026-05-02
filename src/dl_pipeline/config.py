"""Centralised paths, hyperparameters, and seeds for the DL pipeline."""
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parents[2]
DATA_FRAMES     = REPO_ROOT / "data" / "frames"
DATA_LANDMARKS  = REPO_ROOT / "data" / "landmarks"
DATASET_SUMMARY = DATA_LANDMARKS / "dataset_summary.csv"
OUTPUTS_DIR     = REPO_ROOT / "outputs" / "dl_pipeline"
FEATURE_CACHE   = OUTPUTS_DIR / "feature_cache"

SPLIT_TEST_SIZE   = 0.25
SPLIT_RANDOM_SEED = 42

LABEL_MAP   = {"trigger": 1, "non_trigger": 0}
NUM_CLASSES = 2

FACE_PAD_RATIO = 0.10
INPUT_SIZE     = 224

CNN_BATCH_SIZE   = 64
CNN_EPOCHS       = 20
CNN_HEAD_LR      = 1e-4
CNN_BB_LR        = 1e-5
CNN_WEIGHT_DECAY = 1e-4

WINDOW_SIZE         = 16
WINDOW_STRIDE_TRAIN = 1
WINDOW_STRIDE_EVAL  = 4
LSTM_HIDDEN     = 128
LSTM_DROPOUT    = 0.3
LSTM_BATCH_SIZE = 128
LSTM_EPOCHS     = 25
LSTM_LR         = 3e-4

GLOBAL_SEED = 42
