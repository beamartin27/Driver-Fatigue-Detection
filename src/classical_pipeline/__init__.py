"""Member 3 — Classical fatigue detection pipeline."""

from .features import compute_ear, compute_mar, compute_hog_features, extract_frame_features
from .dataset import build_dataset
from .train_eval import train_and_evaluate
