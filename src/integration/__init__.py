"""Member 5 — End-to-end integration: gesture gate, landmarks, fusion, HUD."""

from .fusion import FusionConfig, FusionResult, fuse_predictions
from .classical_infer import ClassicalPredictor, trigger_probability

__all__ = [
    "FusionConfig",
    "FusionResult",
    "fuse_predictions",
    "ClassicalPredictor",
    "trigger_probability",
]
