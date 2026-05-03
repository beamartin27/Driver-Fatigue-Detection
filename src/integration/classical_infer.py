"""Load Member 3 joblib pipelines and score frames from landmarks + BGR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.pipeline import Pipeline

import joblib

from src.classical_pipeline.features import extract_frame_features


def trigger_probability(pipeline: Pipeline, features_2d_row: np.ndarray) -> tuple[float, int]:
    """
    Return (P(class==trigger), predicted_label) for a single row (1, F).

    Member 3's SVC is trained without probability=True; in that case we map
    decision_function → (0,1) via the logistic sigmoid for fusion weights.
    """
    X = features_2d_row
    label = int(pipeline.predict(X)[0])
    if hasattr(pipeline, "predict_proba"):
        proba_trigger = float(pipeline.predict_proba(X)[0, 1])
    elif hasattr(pipeline, "decision_function"):
        d = np.asarray(pipeline.decision_function(X), dtype=np.float64).ravel()
        proba_trigger = float(expit(d[0]))
    else:
        proba_trigger = float(label)
    return proba_trigger, label


class ClassicalPredictor:
    """Thin wrapper mirroring DL's dict output for Member 5's fusion."""

    def __init__(self, model_path: Path | str):
        self.path = Path(model_path)
        self.pipeline: Pipeline = joblib.load(self.path)

    def predict(self, frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> dict:
        vec = extract_frame_features(frame_bgr, landmarks_px)
        if vec is None:
            return {
                "label": 0,
                "proba": 0.0,
                "model": self.path.name,
                "ok": False,
                "detail": "degenerate_roi",
            }
        X = vec.reshape(1, -1)
        proba, label = trigger_probability(self.pipeline, X)
        return {
            "label": label,
            "proba": float(proba),
            "model": self.path.name,
            "ok": True,
            "detail": "",
        }
