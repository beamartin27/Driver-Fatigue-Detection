import numpy as np
from src.dl_pipeline.train_eval import compute_metrics


def test_compute_metrics_returns_expected_keys_and_perfect_score():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.05, 0.95])
    m = compute_metrics(y_true, y_pred, y_proba)
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        assert k in m
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert 0.99 <= m["auc"] <= 1.0


def test_compute_metrics_handles_imbalance():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.1, 0.3, 0.4, 0.9])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 5 / 6
    assert m["recall"] == 0.5
