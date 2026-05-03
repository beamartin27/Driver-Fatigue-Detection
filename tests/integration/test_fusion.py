import pytest

from src.integration.fusion import FusionConfig, FusionStrategy, fuse_predictions


def test_fusion_weighted_mean_masks_missing_branch():
    cfg = FusionConfig(
        strategy=FusionStrategy.WEIGHTED_MEAN,
        classical_weight=0.5,
        cnn_weight=0.5,
        lstm_weight=1.0,
        threshold=0.5,
    )
    fused = fuse_predictions(
        {"proba": 0.2, "label": 0, "ok": True},
        {"proba": 0.8, "label": 1, "ok": False},
        None,
        cfg,
    )
    assert pytest.approx(fused.fused_proba) == 0.2


def test_fusion_both_requires_cnn_when_available():
    cfg = FusionConfig(strategy=FusionStrategy.BOTH, threshold=0.4)
    hi = fuse_predictions(
        {"proba": 0.9, "label": 1, "ok": True},
        {"proba": 0.9, "label": 1, "ok": True},
        None,
        cfg,
    )
    lo = fuse_predictions(
        {"proba": 0.9, "label": 1, "ok": True},
        {"proba": 0.1, "label": 0, "ok": True},
        None,
        cfg,
    )
    assert hi.alert is True
    assert lo.alert is False
