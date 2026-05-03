"""Late fusion across classical (EAR/MAR/HOG+CLF) and deep (CNN / CNN-LSTM) branches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FusionStrategy(str, Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    WEIGHTED_MEAN = "weighted_mean"
    BOTH = "both"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class FusionConfig:
    strategy: FusionStrategy | str = FusionStrategy.WEIGHTED_MEAN
    threshold: float = 0.5
    classical_weight: float = 0.45
    cnn_weight: float = 0.45
    lstm_weight: float = 0.10

    def __post_init__(self):
        if isinstance(self.strategy, str):
            self.strategy = FusionStrategy(self.strategy)


@dataclass
class FusionResult:
    alert: bool
    fused_proba: float
    fused_label: int
    classical: dict[str, Any] | None
    dl_cnn: dict[str, Any] | None
    dl_lstm: dict[str, Any] | None
    note: str


def _scores_from_branch(out: dict | None, key_proba: str = "proba") -> tuple[float | None, bool]:
    if out is None or not out.get("ok", True):
        return None, False
    return float(out[key_proba]), True


def fuse_predictions(
    classical: dict | None,
    dl_cnn: dict | None,
    dl_lstm: dict | None,
    cfg: FusionConfig,
) -> FusionResult:
    """
    Combine branch outputs into one fatigue alert.

    Branches expose `proba` = P(trigger) and optional `ok` (False ⇒ ignored).
    """
    p_c, ok_c = _scores_from_branch(classical)
    p_cn, ok_cn = _scores_from_branch(dl_cnn)
    p_ls, ok_ls = _scores_from_branch(dl_lstm)

    parts: list[tuple[float, float]] = []
    if ok_c:
        parts.append((p_c, cfg.classical_weight))
    if ok_cn:
        parts.append((p_cn, cfg.cnn_weight))
    if ok_ls:
        parts.append((p_ls, cfg.lstm_weight))

    strat = FusionStrategy(cfg.strategy) if isinstance(cfg.strategy, str) else cfg.strategy

    if not parts:
        return FusionResult(
            alert=False,
            fused_proba=0.0,
            fused_label=0,
            classical=classical,
            dl_cnn=dl_cnn,
            dl_lstm=dl_lstm,
            note="no_valid_branch",
        )

    probs_only = [p for p, _ in parts]

    if strat == FusionStrategy.MEAN:
        fused = _mean(probs_only)
        note = "mean"
    elif strat == FusionStrategy.MAX:
        fused = max(probs_only)
        note = "max"
    elif strat == FusionStrategy.MIN:
        fused = min(probs_only)
        note = "min"
    elif strat == FusionStrategy.WEIGHTED_MEAN:
        wsum = sum(w for _, w in parts)
        fused = sum(p * w for p, w in parts) / wsum if wsum > 0 else _mean(probs_only)
        note = "weighted_mean"
    elif strat == FusionStrategy.BOTH:
        fused = _mean(probs_only)
        if ok_c and ok_cn:
            alert = p_c >= cfg.threshold and p_cn >= cfg.threshold
            return FusionResult(
                alert=bool(alert),
                fused_proba=float(fused),
                fused_label=int(alert),
                classical=classical,
                dl_cnn=dl_cnn,
                dl_lstm=dl_lstm,
                note="both_classical_and_cnn",
            )
        note = "both_fallback_mean"
    else:
        fused = _mean(probs_only)
        note = "mean_fallback"

    alert = fused >= cfg.threshold
    return FusionResult(
        alert=bool(alert),
        fused_proba=float(fused),
        fused_label=int(alert),
        classical=classical,
        dl_cnn=dl_cnn,
        dl_lstm=dl_lstm,
        note=note,
    )
