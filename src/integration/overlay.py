"""OpenCV HUD: gesture prompt, fused fatigue bar, branch readouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.gesture_activation.types import FrameResult
    from src.integration.fusion import FusionResult


def _draw_label(img, text: str, org: tuple[int, int], color_bgr: tuple[int, int, int], scale: float = 0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, 2, cv2.LINE_AA)


def _bar(img, x: int, y: int, w: int, h: int, fill01: float, ok_bgr=(0, 200, 0), bad_bgr=(0, 0, 230)):
    fill01 = float(np.clip(fill01, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), 2)
    inner_w = max(0, int((w - 4) * fill01))
    col = tuple(int(lo * (1.0 - fill01) + hi * fill01) for lo, hi in zip(ok_bgr, bad_bgr))
    cv2.rectangle(img, (x + 2, y + 2), (x + 2 + inner_w, y + h - 2), col, -1)


def draw_activation_overlay(frame, result: FrameResult):
    banner = frame.copy()
    color = (0, 210, 0) if result.status.value == "activated" else (0, 60, 255)
    cv2.rectangle(banner, (0, 0), (banner.shape[1], 115), color, thickness=-1)
    alpha = 0.38
    out = cv2.addWeighted(banner, alpha, frame, 1 - alpha, 0)
    # OpenCV's default font cannot render en/em dashes or arrows; use ASCII only.
    _draw_label(out, "GESTURE GATE - Perform OK then Peace sequence", (16, 32), (255, 255, 255), 0.7)
    _draw_label(out, f"system: {result.status.value.upper()}", (16, 64), (255, 255, 255), 0.75)
    _draw_label(out, f"hand: gesture={result.gesture.value}", (16, 95), (255, 255, 255), 0.6)
    return out


def draw_monitoring_overlay(
    frame,
    *,
    fusion: FusionResult,
    fps: float,
    headless_note: str = "",
):
    """Draw fatigue HUD after activation."""

    out = frame.copy()
    h, w = out.shape[:2]

    banner_h = 150
    top = np.zeros_like(out[:banner_h])
    fatigue_color = (0, 0, 255) if fusion.alert else (0, 165, 0)
    top[:] = fatigue_color if fusion.alert else (25, 80, 25)
    out[:banner_h] = cv2.addWeighted(top, 0.55, out[:banner_h], 0.45, 0)

    title = "ALERT - Possible fatigue / distraction" if fusion.alert else "Monitoring - nominal"
    _draw_label(out, title, (16, 36), (255, 255, 255), 0.78)
    _draw_label(out, f"Fusion P(trigger)={fusion.fused_proba:.2f}  ({fusion.note})", (16, 72), (255, 255, 240), 0.58)

    _bar(out, 16, 92, max(220, w - 32), 18, fusion.fused_proba)

    y0 = banner_h + 18
    c = fusion.classical if fusion.classical else {}
    d = fusion.dl_cnn if fusion.dl_cnn else {}
    l = fusion.dl_lstm if fusion.dl_lstm else {}
    texts = [
        (
            "Classical: "
            + (
                "N/A"
                if not c
                else f"P={c.get('proba', 0):.2f} label={c.get('label', '-')}"
                + ("" if c.get("ok") else f" [{c.get('detail', '')}]")
            )
        ),
        (
            "CNN:       "
            + ("N/A" if not d else f"P={d.get('proba', 0):.2f} label={d.get('label', '-')}")
        ),
        (
            "CNN-LSTM:  "
            + ("N/A" if not l else f"P={l.get('proba', 0):.2f} label={l.get('label', '-')}")
        ),
        (f"FPS ~ {fps:4.1f}   {headless_note}"),
    ]

    patch = np.zeros_like(out[y0 : y0 + 110])
    patch[:] = (20, 20, 22)
    y_end = min(h, y0 + patch.shape[0])
    out[y0:y_end] = cv2.addWeighted(patch[: y_end - y0], 0.55, out[y0:y_end], 0.45, 0)
    yy = y0 + 22
    for line in texts:
        _draw_label(out, line, (16, yy), (235, 235, 235), 0.52)
        yy += 26

    return out
