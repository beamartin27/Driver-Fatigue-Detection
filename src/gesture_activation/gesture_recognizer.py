from typing import Tuple

from .config import ActivationConfig
from .landmarks import HandLandmarks
from .types import GestureLabel


class RuleBasedGestureRecognizer:
    """Rule-based baseline for peace and ok gesture recognition.

    Rules are intentionally simple and should be tuned using your own camera setup.
    """

    # MediaPipe hand landmark indices.
    WRIST = 0
    THUMB_TIP = 4
    INDEX_PIP, INDEX_TIP = 6, 8
    MIDDLE_PIP, MIDDLE_TIP = 10, 12
    RING_PIP, RING_TIP = 14, 16
    PINKY_PIP, PINKY_TIP = 18, 20

    def __init__(self, cfg: ActivationConfig):
        self.cfg = cfg

    @staticmethod
    def _dist2d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _is_extended(pip, tip, margin: float) -> bool:
        # In image coordinates, smaller y is higher in the image.
        return tip[1] < (pip[1] - margin)

    @staticmethod
    def _is_bent(pip, tip, margin: float) -> bool:
        return tip[1] > (pip[1] - margin)

    def predict(self, hand: HandLandmarks) -> GestureLabel:
        p = hand.points

        index_extended = self._is_extended(p[self.INDEX_PIP], p[self.INDEX_TIP], self.cfg.finger_extended_margin)
        middle_extended = self._is_extended(p[self.MIDDLE_PIP], p[self.MIDDLE_TIP], self.cfg.finger_extended_margin)
        ring_bent = self._is_bent(p[self.RING_PIP], p[self.RING_TIP], self.cfg.peace_ring_bent_margin)
        pinky_bent = self._is_bent(p[self.PINKY_PIP], p[self.PINKY_TIP], self.cfg.peace_pinky_bent_margin)

        thumb_index_dist = self._dist2d(p[self.THUMB_TIP], p[self.INDEX_TIP])
        ring_extended = self._is_extended(p[self.RING_PIP], p[self.RING_TIP], self.cfg.ok_ring_extended_margin)
        pinky_extended = self._is_extended(p[self.PINKY_PIP], p[self.PINKY_TIP], self.cfg.ok_pinky_extended_margin)

        # Peace rule: index+middle extended, ring+pinky bent.
        if index_extended and middle_extended and ring_bent and pinky_bent:
            return GestureLabel.PEACE

        # OK rule (baseline): thumb and index fingertips close, remaining fingers mostly extended.
        if (
            thumb_index_dist <= self.cfg.ok_thumb_index_dist_max
            and self._is_extended(p[self.MIDDLE_PIP], p[self.MIDDLE_TIP], self.cfg.ok_middle_extended_margin)
            and ring_extended
            and pinky_extended
        ):
            return GestureLabel.OK

        # Placeholder for custom rules for your camera/hand pose specifics.
        return GestureLabel.UNKNOWN
