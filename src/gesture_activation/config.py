from dataclasses import dataclass, field
from typing import Tuple

from .types import GestureLabel


@dataclass
class ActivationConfig:
    # Required sequence: ok then peace.
    target_sequence: Tuple[GestureLabel, GestureLabel] = field(
        default_factory=lambda: (GestureLabel.OK, GestureLabel.PEACE)
    )
    # Maximum time allowed between first and last required gesture.
    sequence_timeout_s: float = 3.0
    # Debounce to avoid repeated detections from one held pose.
    min_seconds_between_accepts: float = 0.35
    # Require a gesture to persist for N consecutive frames before accepting it.
    min_consecutive_frames_for_accept: int = 3

    # MediaPipe hand tracker settings.
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6

    # Gesture rule thresholds. Tune these on your data if needed.
    finger_extended_margin: float = 0.02
    peace_ring_bent_margin: float = 0.02
    peace_pinky_bent_margin: float = 0.02
    ok_thumb_index_dist_max: float = 0.08
    ok_middle_extended_margin: float = 0.015
    ok_ring_extended_margin: float = 0.015
    ok_pinky_extended_margin: float = 0.015
