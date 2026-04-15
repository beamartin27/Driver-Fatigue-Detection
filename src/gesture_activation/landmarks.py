from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp

from .config import ActivationConfig

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
except Exception:
    vision = None
    BaseOptions = None


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = Path("models/hand_landmarker.task")


@dataclass
class HandLandmarks:
    # Normalized landmark list: (x, y, z) for 21 points.
    points: List[Tuple[float, float, float]]


class HandLandmarkExtractor:
    def __init__(self, cfg: ActivationConfig):
        self._use_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        self._hands = None
        self._hand_landmarker = None

        if self._use_legacy_solutions:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=cfg.max_num_hands,
                min_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
            )
            return

        if vision is None or BaseOptions is None:
            raise RuntimeError(
                "MediaPipe does not expose 'solutions' and Tasks API imports failed. "
                "Install a compatible version, e.g. mediapipe>=0.10."
            )

        self._ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=cfg.max_num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_hand_model():
        if HAND_MODEL_PATH.exists() and HAND_MODEL_PATH.stat().st_size > 0:
            return
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

    def extract(self, frame_bgr) -> Optional[HandLandmarks]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._use_legacy_solutions:
            result = self._hands.process(frame_rgb)
            if not result.multi_hand_landmarks:
                return None

            hand = result.multi_hand_landmarks[0]
            points = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            return HandLandmarks(points=points)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._hand_landmarker.detect(mp_img)
        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]
        points = [(lm.x, lm.y, lm.z) for lm in hand]
        return HandLandmarks(points=points)

    def close(self):
        if self._hands is not None:
            self._hands.close()
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
