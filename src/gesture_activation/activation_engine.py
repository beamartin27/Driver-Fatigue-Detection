from .config import ActivationConfig
from .gesture_recognizer import RuleBasedGestureRecognizer
from .landmarks import HandLandmarkExtractor
from .sequence_fsm import GestureSequenceFSM
from .types import FrameResult, GestureLabel, SystemStatus


class ActivationEngine:
    """Coordinates hand landmark extraction, gesture recognition, and FSM logic."""

    def __init__(self, cfg: ActivationConfig | None = None):
        self.cfg = cfg or ActivationConfig()
        self.extractor = HandLandmarkExtractor(self.cfg)
        self.recognizer = RuleBasedGestureRecognizer(self.cfg)
        self.fsm = GestureSequenceFSM(self.cfg)
        self._stable_gesture = GestureLabel.UNKNOWN
        self._stable_count = 0

    def reset(self):
        self.fsm.reset()
        self._stable_gesture = GestureLabel.UNKNOWN
        self._stable_count = 0

    def _confirm_gesture(self, raw_gesture: GestureLabel) -> GestureLabel:
        if raw_gesture == GestureLabel.UNKNOWN:
            self._stable_gesture = GestureLabel.UNKNOWN
            self._stable_count = 0
            return GestureLabel.UNKNOWN

        if raw_gesture == self._stable_gesture:
            self._stable_count += 1
        else:
            self._stable_gesture = raw_gesture
            self._stable_count = 1

        if self._stable_count >= self.cfg.min_consecutive_frames_for_accept:
            return self._stable_gesture
        return GestureLabel.UNKNOWN

    def process_frame(self, frame_bgr, timestamp_s: float, frame_index: int | None = None) -> FrameResult:
        hand = self.extractor.extract(frame_bgr)
        if hand is None:
            return FrameResult(
                timestamp_s=timestamp_s,
                status=self.fsm.state.status,
                gesture=GestureLabel.UNKNOWN,
                sequence_progress=self.fsm.state.progress,
                note="no_hand",
                hand_detected=False,
                frame_index=frame_index,
            )

        raw_gesture = self.recognizer.predict(hand)
        gesture = self._confirm_gesture(raw_gesture)
        status, note = self.fsm.update(gesture, timestamp_s)

        if raw_gesture != gesture and raw_gesture != GestureLabel.UNKNOWN:
            note = f"{note}|stabilizing"

        return FrameResult(
            timestamp_s=timestamp_s,
            status=status,
            gesture=gesture,
            sequence_progress=self.fsm.state.progress,
            note=note,
            hand_detected=True,
            frame_index=frame_index,
        )

    def close(self):
        self.extractor.close()

    @staticmethod
    def is_activated(result: FrameResult) -> bool:
        return result.status == SystemStatus.ACTIVATED
