from dataclasses import dataclass
from typing import Tuple

from .config import ActivationConfig
from .types import GestureLabel, SystemStatus


@dataclass
class SequenceState:
    status: SystemStatus = SystemStatus.INACTIVE
    progress: int = 0
    start_time_s: float | None = None
    last_accept_time_s: float | None = None


class GestureSequenceFSM:
    """Finite-state machine for ordered gesture activation with timeout."""

    def __init__(self, cfg: ActivationConfig):
        self.cfg = cfg
        self.sequence: Tuple[GestureLabel, ...] = cfg.target_sequence
        self.state = SequenceState()

    def reset(self):
        self.state = SequenceState(status=SystemStatus.INACTIVE)

    def _timed_out(self, now_s: float) -> bool:
        if self.state.start_time_s is None:
            return False
        return (now_s - self.state.start_time_s) > self.cfg.sequence_timeout_s

    def update(self, gesture: GestureLabel, now_s: float) -> tuple[SystemStatus, str]:
        # Once activated, stay activated for this sample/session.
        if self.state.status == SystemStatus.ACTIVATED:
            return self.state.status, "already_activated"

        if self._timed_out(now_s):
            self.reset()
            return self.state.status, "timeout_reset"

        if gesture == GestureLabel.UNKNOWN:
            return self.state.status, "no_valid_gesture"

        if self.state.last_accept_time_s is not None:
            if (now_s - self.state.last_accept_time_s) < self.cfg.min_seconds_between_accepts:
                return self.state.status, "debounced"

        expected = self.sequence[self.state.progress]

        if gesture != expected:
            # Wrong order invalidates attempt.
            self.reset()
            return self.state.status, "wrong_order_reset"

        # Correct next gesture.
        if self.state.progress == 0:
            self.state.start_time_s = now_s

        self.state.progress += 1
        self.state.last_accept_time_s = now_s

        if self.state.progress == len(self.sequence):
            if self.state.start_time_s is not None and (now_s - self.state.start_time_s) <= self.cfg.sequence_timeout_s:
                self.state.status = SystemStatus.ACTIVATED
                return self.state.status, "activated"

            self.reset()
            return self.state.status, "timeout_on_completion_reset"

        return self.state.status, "progressed"
