from typing import Callable, Optional

from .types import FrameResult, SystemStatus


class ActivationTrigger:
    """One-shot trigger invoked only when activation status becomes activated."""

    def __init__(self, callback: Optional[Callable[[FrameResult], None]] = None):
        self._callback = callback
        self._fired = False

    def reset(self):
        self._fired = False

    def maybe_fire(self, result: FrameResult) -> bool:
        if self._fired:
            return False
        if result.status != SystemStatus.ACTIVATED:
            return False

        self._fired = True
        if self._callback is not None:
            self._callback(result)
        return True
