from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SystemStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVATED = "activated"


class GestureLabel(str, Enum):
    UNKNOWN = "unknown"
    PEACE = "peace"
    OK = "ok"


@dataclass
class FrameResult:
    timestamp_s: float
    status: SystemStatus
    gesture: GestureLabel
    sequence_progress: int
    note: str = ""
    hand_detected: bool = False
    frame_index: Optional[int] = None
