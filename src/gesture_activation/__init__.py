"""Gesture activation baseline package.

Keeps fatigue detection inactive by default and activates only when
an ordered gesture sequence is completed within a timeout window.
"""

from .config import ActivationConfig
from .activation_engine import ActivationEngine

__all__ = ["ActivationConfig", "ActivationEngine"]
