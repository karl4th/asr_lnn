"""
DREAM Layers - Modular building blocks.

Each block can be enabled/disabled independently via config.
"""

from .predictive_coding import PredictiveCoding
from .surprise_gate import SurpriseGate
from .fast_weights import FastWeights
from .ltc import LiquidTimeConstants
from .sleep_consolidation import SleepConsolidation
from .coordination import CoordinatedDREAMCell, CoordinatedDREAMStack, CoordinatedState

__all__ = [
    # Basic blocks
    "PredictiveCoding",
    "SurpriseGate",
    "FastWeights",
    "LiquidTimeConstants",
    "SleepConsolidation",
    # Coordination (optional)
    "CoordinatedDREAMCell",
    "CoordinatedDREAMStack",
    "CoordinatedState",
]
