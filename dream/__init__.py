"""
DREAM: Dynamic Recall and Elastic Adaptive Memory

A modular continuous-time RNN with:
- Surprise-driven plasticity
- Liquid Time-Constants (LTC)
- Fast weights with Hebbian learning
- Sleep consolidation

Example
-------
>>> from dream import DREAM, DREAMConfig, DREAMCell
>>> model = DREAM(input_dim=64, hidden_dim=128, rank=8)
>>> x = torch.randn(4, 50, 64)
>>> output, state = model(x)

Coordination (optional)
-----------------------
>>> from dream import CoordinatedDREAMStack
>>> model = CoordinatedDREAMStack(input_dim=80, hidden_dims=[128, 128, 128])
"""

from .config import DREAMConfig
from .state import DREAMState
from .cell import DREAMCell
from .layer import DREAM
from .stack import DREAMStack
from .layers import (
    CoordinatedDREAMCell,
    CoordinatedDREAMStack,
    CoordinatedState,
)
from . import layers

__version__ = "0.2.0"

__all__ = [
    # Config & State
    "DREAMConfig",
    "DREAMState",

    # Core
    "DREAMCell",
    "layers",

    # High-level API
    "DREAM",
    "DREAMStack",

    # Coordination (optional)
    "CoordinatedDREAMCell",
    "CoordinatedDREAMStack",
    "CoordinatedState",
]
