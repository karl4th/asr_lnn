"""
DREAM Layer - High-level sequence model.

Provides nn.LSTM-like interface for processing sequences.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .cell import DREAMCell
from .config import DREAMConfig
from .state import DREAMState


class DREAM(nn.Module):
    """
    High-level DREAM sequence model.

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_dim : int
        Dimension of hidden state
    rank : int
        Fast weights rank
    freeze_fast_weights : bool
        If True, fast weights frozen during training
    **kwargs
        Additional arguments passed to DREAMConfig

    Examples
    --------
    >>> model = DREAM(input_dim=64, hidden_dim=128, rank=8)
    >>> x = torch.randn(32, 50, 64)
    >>> output, state = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rank: int = 8,
        freeze_fast_weights: bool = False,
        **kwargs
    ):
        super().__init__()
        self.config = DREAMConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            **kwargs
        )
        self.cell = DREAMCell(self.config, freeze_fast_weights=freeze_fast_weights)
        self.hidden_dim = hidden_dim

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> DREAMState:
        """Initialize model state."""
        return self.cell.init_state(batch_size, device, dtype)

    def set_fast_weights_mode(self, freeze: bool):
        """Set fast weights training mode."""
        self.cell.set_fast_weights_mode(freeze)

    def train(self, mode: bool = True):
        """Set training mode and auto-freeze fast weights."""
        super().train(mode)
        self.set_fast_weights_mode(freeze=mode)
        return self

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[DREAMState] = None,
        return_sequences: bool = True
    ) -> Tuple[torch.Tensor, DREAMState]:
        """
        Process input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, time, input_dim)
        state : DREAMState, optional
            Initial state
        return_sequences : bool
            If True, return all hidden states

        Returns
        -------
        output : torch.Tensor
            If return_sequences=True: (batch, time, hidden_dim)
            If False: (batch, hidden_dim)
        state : DREAMState
            Final state
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.shape}")

        batch_size, time_steps, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(time_steps):
            x_t = x[:, t, :]
            h, state = self.cell(x_t, state)

            if return_sequences:
                outputs.append(h.unsqueeze(1))

        if return_sequences:
            output = torch.cat(outputs, dim=1)
        else:
            output = h

        return output, state
