"""
DREAM Stack - Multi-layer DREAM.

Stack of DREAM layers with optional dropout.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .layer import DREAM
from .state import DREAMState


class DREAMStack(nn.Module):
    """
    Stack of multiple DREAM layers.

    Parameters
    ----------
    input_dim : int
        Input dimension for first layer
    hidden_dims : list of int
        Hidden dimensions for each layer
    rank : int
        Fast weights rank for all layers
    dropout : float
        Dropout between layers
    **kwargs
        Additional arguments for DREAMConfig

    Examples
    --------
    >>> model = DREAMStack(
    ...     input_dim=64,
    ...     hidden_dims=[128, 128, 64],
    ...     rank=8,
    ...     dropout=0.1
    ... )
    >>> x = torch.randn(32, 50, 64)
    >>> output, states = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        rank: int = 8,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(DREAM(input_dim, hidden_dims[0], rank, **kwargs))

        # Subsequent layers
        for hidden_dim in hidden_dims[1:]:
            self.layers.append(DREAM(hidden_dims[0], hidden_dim, rank, **kwargs))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.hidden_dims = hidden_dims

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> List[DREAMState]:
        """Initialize state for all layers."""
        return [
            layer.init_state(batch_size, device, dtype)
            for layer in self.layers
        ]

    def set_fast_weights_mode(self, freeze: bool):
        """Set fast weights mode for all layers."""
        for layer in self.layers:
            layer.set_fast_weights_mode(freeze)

    def train(self, mode: bool = True):
        """Set training mode and auto-freeze fast weights."""
        super().train(mode)
        self.set_fast_weights_mode(freeze=mode)
        return self

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[DREAMState]] = None,
        return_sequences: bool = True
    ) -> Tuple[torch.Tensor, List[DREAMState]]:
        """
        Process sequence through all layers.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, time, input_dim)
        states : list, optional
            Initial states for each layer
        return_sequences : bool
            Whether to return all timesteps

        Returns
        -------
        output : torch.Tensor
            Output from final layer
        states : list
            Final states for all layers
        """
        if states is None:
            states = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)

        output = x

        for i, layer in enumerate(self.layers):
            output, states[i] = layer(output, states[i], return_sequences)

            # Apply dropout between layers (not on last layer)
            if self.dropout is not None and i < len(self.layers) - 1:
                output = self.dropout(output)

        return output, states
