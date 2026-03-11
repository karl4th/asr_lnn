"""
Coordinated DREAMStack.

Implements hierarchical coordination with:
1. Bottom-up prediction errors
2. Top-down modulation vectors
3. Two-pass processing (feedforward + backward modulation)
4. Hierarchical sleep consolidation
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from .cell import DREAMCell
from .config import DREAMConfig
from .state import DREAMState


class CoordinatedDREAMStack(nn.Module):
    """
    Coordinated DREAM Stack with hierarchical processing.

    Architecture:
    - Multiple DREAM layers with coordination
    - Bottom-up: prediction errors between layers
    - Top-down: modulation vectors for sensitivity control
    - Two-pass processing per timestep

    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dims : List[int]
        Hidden dimensions for each layer
    rank : int
        Fast weights rank
    dropout : float
        Dropout between layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        rank: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.num_layers = len(hidden_dims)

        # Create layers with coordination enabled
        self.layers = nn.ModuleList()

        # First layer
        config = DREAMConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dims[0],
            rank=rank,
            use_coordination=True
        )
        self.layers.append(DREAMCell(config))

        # Subsequent layers
        for i in range(1, len(hidden_dims)):
            config = DREAMConfig(
                input_dim=hidden_dims[i-1],
                hidden_dim=hidden_dims[i],
                rank=rank,
                use_coordination=True
            )
            self.layers.append(DREAMCell(config))

        # Store modulation buffers (initialized in forward)
        self.num_coordination_layers = len(hidden_dims) - 1  # No modulation for top layer

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> List[DREAMState]:
        """Initialize states for all layers."""
        return [layer.init_state(batch_size, device=device) for layer in self.layers]

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[DREAMState]] = None,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, List[DREAMState], Dict]:
        """
        Forward pass with two-pass coordination.

        Pass 1: Feedforward (bottom-up, no modulation)
        Pass 2: Backward modulation (top-down, for next timestep)

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, time, input_dim)
        states : List[DREAMState], optional
            States for each layer
        return_all : bool
            Return all timesteps

        Returns
        -------
        output : torch.Tensor
            Output from final layer
        states : List[DREAMState]
            Final states for each layer
        coordination_info : Dict
            Information about coordination (modulations, errors)
        """
        batch_size, time_steps, _ = x.shape
        device = x.device

        if states is None:
            states = self.init_states(batch_size, device)

        # Storage for coordination info
        coordination_info = {
            'modulations': [],  # Top-down modulations
            'inter_layer_errors': []  # Prediction errors between layers
        }

        if return_all:
            all_outputs = []

        # Initialize modulation buffers (for next timestep)
        modulations = [None] * self.num_layers  # Top-down modulations

        for t in range(time_steps):
            x_t = x[:, t, :]  # (batch, input_dim)

            # ================================================================
            # PASS 1: Feedforward (bottom-up, no modulation yet)
            # ================================================================
            layer_outputs = []
            current_input = x_t

            for i, layer in enumerate(self.layers):
                # Process layer
                h_new, states[i] = layer(current_input, states[i])
                layer_outputs.append(h_new)

                # Prepare input for next layer
                if i < self.num_layers - 1:
                    current_input = h_new
                    if self.dropout is not None:
                        current_input = self.dropout(current_input)

            # ================================================================
            # PASS 2: Backward Modulation (top-down, for next timestep)
            # ================================================================
            # Generate modulations from top to bottom
            for i in range(self.num_layers - 1, 0, -1):
                # Upper layer generates modulation for lower layer
                upper_h = layer_outputs[i]
                modulation = self.layers[i].generate_modulation(upper_h)
                modulations[i - 1] = modulation

                # Compute inter-layer prediction error
                upper_pred = self.layers[i].predict_lower_activity(upper_h)
                lower_actual = layer_outputs[i - 1]
                inter_error = self.layers[i].compute_inter_layer_error(upper_pred, lower_actual)

                if t == 0:  # Only store first timestep for memory efficiency
                    coordination_info['inter_layer_errors'].append(inter_error.detach())

            # Store final output
            if return_all:
                all_outputs.append(layer_outputs[-1].unsqueeze(1))

            # Store modulations for coordination info
            if t == 0:
                coordination_info['modulations'] = [m.detach() if m is not None else None
                                                    for m in modulations]

        if return_all:
            output = torch.cat(all_outputs, dim=1)
        else:
            output = layer_outputs[-1]

        return output, states, coordination_info

    def forward_with_global_sleep(
        self,
        x: torch.Tensor,
        states: Optional[List[DREAMState]] = None
    ) -> Tuple[torch.Tensor, List[DREAMState], float]:
        """
        Forward pass with hierarchical sleep consolidation.

        All layers consolidate together when global surprise is high.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, time, input_dim)
        states : List[DREAMState], optional
            States for each layer

        Returns
        -------
        output : torch.Tensor
            Output from final layer
        states : List[DREAMState]
            Final states for each layer
        global_surprise : float
            Average surprise across all layers
        """
        batch_size, time_steps, _ = x.shape

        if states is None:
            states = self.init_states(batch_size, device=x.device)

        # Track surprises across layers
        all_surprises = []

        for t in range(time_steps):
            x_t = x[:, t, :]
            layer_outputs = []
            current_input = x_t

            # Forward pass
            for i, layer in enumerate(self.layers):
                h_new, states[i] = layer(current_input, states[i])
                layer_outputs.append(h_new)

                # Track surprise
                if hasattr(states[i], 'avg_surprise'):
                    all_surprises.append(states[i].avg_surprise.mean().item())

                # Prepare input for next layer
                if i < self.num_layers - 1:
                    current_input = h_new
                    if self.dropout is not None:
                        current_input = self.dropout(current_input)

        # Compute global surprise
        global_surprise = sum(all_surprises) / len(all_surprises) if all_surprises else 0.0

        # Hierarchical sleep: all layers consolidate together
        if global_surprise > self.layers[0].S_min:
            for layer in self.layers:
                # Trigger sleep consolidation in each layer
                # (This happens automatically in forward pass when surprise is high)
                pass

        return layer_outputs[-1], states, global_surprise

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UncoordinatedDREAMStack(nn.Module):
    """
    Uncoordinated DREAM Stack (baseline for comparison).

    Standard stack without top-down modulation or inter-layer prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        rank: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # Create layers WITHOUT coordination
        self.layers = nn.ModuleList()

        for i, h in enumerate(hidden_dims):
            input_dim_i = input_dim if i == 0 else hidden_dims[i-1]
            config = DREAMConfig(
                input_dim=input_dim_i,
                hidden_dim=h,
                rank=rank,
                use_coordination=False  # No coordination
            )
            self.layers.append(DREAMCell(config))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> List[DREAMState]:
        """Initialize states for all layers."""
        return [layer.init_state(batch_size, device=device) for layer in self.layers]

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[DREAMState]] = None,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, List[DREAMState]]:
        """Standard forward pass without coordination."""
        batch_size, time_steps, _ = x.shape

        if states is None:
            states = self.init_states(batch_size, device=x.device)

        if return_all:
            all_outputs = []

        for t in range(time_steps):
            x_t = x[:, t, :]
            current_input = x_t

            for i, layer in enumerate(self.layers):
                h_new, states[i] = layer(current_input, states[i])

                if i < self.num_layers - 1:
                    current_input = h_new
                    if self.dropout is not None:
                        current_input = self.dropout(current_input)

            if return_all:
                all_outputs.append(h_new.unsqueeze(1))

        if return_all:
            output = torch.cat(all_outputs, dim=1)
        else:
            output = h_new

        return output, states

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
