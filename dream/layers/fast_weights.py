"""
Block 3: Fast Weights

Fast weights with Hebbian learning and surprise-driven plasticity:
- U updated via: dU = -λ(U - U_target) + (η * S) * (h ⊗ e) @ V
- Orthogonal decomposition via fixed V matrix
"""

import torch
import torch.nn as nn


class FastWeights(nn.Module):
    """
    Fast Weights block.

    Implements fast weights with:
    - Low-rank decomposition (U @ V.T)
    - Hebbian learning modulated by surprise
    - Forgetting toward target weights
    - Homeostatic normalization

    Parameters
    ----------
    hidden_dim : int
        Dimension of hidden state
    input_dim : int
        Dimension of input features
    rank : int
        Rank of fast weights decomposition
    forgetting_rate : float
        Lambda (λ) - forgetting rate
    base_plasticity : float
        Base plasticity coefficient (η)
    target_norm : float
        Target norm for homeostasis
    time_step : float
        Integration time step (dt)
    freeze_fast_weights : bool
        If True, fast weights are frozen (static base training)
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        rank: int = 16,
        forgetting_rate: float = 0.005,
        base_plasticity: float = 0.5,
        target_norm: float = 2.0,
        time_step: float = 0.1,
        freeze_fast_weights: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.rank = rank
        self.forgetting_rate = forgetting_rate
        self.time_step = time_step
        self.freeze_fast_weights = freeze_fast_weights
        self.target_norm = target_norm

        # V: fixed orthogonal sensory filter (input_dim, rank)
        V_init = torch.randn(input_dim, rank)
        Q, _ = torch.linalg.qr(V_init)
        self.register_buffer('V', Q)

        # eta: vector plasticity coefficient (hidden_dim,)
        self.eta = nn.Parameter(torch.ones(hidden_dim) * base_plasticity)

    def compute_fast_effect(
        self,
        U: torch.Tensor,
        V: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fast weights contribution to input projection.

        fast_effect = (U @ V.T) @ x

        Parameters
        ----------
        U : torch.Tensor
            Fast weights (batch, hidden_dim, rank)
        V : torch.Tensor
            Fixed filter (input_dim, rank)
        x : torch.Tensor
            Input (batch, input_dim)

        Returns
        -------
        fast_effect : torch.Tensor
            Fast weights contribution (batch, hidden_dim)
        """
        batch_size = x.shape[0]
        # Efficient: (U @ V.T) @ x = einsum('bhr,ir,bi->bh', U, V, x)
        fast_effect = torch.einsum('bhr,ir,bi->bh', U, V, x)
        return fast_effect

    def update(
        self,
        h: torch.Tensor,
        error: torch.Tensor,
        surprise: torch.Tensor,
        U: torch.Tensor,
        U_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Update fast weights via STDP.

        dU = -λ(U - U_target) + (η * S) * (h ⊗ e) @ V

        Parameters
        ----------
        h : torch.Tensor
            Hidden state (batch, hidden_dim)
        error : torch.Tensor
            Prediction error (batch, input_dim)
        surprise : torch.Tensor
            Surprise (batch,)
        U : torch.Tensor
            Current fast weights (batch, hidden_dim, rank)
        U_target : torch.Tensor
            Target fast weights (batch, hidden_dim, rank)

        Returns
        -------
        U_new : torch.Tensor
            Updated fast weights
        """
        if self.freeze_fast_weights:
            return U

        batch_size = h.shape[0]

        # Hebbian term: outer(h, e) @ V
        # = h.unsqueeze(2) * (e @ V).unsqueeze(1)
        eV = error @ self.V  # (batch, rank)
        hebbian = h.unsqueeze(2) * eV.unsqueeze(1)  # (batch, hidden, rank)

        # Plasticity modulation (eta * surprise)
        plasticity = self.eta.unsqueeze(0) * surprise.unsqueeze(1)
        plasticity = plasticity.unsqueeze(2)  # (batch, hidden, 1)

        # Forgetting term
        forgetting = -self.forgetting_rate * (U - U_target)

        # Full STDP update
        dU = forgetting + plasticity * hebbian

        # Euler integration
        U_new = U + dU * self.time_step

        # Normalize to target norm (homeostasis)
        U_norm = U_new.norm(dim=(1, 2), keepdim=True)
        scale = (self.target_norm / (U_norm + 1e-6)).clamp(max=2.0)
        U_new = U_new * scale

        return U_new
