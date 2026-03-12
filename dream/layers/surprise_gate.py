"""
Block 2: Surprise Gate

Computes surprise from prediction error:
- S = sigmoid((||e|| - τ) / γ)
- τ = τ₀ * (1 + α * H) where H = entropy from error variance
"""

import torch
import torch.nn as nn


class SurpriseGate(nn.Module):
    """
    Surprise Gate block.

    Computes surprise as a normalized prediction error,
    modulated by uncertainty (entropy).

    Parameters
    ----------
    hidden_dim : int
        Dimension of hidden state
    base_threshold : float
        Base surprise threshold (τ₀)
    entropy_influence : float
        Alpha (α) - entropy influence on threshold
    surprise_temperature : float
        Gamma (γ) - surprise temperature
    kappa : float
        Gain modulation coefficient
    """

    def __init__(
        self,
        hidden_dim: int,
        base_threshold: float = 0.3,
        entropy_influence: float = 0.1,
        surprise_temperature: float = 0.05,
        kappa: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.tau_0 = nn.Parameter(torch.tensor(base_threshold))
        self.alpha = nn.Parameter(torch.tensor(entropy_influence))
        self.gamma = nn.Parameter(torch.tensor(surprise_temperature))
        self.kappa = nn.Parameter(torch.tensor(kappa))

    def compute_entropy(self, variance: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy from variance.

        H = 0.5 * log(2πe * var)
        """
        eps = 1e-6
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * (variance + eps))
        return torch.clamp(entropy, 0.0, 2.0)

    def forward(
        self,
        error: torch.Tensor,
        error_var: torch.Tensor,
        error_mean: torch.Tensor
    ) -> tuple:
        """
        Compute surprise.

        Parameters
        ----------
        error : torch.Tensor
            Prediction error (batch, input_dim)
        error_var : torch.Tensor
            Error variance (batch, input_dim)
        error_mean : torch.Tensor
            Error mean (batch, input_dim)

        Returns
        -------
        surprise : torch.Tensor
            Surprise values (batch,)
        error_norm : torch.Tensor
            Error norm (batch,)
        gain : torch.Tensor
            Gain modulation (batch, 1)
        """
        batch_size = error.shape[0]
        eps = 1e-6

        # Error norm
        error_norm = error.norm(dim=-1)

        # Entropy from variance
        variance = error_var.mean(dim=-1)
        entropy = self.compute_entropy(variance)

        # Adaptive threshold
        tau = 1.0 + self.alpha * entropy

        # Relative error (compared to expected)
        baseline_error = error_mean.norm(dim=-1) + eps
        relative_error = error_norm / baseline_error

        # Surprise
        surprise = torch.sigmoid((relative_error - tau) / (self.gamma * 2))

        # Gain modulation
        gain = 1.0 + self.kappa * surprise.unsqueeze(1)

        return surprise, error_norm, gain
