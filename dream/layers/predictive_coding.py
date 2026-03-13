"""
Block 1: Predictive Coding

Implements predictive coding mechanism:
- x_hat = C^T @ h (prediction)
- e = x - x_hat (prediction error)
"""

import torch
import torch.nn as nn


class PredictiveCoding(nn.Module):
    """
    Predictive Coding block.

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_dim : int
        Dimension of hidden state
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # C: decoding matrix (hidden_dim -> input_dim)
        self.C = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.02)

        # W: error injection matrix (input_dim -> hidden_dim)
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)

        # B_base: base input projection (input_dim -> hidden_dim)
        self.B_base = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        
        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple:
        """
        Compute prediction and error.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, input_dim)
        h : torch.Tensor
            Hidden state (batch, hidden_dim)

        Returns
        -------
        x_pred : torch.Tensor
            Prediction (batch, input_dim)
        error : torch.Tensor
            Prediction error (batch, input_dim)
        """
        # Apply LayerNorm to hidden state for stability
        h_norm = self.layer_norm(h)
        x_pred = torch.tanh(h_norm @ self.C)
        error = x - x_pred
        return x_pred, error

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project input through base matrix."""
        return x @ self.B_base

    def inject_error(self, error: torch.Tensor) -> torch.Tensor:
        """Inject error into hidden state."""
        return error @ self.W
