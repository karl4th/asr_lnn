"""
Block 4: Liquid Time-Constants (LTC)

Adaptive integration speeds based on surprise:
- τ = τ_sys / (1 + S * scale)
- dh/dt = (-h + tanh(u_eff)) / τ

High surprise → small τ → fast updates
Low surprise → large τ → slow integration
"""

import torch
import torch.nn as nn


class LiquidTimeConstants(nn.Module):
    """
    Liquid Time-Constants block.

    Implements adaptive integration time constants modulated by surprise.

    Parameters
    ----------
    ltc_tau_sys : float
        Base system time constant (τ_sys)
    ltc_surprise_scale : float
        Scaling factor for surprise modulation
    time_step : float
        Integration time step (dt)
    ltc_enabled : bool
        If False, uses classic tanh update instead of LTC
    """

    def __init__(
        self,
        ltc_tau_sys: float = 5.0,
        ltc_surprise_scale: float = 10.0,
        time_step: float = 0.1,
        ltc_enabled: bool = True
    ):
        super().__init__()
        self.ltc_enabled = ltc_enabled
        self.time_step = time_step

        self.tau_sys = nn.Parameter(torch.tensor(ltc_tau_sys))
        self.tau_surprise_scale = nn.Parameter(torch.tensor(ltc_surprise_scale))

    def forward(
        self,
        h_prev: torch.Tensor,
        u_eff: torch.Tensor,
        surprise: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LTC update.

        Parameters
        ----------
        h_prev : torch.Tensor
            Previous hidden state (batch, hidden_dim)
        u_eff : torch.Tensor
            Effective input (batch, hidden_dim)
        surprise : torch.Tensor
            Surprise (batch,)

        Returns
        -------
        h_new : torch.Tensor
            Updated hidden state
        """
        if not self.ltc_enabled or self.tau_sys.item() < 0.01:
            # Classic update (no LTC)
            return torch.tanh(u_eff)

        # Dynamic time constant
        tau = self.tau_sys / (1.0 + surprise * self.tau_surprise_scale)
        tau = torch.clamp(tau, 0.01, 50.0)

        # Target state
        h_target = torch.tanh(u_eff)

        # Euler integration: dh/dt = (-h + h_target) / τ
        dt_over_tau = self.time_step / (tau.unsqueeze(1) + self.time_step)
        dt_over_tau = torch.clamp(dt_over_tau, 0.01, 0.5)

        # Interpolate between current and target
        h_new = (1 - dt_over_tau) * h_prev + dt_over_tau * h_target

        return h_new
