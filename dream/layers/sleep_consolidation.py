"""
Block 5: Sleep Consolidation

Memory stabilization during high surprise:
- U_target updated when avg_surprise > S_min
- Consolidates fast weights into long-term memory
"""

import torch
import torch.nn as nn


class SleepConsolidation(nn.Module):
    """
    Sleep Consolidation block.

    Consolidates fast weights (U) into target weights (U_target)
    during periods of high surprise (simulating sleep).

    Parameters
    ----------
    sleep_rate : float
        Sleep consolidation rate (ζ_sleep)
    min_surprise_for_sleep : float
        Minimum surprise threshold for sleep (S_min)
    target_norm : float
        Target norm for homeostasis
    """

    def __init__(
        self,
        sleep_rate: float = 0.005,
        min_surprise_for_sleep: float = 0.2,
        target_norm: float = 2.0
    ):
        super().__init__()
        self.sleep_rate = sleep_rate
        self.S_min = min_surprise_for_sleep
        self.target_norm = target_norm

    def forward(
        self,
        U: torch.Tensor,
        U_target: torch.Tensor,
        avg_surprise: torch.Tensor
    ) -> torch.Tensor:
        """
        Update target weights if surprise is high enough.

        Parameters
        ----------
        U : torch.Tensor
            Current fast weights (batch, hidden_dim, rank)
        U_target : torch.Tensor
            Target fast weights (batch, hidden_dim, rank)
        avg_surprise : torch.Tensor
            Average surprise (batch,)

        Returns
        -------
        U_target_new : torch.Tensor
            Updated target weights
        """
        avg_surprise_mean = avg_surprise.mean()

        if avg_surprise_mean > self.S_min:
            # Consolidate U into U_target
            dU_target = self.sleep_rate * avg_surprise_mean * (U - U_target)
            U_target_new = U_target + dU_target

            # Homeostasis
            U_target_norm = U_target_new.norm(dim=(1, 2), keepdim=True)
            scale = (self.target_norm / (U_target_norm + 1e-6)).clamp(max=2.0)
            U_target_new = U_target_new * scale

            return U_target_new

        return U_target
