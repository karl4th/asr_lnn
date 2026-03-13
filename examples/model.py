"""
ASR Models with DREAM.

Two model variants:
1. DREAMASR - Standard DREAM Stack (no coordination)
2. CoordinatedDREAMASR - Coordinated DREAM Stack with hierarchical predictive coding
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dream import DREAMStack, DREAMConfig, DREAMCell
from dream.layers.coordination import CoordinatedDREAMStack


class DREAMASR(nn.Module):
    """
    ASR model with standard DREAM Stack.
    
    Architecture:
    Audio → Mel (80) → DREAM Stack (3 layers) → Linear(27) → CTC Loss
    
    Parameters
    ----------
    n_mels : int
        Number of Mel bins (input dimension)
    hidden_dims : list of int
        Hidden dimensions for each DREAM layer
    rank : int
        Fast weights rank
    num_classes : int
        Number of output classes (27: 26 letters + space)
    dropout : float
        Dropout between layers
    use_fast_weights : bool
        Enable fast weights (default: True)
    use_ltc : bool
        Enable Liquid Time-Constants (default: True)
    use_sleep : bool
        Enable sleep consolidation (default: True)
    freeze_fast_weights : bool
        Freeze fast weights during training (default: False)
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_dims: list = None,
        rank: int = 16,
        num_classes: int = 27,
        dropout: float = 0.1,
        use_fast_weights: bool = True,
        use_ltc: bool = True,
        use_sleep: bool = True,
        freeze_fast_weights: bool = False
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        self.n_mels = n_mels
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.num_classes = num_classes
        self.freeze_fast_weights = freeze_fast_weights
        
        # DREAM Stack
        self.dream = DREAMStack(
            input_dim=n_mels,
            hidden_dims=hidden_dims,
            rank=rank,
            dropout=dropout,
            use_fast_weights=use_fast_weights,
            use_ltc=use_ltc,
            use_sleep=use_sleep,
            freeze_fast_weights=freeze_fast_weights
        )
        
        # Output projection with LayerNorm
        self.output_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_proj = nn.Linear(hidden_dims[-1], num_classes)
        
        # Log softmax for CTC
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def set_fast_weights_mode(self, freeze: bool):
        """Freeze/unfreeze fast weights."""
        self.freeze_fast_weights = freeze
        self.dream.set_fast_weights_mode(freeze)
    
    def forward(
        self,
        mel_spec: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        mel_spec : torch.Tensor
            Mel spectrogram (batch, time, n_mels)
        mel_lengths : torch.Tensor, optional
            Lengths of mel spectrograms (batch,)
        
        Returns
        -------
        log_probs : torch.Tensor
            Log probabilities (batch, time, num_classes)
        out_lengths : torch.Tensor
            Output lengths (batch,)
        """
        batch_size, time_steps, _ = mel_spec.shape
        
        # DREAM Stack
        output, states = self.dream(mel_spec)
        
        # Output projection with LayerNorm
        output = self.output_norm(output)
        output = self.output_proj(output)
        
        # Log softmax
        log_probs = self.log_softmax(output)
        
        # Compute output lengths (same as input for DREAM)
        if mel_lengths is not None:
            out_lengths = mel_lengths
        else:
            out_lengths = torch.full(
                (batch_size,), time_steps,
                dtype=torch.long, device=mel_spec.device
            )
        
        return log_probs, out_lengths
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CoordinatedDREAMASR(nn.Module):
    """
    ASR model with Coordinated DREAM Stack.
    
    Architecture:
    Audio → Mel (80) → Coordinated DREAM Stack (3 layers) → Linear(27) → CTC Loss
    
    Features:
    - Top-down modulation (влияет на пластичность)
    - Hierarchical tau (верхние слои медленнее)
    - Inter-layer prediction loss (optional)
    
    Parameters
    ----------
    n_mels : int
        Number of Mel bins (input dimension)
    hidden_dims : list of int
        Hidden dimensions for each DREAM layer
    rank : int
        Fast weights rank
    num_classes : int
        Number of output classes (27: 26 letters + space)
    dropout : float
        Dropout between layers
    use_hierarchical_tau : bool
        Enable hierarchical tau (default: True)
    use_inter_layer_prediction : bool
        Enable inter-layer prediction loss (default: True)
    freeze_fast_weights : bool
        Freeze fast weights during training (default: False)
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_dims: list = None,
        rank: int = 16,
        num_classes: int = 27,
        dropout: float = 0.1,
        use_hierarchical_tau: bool = True,
        use_inter_layer_prediction: bool = True,
        freeze_fast_weights: bool = False
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        self.n_mels = n_mels
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.num_classes = num_classes
        self.freeze_fast_weights = freeze_fast_weights
        self.use_inter_layer_prediction = use_inter_layer_prediction
        
        # Coordinated DREAM Stack
        self.dream = CoordinatedDREAMStack(
            input_dim=n_mels,
            hidden_dims=hidden_dims,
            rank=rank,
            dropout=dropout,
            use_hierarchical_tau=use_hierarchical_tau,
            use_inter_layer_prediction=use_inter_layer_prediction,
            freeze_fast_weights=freeze_fast_weights
        )
        
        # Output projection: CoordinatedDREAMStack returns (batch, time, input_dim)
        # So we project from input_dim (n_mels=80) to num_classes
        self.output_proj = nn.Linear(n_mels, num_classes)
        
        # Log softmax for CTC
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def set_fast_weights_mode(self, freeze: bool):
        """Freeze/unfreeze fast weights."""
        self.freeze_fast_weights = freeze
        self.dream.set_fast_weights_mode(freeze)
    
    def forward(
        self,
        mel_spec: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None,
        return_losses: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Forward pass.
        
        Parameters
        ----------
        mel_spec : torch.Tensor
            Mel spectrogram (batch, time, n_mels)
        mel_lengths : torch.Tensor, optional
            Lengths of mel spectrograms (batch,)
        return_losses : bool
            Return inter-layer and reconstruction losses
        
        Returns
        -------
        log_probs : torch.Tensor
            Log probabilities (batch, time, num_classes)
        out_lengths : torch.Tensor
            Output lengths (batch,)
        losses : dict, optional
            Additional losses (reconstruction, inter_layer)
        """
        batch_size, time_steps, _ = mel_spec.shape
        
        # Coordinated DREAM Stack
        output, states, coord_losses = self.dream(
            mel_spec,
            return_losses=return_losses
        )
        # output shape: (batch, time, input_dim) = (batch, time, 80)
        
        # Output projection
        output = self.output_proj(output)
        
        # Log softmax
        log_probs = self.log_softmax(output)
        
        # Compute output lengths
        if mel_lengths is not None:
            out_lengths = mel_lengths
        else:
            out_lengths = torch.full(
                (batch_size,), time_steps,
                dtype=torch.long, device=mel_spec.device
            )
        
        losses = None
        if return_losses and coord_losses is not None:
            losses = {
                'reconstruction': coord_losses['reconstruction'],
                'inter_layer': coord_losses['inter_layer']
            }
        
        return log_probs, out_lengths, losses
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_type: str,
    n_mels: int = 80,
    hidden_dims: list = None,
    rank: int = 16,
    num_classes: int = 27,
    dropout: float = 0.1,
    use_fast_weights: bool = True,
    use_ltc: bool = True,
    use_sleep: bool = True,
    use_hierarchical_tau: bool = True,
    use_inter_layer_prediction: bool = True,
    freeze_fast_weights: bool = False
) -> nn.Module:
    """
    Factory function to create ASR model.
    
    Parameters
    ----------
    model_type : str
        'dream' or 'coordinated'
    n_mels : int
        Number of Mel bins
    hidden_dims : list
        Hidden dimensions for each layer
    rank : int
        Fast weights rank
    num_classes : int
        Number of output classes
    dropout : float
        Dropout rate
    use_fast_weights : bool
        Enable fast weights
    use_ltc : bool
        Enable LTC
    use_sleep : bool
        Enable sleep
    use_hierarchical_tau : bool
        Enable hierarchical tau (coordinated only)
    use_inter_layer_prediction : bool
        Enable inter-layer prediction (coordinated only)
    freeze_fast_weights : bool
        Freeze fast weights during training
    
    Returns
    -------
    model : nn.Module
        ASR model
    """
    if hidden_dims is None:
        hidden_dims = [256, 256, 256]
    
    if model_type == 'dream':
        model = DREAMASR(
            n_mels=n_mels,
            hidden_dims=hidden_dims,
            rank=rank,
            num_classes=num_classes,
            dropout=dropout,
            use_fast_weights=use_fast_weights,
            use_ltc=use_ltc,
            use_sleep=use_sleep,
            freeze_fast_weights=freeze_fast_weights
        )
    elif model_type == 'coordinated':
        model = CoordinatedDREAMASR(
            n_mels=n_mels,
            hidden_dims=hidden_dims,
            rank=rank,
            num_classes=num_classes,
            dropout=dropout,
            use_hierarchical_tau=use_hierarchical_tau,
            use_inter_layer_prediction=use_inter_layer_prediction,
            freeze_fast_weights=freeze_fast_weights
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test models
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dream", choices=["dream", "coordinated"])
    args = parser.parse_args()
    
    print(f"Testing {args.model} model...")
    
    model = create_model(args.model)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Forward pass
    batch_size = 4
    time_steps = 500
    n_mels = 80
    
    mel_spec = torch.randn(batch_size, time_steps, n_mels)
    mel_lengths = torch.tensor([500, 450, 480, 490])
    
    if args.model == 'dream':
        log_probs, out_lengths = model(mel_spec, mel_lengths)
    else:
        log_probs, out_lengths, losses = model(mel_spec, mel_lengths, return_losses=True)
        print(f"Losses: {losses}")
    
    print(f"Input: {mel_spec.shape}")
    print(f"Output: {log_probs.shape}")
    print(f"Output lengths: {out_lengths}")
    print(f"✓ Test passed!")
