#!/usr/bin/env python3
"""
Continue Stage 2 from saved Stage 1 checkpoint.

Usage:
    python run_continue_stage2.py \
        --audio-dir /path/to/LJSpeech-1.1/wavs \
        --checkpoint sanday/checkpoints_2stage/stage1_pretrain.pt
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sanday.model import SandayASR
from sanday.phonemes import EnglishPhonemes
from sanday.data import LJSpeechDataset, create_dataloader


def main():
    parser = argparse.ArgumentParser(description='Continue Stage 2 from Stage 1 checkpoint')
    
    # Data
    parser.add_argument('--audio-dir', type=str, required=True)
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--max-audio-len', type=float, default=5.0)
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint (stage1_pretrain.pt)')
    
    # Stage 2
    parser.add_argument('--stage2-epochs', type=int, default=50)
    parser.add_argument('--stage2-lr', type=float, default=0.0001)
    
    # Common
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='sanday/checkpoints_2stage')
    
    args = parser.parse_args()
    
    # Device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Stage 1 checkpoint
    print(f"\n📥 Loading Stage 1 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"✅ Loaded Stage 1 (best loss: {min(checkpoint['loss']):.4f})")
    
    # Metadata path
    if args.metadata_path is None:
        args.metadata_path = str(Path(args.audio_dir).parent / 'metadata.csv')
    
    # Phonemes
    phonemes = EnglishPhonemes(use_stress=False)
    print(f"Phoneme vocabulary: {len(phonemes)} classes")
    
    # Dataset
    print(f"\nLoading dataset...")
    dataset = LJSpeechDataset(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata_path,
        phonemes=phonemes,
        max_audio_len=args.max_audio_len,
    )
    
    # Subset
    if args.subset is not None:
        n_samples = min(args.subset, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(n_samples))
        print(f"Using subset: {n_samples} samples")
    
    # Split
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}, Val: {n_val}")
    
    # DataLoaders
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Model (start with frozen fast weights for Stage 1 config)
    model = SandayASR(
        input_dim=80,
        hidden_dims=[256, 256, 256],
        rank=16,
        num_phonemes=len(phonemes),
        freeze_fast_weights=True,
    )
    
    # Load Stage 1 weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model weights loaded from Stage 1")
    
    # Move to device
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # ================================================================
    # STAGE 2: Adaptation Training
    # ================================================================
    
    print(f"\n{'='*70}")
    print(f"STAGE 2: ADAPTATION TRAINING ({args.stage2_epochs} epochs)")
    print(f"{'='*70}")
    print(f"  - Fast weights: ACTIVE (plasticity enabled)")
    print(f"  - Loss: CTC")
    print(f"  - LR: {args.stage2_lr}")
    print(f"{'='*70}\n")
    
    # Switch to adaptation mode
    model.switch_to_adaptation()
    model.train()  # Back to train mode!
    
    # Optimizer: ALL parameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.stage2_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    
    # CTC loss
    ctc_loss = nn.CTCLoss(blank=1, zero_infinity=True)
    
    history = {
        'stage2_loss': [],
        'stage2_val_loss': [],
    }
    
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.stage2_epochs):
        start_time = time.time()
        train_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            phonemes = batch['phonemes'].to(device)
            feat_lengths = batch['feature_lengths'].to(device)
            phoneme_lengths = batch['phoneme_lengths'].to(device)
            
            optimizer.zero_grad()
            
            # Forward with explicit stage='adaptation'
            outputs = model(features, stage='adaptation')
            
            # CTC loss
            log_probs = outputs['log_probs'].transpose(0, 1)  # (time, batch, num_phonemes)
            loss = ctc_loss(
                log_probs.log(),
                phonemes,
                feat_lengths,
                phoneme_lengths,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches
        
        # Validate
        model.eval()
        val_loss_total = 0.0
        val_n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                phonemes = batch['phonemes'].to(device)
                feat_lengths = batch['feature_lengths'].to(device)
                phoneme_lengths = batch['phoneme_lengths'].to(device)
                
                outputs = model(features, stage='adaptation')
                
                log_probs = outputs['log_probs'].transpose(0, 1)
                loss = ctc_loss(
                    log_probs.log(),
                    phonemes,
                    feat_lengths,
                    phoneme_lengths,
                )
                
                val_loss_total += loss.item()
                val_n_batches += 1
        
        val_loss = val_loss_total / val_n_batches
        model.train()
        
        history['stage2_loss'].append(avg_train_loss)
        history['stage2_val_loss'].append(val_loss)
        
        elapsed = time.time() - start_time
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = save_dir / 'stage2_adaptation_best.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch,
            }, save_path)
            print(f"  ✅ val_loss improved to {val_loss:.4f}")
        
        print(f"Epoch {epoch+1:3d}/{args.stage2_epochs}: train={avg_train_loss:.4f}, val={val_loss:.4f} ({elapsed:.1f}s)")
    
    # Save final checkpoint
    save_path = save_dir / 'stage2_adaptation_last.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': best_val_loss,
        'history': history,
    }, save_path)
    print(f"\n💾 Checkpoint saved: {save_path}")
    
    # Save history
    history_path = save_dir / 'stage2_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 History saved to: {history_path}")
    
    print(f"\n{'='*70}")
    print(f"STAGE 2 COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
