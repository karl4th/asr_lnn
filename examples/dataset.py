"""
LJSpeech Dataset for ASR with CTC.

Loads audio files and transcriptions, extracts Mel spectrograms.
Supports subset mode for quick experiments.
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import csv


class LJSpeechCTCDataset(Dataset):
    """
    LJSpeech dataset with CTC-compatible text processing.
    
    - Resamples audio to 16kHz
    - Extracts 80-dim Mel spectrograms
    - Converts text to character-level tokens (26 letters + space = 27 classes)
    
    Parameters
    ----------
    root_dir : str
        Path to LJSpeech directory (contains wavs/ and metadata.csv)
    sample_rate : int
        Target sample rate (16000)
    n_mels : int
        Number of Mel bins (80)
    hop_length : int
        Hop length in samples (160 = 10ms at 16kHz)
    n_fft : int
        FFT window size (512 = 32ms)
    subset : int, optional
        Limit to first N samples (for quick experiments)
    """
    
    # Character vocabulary: 26 letters + space = 27 classes
    # CTC blank token is index 0 (added automatically by CTC loss)
    CHARS = " abcdefghijklmnopqrstuvwxyz"
    CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARS)}
    IDX_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}
    
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,
        n_fft: int = 512,
        subset: Optional[int] = None
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.subset = subset
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            window_fn=torch.hann_window
        )
        
        # Load metadata
        self.samples = self._load_metadata()
        
        print(f"LJSpeech dataset loaded: {len(self.samples)} samples")
        if subset:
            print(f"  (subset mode: {subset} files)")
    
    def _load_metadata(self) -> List[Tuple[str, str]]:
        """Load metadata.csv and filter to subset."""
        metadata_path = os.path.join(self.root_dir, "metadata.csv")
        
        samples = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 3:
                    file_id = row[0]
                    text = row[2].lower()  # Use normalized text
                    samples.append((file_id, text))
        
        # Filter to subset
        if self.subset and self.subset > 0:
            samples = samples[:self.subset]
        
        return samples
    
    def _text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Convert text to token IDs.
        
        - Lowercase
        - Keep only a-z and space
        - Map to 0-26 (space=0, a=1, ..., z=26)
        """
        tokens = []
        for ch in text:
            if ch in self.CHAR_TO_IDX:
                tokens.append(self.CHAR_TO_IDX[ch])
            # Skip unknown characters (punctuation, digits, etc.)
        
        if len(tokens) == 0:
            # Fallback for empty text
            tokens = [self.CHAR_TO_IDX[' ']]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _load_audio(self, file_id: str) -> torch.Tensor:
        """Load and resample audio file."""
        wav_path = os.path.join(self.root_dir, "wavs", f"{file_id}.wav")
        
        waveform, sr = torchaudio.load(wav_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze(0)  # (time,)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Get a single sample.
        
        Returns
        -------
        mel_spec : torch.Tensor
            Mel spectrogram (time, n_mels)
        tokens : torch.Tensor
            Token IDs (text_length,)
        mel_length : int
            Length of mel spectrogram (for CTC)
        token_length : int
            Length of tokens (for CTC)
        """
        file_id, text = self.samples[idx]
        
        # Load audio
        waveform = self._load_audio(file_id)
        
        # Extract Mel spectrogram
        with torch.no_grad():
            mel_spec = self.mel_transform(waveform.unsqueeze(0))
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # (time, n_mels)
        
        # Add small epsilon for numerical stability
        mel_spec = torch.log(mel_spec + 1e-6)
        
        # Normalize (per-speaker stats would be better, but simple normalization for now)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        
        # Convert text to tokens
        tokens = self._text_to_tokens(text)
        
        return mel_spec, tokens, mel_spec.shape[0], len(tokens)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Pads mel spectrograms and tokens to maximum length in batch.
    Returns lengths for CTC loss.
    """
    mel_specs, tokens, mel_lengths, token_lengths = zip(*batch)
    
    # Get max lengths
    max_mel_len = max(mel_lengths)
    max_token_len = max(token_lengths)
    
    # Pad mel spectrograms
    mel_specs_padded = torch.zeros(len(mel_specs), max_mel_len, mel_specs[0].shape[1])
    for i, mel in enumerate(mel_specs):
        mel_specs_padded[i, :mel.shape[0], :] = mel
    
    # Pad tokens
    tokens_padded = torch.zeros(len(tokens), max_token_len, dtype=torch.long)
    for i, tok in enumerate(tokens):
        tokens_padded[i, :len(tok)] = tok
    
    # Length tensors
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)
    token_lengths = torch.tensor(token_lengths, dtype=torch.long)
    
    return mel_specs_padded, tokens_padded, mel_lengths, token_lengths


def create_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    subset: Optional[int] = None,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Parameters
    ----------
    root_dir : str
        Path to LJSpeech directory
    batch_size : int
        Batch size
    num_workers : int
        Number of data loading workers
    subset : int, optional
        Limit dataset size
    val_split : float
        Fraction of data for validation
    
    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """
    # Full dataset
    dataset = LJSpeechCTCDataset(
        root_dir=root_dir,
        subset=subset
    )
    
    # Split into train/val
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/content/drive/MyDrive/dream/dataset/ljspeech")
    parser.add_argument("--subset", type=int, default=10)
    args = parser.parse_args()
    
    dataset = LJSpeechCTCDataset(args.root, subset=args.subset)
    
    print(f"\nVocabulary: {len(dataset.CHARS)} classes")
    print(f"Characters: '{dataset.CHARS}'")
    
    # Test loading
    mel, tokens, mel_len, token_len = dataset[0]
    print(f"\nSample 0:")
    print(f"  Mel spec: {mel.shape}")
    print(f"  Tokens: {tokens.tolist()}")
    print(f"  Text: '{''.join(dataset.IDX_TO_CHAR[t.item()] for t in tokens)}'")
