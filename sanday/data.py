"""
Data loading for LJSpeech dataset with phoneme labels.
"""

import torch
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader

from .phonemes import EnglishPhonemes


class LJSpeechDataset(Dataset):
    """
    LJSpeech dataset with phoneme labels.
    
    Loads audio files and converts text to phonemes using g2p-en.
    """
    
    def __init__(self,
                 audio_dir: str,
                 metadata_path: str,
                 phonemes: EnglishPhonemes,
                 target_sr: int = 16000,
                 n_mels: int = 80,
                 hop_length: int = 256,
                 n_fft: int = 1024,
                 max_audio_len: Optional[float] = None,  # in seconds
                 ):
        """
        Initialize LJSpeech dataset.
        
        Parameters
        ----------
        audio_dir : str
            Directory containing LJSpeech wav files
        metadata_path : str
            Path to metadata.csv
        phonemes : EnglishPhonemes
            Phoneme converter
        target_sr : int
            Target sample rate
        n_mels : int
            Number of mel bins
        hop_length : int
            Hop length for mel spectrogram
        n_fft : int
            FFT window size
        max_audio_len : float, optional
            Maximum audio length in seconds (longer files will be skipped)
        """
        self.audio_dir = Path(audio_dir)
        self.phonemes = phonemes
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Load metadata
        self.metadata = pd.read_csv(
            metadata_path,
            sep='|',
            header=None,
            names=['id', 'text', 'normalized_text']
        )
        
        # Filter by audio length (optional)
        if max_audio_len is not None:
            max_samples = int(max_audio_len * target_sr)
            valid_ids = []
            
            for idx, row in self.metadata.iterrows():
                audio_file = self.audio_dir / f"{row['id']}.wav"
                if audio_file.exists():
                    # Quick length check
                    y, sr = librosa.load(str(audio_file), sr=target_sr, duration=1)
                    if len(y) * (audio_file.stat().st_size / len(y.tobytes())) <= max_samples * 4:
                        valid_ids.append(idx)
            
            self.metadata = self.metadata.loc[valid_ids].reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} samples from LJSpeech")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_file = self.audio_dir / f"{row['id']}.wav"
        y, sr = librosa.load(str(audio_file), sr=self.target_sr)
        
        # Extract mel spectrogram
        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        log_mels = librosa.power_to_db(melspec, ref=np.max)
        
        # Convert to tensor and normalize
        feat = torch.tensor(log_mels.T, dtype=torch.float32)
        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        
        # Convert text to phonemes
        text = row['normalized_text']
        phoneme_ids = self.phonemes.text_to_ids(text)
        
        # Convert to tensor
        phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.long)
        
        return feat, phoneme_tensor, row['id']


def phoneme_collate_fn(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
    """
    Collate function for phoneme dataset.
    
    Pads audio features and phoneme sequences to maximum length in batch.
    
    Parameters
    ----------
    batch : List[Tuple]
        List of (feat, phoneme_ids, audio_id) tuples
    
    Returns
    -------
    batch_dict : Dict[str, torch.Tensor]
        Dictionary with padded tensors and lengths
    """
    feats, phonemes, ids = zip(*batch)
    
    # Get max lengths
    max_feat_len = max(f.shape[0] for f in feats)
    max_phoneme_len = max(p.shape[0] for p in phonemes)
    
    # Pad features
    feat_dim = feats[0].shape[1]
    feats_padded = torch.zeros(len(feats), max_feat_len, feat_dim)
    feat_lengths = torch.zeros(len(feats), dtype=torch.long)
    
    for i, f in enumerate(feats):
        feats_padded[i, :f.shape[0], :] = f
        feat_lengths[i] = f.shape[0]
    
    # Pad phonemes (pad with 0 = <pad>)
    phonemes_padded = torch.zeros(len(phonemes), max_phoneme_len, dtype=torch.long)
    phoneme_lengths = torch.zeros(len(phonemes), dtype=torch.long)
    
    for i, p in enumerate(phonemes):
        phonemes_padded[i, :p.shape[0]] = p
        phoneme_lengths[i] = p.shape[0]
    
    return {
        'features': feats_padded,
        'feature_lengths': feat_lengths,
        'phonemes': phonemes_padded,
        'phoneme_lengths': phoneme_lengths,
        'ids': list(ids),
    }


def create_dataloader(dataset: LJSpeechDataset,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     ) -> DataLoader:
    """
    Create DataLoader for LJSpeech dataset.
    
    Parameters
    ----------
    dataset : LJSpeechDataset
        The dataset
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle
    num_workers : int
        Number of data loading workers
    
    Returns
    -------
    dataloader : DataLoader
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=phoneme_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


# Example usage
if __name__ == "__main__":
    from phonemes import EnglishPhonemes
    
    phonemes = EnglishPhonemes(use_stress=False)
    
    dataset = LJSpeechDataset(
        audio_dir="/path/to/LJSpeech-1.1/wavs",
        metadata_path="/path/to/LJSpeech-1.1/metadata.csv",
        phonemes=phonemes,
        max_audio_len=5.0,  # Only load files < 5 seconds
    )
    
    loader = create_dataloader(dataset, batch_size=4)
    
    for batch in loader:
        print(f"Features: {batch['features'].shape}")
        print(f"Phonemes: {batch['phonemes'].shape}")
        print(f"Feature lengths: {batch['feature_lengths']}")
        print(f"Phoneme lengths: {batch['phoneme_lengths']}")
        break
