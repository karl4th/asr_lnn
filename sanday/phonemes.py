"""
Phoneme handling for English phonemes.

Uses g2p-en for grapheme-to-phoneme conversion.
"""

import re
from typing import List, Dict, Optional, Tuple
import torch


class EnglishPhonemes:
    """
    English phoneme vocabulary with g2p-en conversion.
    
    ARPABET phonemes (44 total):
    Vowels (15): AA, AE, AH, AO, AW, AX, AXR, AY, EH, ER, EY, IH, IX, OW, OY, UH, UW
    Consonants (25): B, CH, D, DH, DX, DZ, F, G, HH, HV, JH, K, L, M, N, NG, NX, P, Q, R, S, SH, T, TH, V, W, Y, Z, ZH
    
    Stress markers: 0 (no stress), 1 (primary), 2 (secondary)
    """
    
    # ARPABET phonemes (without stress numbers)
    VOWELS = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 
              'EH', 'ER', 'EY', 'IH', 'IX', 'OW', 'OY', 'UH', 'UW']
    
    CONSONANTS = ['B', 'CH', 'D', 'DH', 'DX', 'DZ', 'F', 'G', 'HH', 'HV',
                  'JH', 'K', 'L', 'M', 'N', 'NG', 'NX', 'P', 'Q', 'R', 
                  'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    
    # Special tokens
    PAD = '<pad>'
    BLANK = '<blank>'  # For CTC
    BOS = '<bos>'      # Beginning of sequence
    EOS = '<eos>'      # End of sequence
    
    def __init__(self, use_stress: bool = False):
        """
        Initialize phoneme vocabulary.
        
        Parameters
        ----------
        use_stress : bool
            If True, include stress markers in vocabulary.
            For now, we ignore stress (use_stress=False).
        """
        self.use_stress = use_stress
        
        # Build vocabulary
        self.phonemes = [self.PAD, self.BLANK, self.BOS, self.EOS]
        
        # Add phonemes (without stress for now)
        self.phonemes.extend(self.CONSONANTS + self.VOWELS)
        
        # Create mappings
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phonemes)}
        self.id_to_phoneme = {i: p for i, p in enumerate(self.phonemes)}
        
        self.num_phonemes = len(self.phonemes)
        
        # g2p converter (lazy loading)
        self._g2p = None
    
    @property
    def g2p(self):
        """Lazy load g2p converter."""
        if self._g2p is None:
            try:
                from g2p_en import G2p
                self._g2p = G2p()
                
                # Download required NLTK data if not present
                import nltk
                try:
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                except LookupError:
                    print("[g2p] Downloading NLTK averaged_perceptron_tagger...")
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                
                try:
                    nltk.data.find('corpora/cmudict')
                except LookupError:
                    print("[g2p] Downloading NLTK cmudict...")
                    nltk.download('cmudict', quiet=True)
                    
            except ImportError:
                raise ImportError(
                    "g2p-en not installed. Install with: pip install g2p-en"
                )
            except LookupError as e:
                # Handle g2p-en specific NLTK errors
                if "averaged_perceptron_tagger_eng" in str(e):
                    # Use standard tagger instead
                    import nltk
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                raise
                    
        return self._g2p
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phonemes using g2p-en.
        
        Parameters
        ----------
        text : str
            Input text (e.g., "The quick brown fox")
        
        Returns
        -------
        phonemes : List[str]
            List of phonemes (e.g., ['DH', 'AH0', 'K', 'W', 'IH1', 'K', ...])
        """
        # g2p-en returns phonemes with stress markers
        phonemes_with_stress = self.g2p(text)
        
        if self.use_stress:
            # Keep stress markers
            return phonemes_with_stress
        else:
            # Remove stress markers (convert AH0 → AH, IH1 → IH, etc.)
            phonemes_no_stress = []
            for p in phonemes_with_stress:
                # Remove stress number (0, 1, 2)
                p_clean = re.sub(r'[0-2]$', '', p)
                if p_clean:
                    phonemes_no_stress.append(p_clean)
            return phonemes_no_stress
    
    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """
        Convert phoneme list to ID list.
        
        Parameters
        ----------
        phonemes : List[str]
            List of phonemes
        
        Returns
        -------
        ids : List[int]
            List of phoneme IDs
        """
        ids = []
        for p in phonemes:
            if p in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[p])
            else:
                # Unknown phoneme → skip or use UNK
                pass
        return ids
    
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        """
        Convert ID list to phoneme list.
        
        Parameters
        ----------
        ids : List[int]
            List of phoneme IDs
        
        Returns
        -------
        phonemes : List[str]
            List of phonemes
        """
        return [self.id_to_phoneme[i] for i in ids if i in self.id_to_phoneme]
    
    def text_to_ids(self, text: str) -> List[int]:
        """
        Convert text directly to phoneme IDs.
        
        Parameters
        ----------
        text : str
            Input text
        
        Returns
        -------
        ids : List[int]
            List of phoneme IDs
        """
        phonemes = self.text_to_phonemes(text)
        return self.phonemes_to_ids(phonemes)
    
    def decode_ctc_output(self, 
                         ctc_output: torch.Tensor, 
                         blank_id: int = 1) -> List[str]:
        """
        Decode CTC output to phonemes (greedy decoding).
        
        Parameters
        ----------
        ctc_output : torch.Tensor
            CTC output logits (time, num_phonemes)
        blank_id : int
            ID of blank token (default: 1)
        
        Returns
        -------
        phonemes : List[str]
            Decoded phonemes (with duplicates and blanks removed)
        """
        # Greedy decoding
        pred_ids = ctc_output.argmax(dim=-1).cpu().tolist()
        
        # Remove blanks and duplicates
        prev_id = blank_id
        result = []
        for curr_id in pred_ids:
            if curr_id != blank_id and curr_id != prev_id:
                result.append(curr_id)
            prev_id = curr_id
        
        return self.ids_to_phonemes(result)
    
    def __len__(self):
        return self.num_phonemes
    
    def __repr__(self):
        return (f"EnglishPhonemes(num_phonemes={self.num_phonemes}, "
                f"use_stress={self.use_stress})")


# Example usage
if __name__ == "__main__":
    phonemes = EnglishPhonemes(use_stress=False)
    
    text = "The quick brown fox"
    phoneme_list = phonemes.text_to_phonemes(text)
    phoneme_ids = phonemes.text_to_ids(text)
    
    print(f"Text: {text}")
    print(f"Phonemes: {phoneme_list}")
    print(f"Phoneme IDs: {phoneme_ids}")
    print(f"Decoded: {phonemes.ids_to_phonemes(phoneme_ids)}")
    print(f"Vocabulary size: {len(phonemes)}")
