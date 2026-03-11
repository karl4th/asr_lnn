#!/usr/bin/env python3
"""
Wrapper script to run SANDAY training.

Usage:
    python run_train.py --audio-dir /path/to/LJSpeech-1.1/wavs
"""

import sys
import os

# Download NLTK data FIRST (before g2p-en tries to use it)
print("[SANDAY] Downloading required NLTK data...")
import nltk
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)
print("[SANDAY] NLTK data ready!")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import and run
from sanday.train import main

if __name__ == '__main__':
    main()
