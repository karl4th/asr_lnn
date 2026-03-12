"""
DREAM ASR Examples.

Scripts for training ASR models with DREAM on LJSpeech dataset.

Usage:
------
# Standard DREAM
python examples/train.py --root /path/to/ljspeech --log-dir /path/to/logs

# Coordinated DREAM
python examples/train.py --root /path/to/ljspeech --model coordinated

# Disable fast weights (ablation)
python examples/train.py --root /path/to/ljspeech --no-fast-weights
"""
