# Changelog

All notable changes to DREAM (Dynamic Recall and Elastic Adaptive Memory) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.3] - 2026-03-08

### Added
- **Optimized Cell** — `DREAMCellOptimized` with fused operations for 1.5-2.5x speedup
- **Mixed Precision (AMP)** — `DREAMCellAMP` for 2-3x speedup on Tensor Core GPUs
- **Performance Utilities** — Benchmarking, memory profiling, recommendations
- **Examples** — 5 practical examples in `examples/` directory
- **Documentation** — Complete developer documentation (`documentation_en_v0_1_3.md`)
- **Freeze Fast Weights Flag** — `freeze_fast_weights` parameter for two-phase training
- **Auto Mode Switching** — `model.train()` auto-freezes fast weights, `model.eval()` unfreezes

### Changed
- **Relative Surprise** — Improved anomaly detection with relative error instead of absolute
- **Optimized Parameters** — Tuned defaults for audio/speech tasks
- **Fast Weights Update** — Fixed and optimized STDP update logic

### Fixed
- Fast weights update logic in `update_fast_weights()`
- State management in bidirectional processing
- Documentation consistency

### Performance
- **T4 GPU:** 15ms → 5ms latency (3x speedup with AMP)
- **A100 GPU:** 5ms → 1.5ms latency (3.3x speedup with AMP + TF32)
- **CPU:** 50ms → 30ms latency (1.7x speedup)

---

## [0.1.2] - 2026-03-05

### Added
- Benchmark suite with 3 tests (ASR, Speaker Adaptation, Noise Robustness)
- Visualization tools for benchmark results
- Technical reports (TECHNICAL_REPORT.md, second.md)

### Changed
- Updated default parameters for better convergence

---

## [0.1.1] - 2026-02-20

### Added
- Basic DREAM cell implementation
- High-level API (DREAM, DREAMStack)
- Unit tests (17 tests)
- PyPI publication

---

## [0.1.0] - 2026-02-18

### Added
- Initial release
- NNAI-S architecture implementation
- Predictive coding + STDP + LTC
