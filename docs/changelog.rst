Changelog
=========

All notable changes to DREAM will be documented in this file.

[0.1.3] - 2024-03-03
--------------------

Added
~~~~~
- Per-batch U matrices for independent memory per example
- Learnable LTC parameters (tau_sys, ltc_surprise_scale)
- Sleep consolidation for memory stabilization
- Truncated BPTT for memory efficiency
- DREAMStack for multi-layer models
- forward_with_mask for padded sequence processing
- Full Sphinx documentation for ReadTheDocs

Changed
~~~~~~~
- State management: state now persists between epochs
- Forgetting rate default changed to 0.01 for homeostasis
- Improved gradient flow with single backward per epoch

Fixed
~~~~~
- Multiple backward() error in training loops
- Memory leak from not detaching state between segments
- LTC numerical instability with proper clamping

[0.1.2] - 2024-03-02
--------------------

Added
~~~~~
- Initial PyPI release
- Unit tests (17 tests)
- Audio overfitting test
- GitHub Actions CI/CD

[0.1.1] - 2024-03-01
--------------------

Added
~~~~~
- DREAMCell core implementation
- DREAM high-level API
- Basic documentation

Version Numbering
-----------------

We use semantic versioning: ``MAJOR.MINOR.PATCH``

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)
