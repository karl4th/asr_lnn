# DREAM Documentation

**Dynamic Recall and Elastic Adaptive Memory**

A PyTorch library for adaptive sequence modeling with surprise-driven plasticity and liquid time-constants.

## Quick Navigation

- :doc:`installation` — Installation guide
- :doc:`quickstart` — Quick start tutorial
- :doc:`api` — API reference
- :doc:`architecture` — Architecture details
- :doc:`examples` — Usage examples

## Key Features

🧠 **Surprise-Driven Plasticity**
   Learning through Hebbian plasticity modulated by prediction error surprise

⏱️ **Liquid Time-Constants (LTC)**
   Adaptive integration speed based on signal novelty

🔁 **Fast Weights**
   Low-rank decomposition for efficient meta-learning

😴 **Sleep Consolidation**
   Memory stabilization during "sleep" phases

📦 **Batch Support**
   Efficient batch processing with independent memory per example

## Installation

```bash
pip install dreamnn
```

## Quick Example

```python
from dream import DREAM, DREAMConfig
import torch

# Configuration
config = DREAMConfig(
    input_dim=39,
    hidden_dim=256,
    rank=16,
    ltc_enabled=True,
)

# Model
model = DREAM(
    input_dim=39,
    hidden_dim=256,
    rank=16,
)

# Process sequence
x = torch.randn(4, 100, 39)  # (batch, time, features)
output, state = model(x)

print(f"Output shape: {output.shape}")
```

## Documentation Contents

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   architecture
   examples
   changelog

## Links

* `GitHub Repository <https://github.com/karl4th/dream-nn>`_
* `PyPI Package <https://pypi.org/project/dreamnn/>`_
* `Technical Report <https://github.com/karl4th/dream-nn/blob/main/tech_report.md>`_

## Indices and Tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
