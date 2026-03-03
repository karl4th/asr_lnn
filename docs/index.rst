DREAM Documentation
===================

.. image:: https://img.shields.io/badge/version-0.1.3-blue
   :target: https://pypi.org/project/dreamnn/

**Dynamic Recall and Elastic Adaptive Memory**

A PyTorch library for adaptive sequence modeling with surprise-driven plasticity and liquid time-constants.

.. note::
   This documentation is for version 0.1.3. Latest version available on `PyPI <https://pypi.org/project/dreamnn/>`_.

Quick Example
-------------

.. code-block:: python

   from dream import DREAM, DREAMConfig
   import torch
   
   # Create model
   model = DREAM(input_dim=39, hidden_dim=256, rank=16)
   
   # Process sequence
   x = torch.randn(4, 100, 39)  # batch=4, time=100, features=39
   output, state = model(x)
   
   print(f"Output: {output.shape}")  # (4, 100, 256)

Installation
------------

.. code-block:: bash

   pip install dreamnn

For audio processing support:

.. code-block:: bash

   pip install dreamnn[audio]

Key Features
------------

🧠 **Surprise-Driven Plasticity**
   Learning through Hebbian plasticity modulated by prediction error surprise.

⏱️ **Liquid Time-Constants**
   Adaptive integration speed based on signal novelty.

🔁 **Fast Weights**
   Low-rank decomposition for efficient meta-learning.

😴 **Sleep Consolidation**
   Memory stabilization during sleep phases.

📦 **Batch Support**
   Independent memory for each batch element.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   architecture

.. toctree::
   :maxdepth: 1
   :caption: Additional Info

   changelog

Useful Links
------------

* `GitHub Repository <https://github.com/karl4th/dream-nn>`_
* `PyPI Package <https://pypi.org/project/dreamnn/>`_
* `Technical Report <https://github.com/karl4th/dream-nn/blob/main/tech_report.md>`_
* `Issue Tracker <https://github.com/karl4th/dream-nn/issues>`_
