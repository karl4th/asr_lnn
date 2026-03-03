Installation
============

This section describes various ways to install DREAM.

Requirements
------------

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

Installation via pip
--------------------

The simplest way:

.. code-block:: bash

   pip install dreamnn

Installation with optional dependencies:

.. code-block:: bash

   # For audio processing
   pip install dreamnn[audio]
   
   # For development
   pip install dreamnn[dev]

Installation from Source
------------------------

From GitHub:

.. code-block:: bash

   git clone https://github.com/karl4th/dream-nn.git
   cd dream-nn
   pip install -e .

Or with optional dependencies:

.. code-block:: bash

   pip install -e ".[audio,dev]"

Installation in Google Colab
----------------------------

.. code-block:: python

   !pip install dreamnn librosa matplotlib
   
   from dream import DREAM, DREAMConfig
   print("✅ DREAM installed!")

Verifying Installation
----------------------

.. code-block:: python

   import torch
   from dream import DREAM, DREAMConfig
   
   # Create model
   config = DREAMConfig(input_dim=39, hidden_dim=256, rank=16)
   model = DREAM(**config.__dict__)
   
   # Test forward pass
   x = torch.randn(2, 50, 39)
   output, state = model(x)
   
   print(f"✅ Model output shape: {output.shape}")
   print(f"✅ State hidden shape: {state.h.shape}")

Troubleshooting
---------------

**Error: No module named 'dream'**

Make sure you installed the correct package name:

.. code-block:: bash

   pip install dreamnn  # NOT dream-nn (that's the PyPI project name)

**Error: CUDA not available**

Check your PyTorch version:

.. code-block:: python

   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())

If needed, reinstall PyTorch with CUDA support:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu118
