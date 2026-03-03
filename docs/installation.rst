Installation
============

Requirements
------------

* Python >= 3.10
* PyTorch >= 2.0.0
* NumPy >= 1.24.0

Quick Install
-------------

.. code-block:: bash

   pip install dreamnn

With Optional Dependencies
--------------------------

For audio processing:

.. code-block:: bash

   pip install dreamnn[audio]

For development:

.. code-block:: bash

   pip install dreamnn[dev]

From Source
-----------

.. code-block:: bash

   git clone https://github.com/karl4th/dream-nn.git
   cd dream-nn
   pip install -e .

Google Colab
------------

.. code-block:: python

   !pip install dreamnn librosa matplotlib
   
   from dream import DREAM, DREAMConfig
   print("DREAM installed!")

Verify Installation
-------------------

.. code-block:: python

   import torch
   from dream import DREAM
   
   model = DREAM(input_dim=39, hidden_dim=256, rank=16)
   x = torch.randn(2, 50, 39)
   output, state = model(x)
   
   print(f"Success! Output shape: {output.shape}")
