Examples
========

ASR: Speech Recognition
-----------------------

.. code-block:: python

   import librosa
   from dream import DREAMStack
   
   # Load audio
   audio, sr = librosa.load("speech.wav", sr=16000)
   
   # MFCC 39D
   mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
   delta = librosa.feature.delta(mfcc, order=1)
   delta_delta = librosa.feature.delta(mfcc, order=2)
   features = np.vstack([mfcc, delta, delta_delta]).T
   
   # Model
   model = DREAMStack(input_dim=39, hidden_dims=[256, 512, 256])
   output, states = model(torch.tensor(features).unsqueeze(0))

Time Series Prediction
----------------------

.. code-block:: python

   from dream import DREAM
   import torch.nn as nn
   
   model = DREAM(input_dim=10, hidden_dim=128, rank=8)
   head = nn.Linear(128, 1)
   
   x = torch.randn(32, 200, 10)
   y = torch.randn(32, 200, 1)
   
   output, state = model(x)
   loss = nn.functional.mse_loss(head(output), y)

Anomaly Detection
-----------------

.. code-block:: python

   model = DREAM(input_dim=10, hidden_dim=128)
   
   # Train on normal data
   for epoch in range(20):
       output, state = model(normal_data)
       # ...
   
   # Detect anomalies
   state = model.init_state(1)
   with torch.no_grad():
       _, _, _, surprise = model.cell(x, state.h)
   
   if surprise > threshold:
       print("Anomaly detected!")

Multi-Layer (DREAMStack)
------------------------

.. code-block:: python

   from dream import DREAMStack
   
   stack = DREAMStack(
       input_dim=39,
       hidden_dims=[128, 256, 128],
       rank=16,
       dropout=0.1,
   )
   
   x = torch.randn(4, 100, 39)
   output, states = stack(x)
