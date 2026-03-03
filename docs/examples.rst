Examples
========

Practical usage examples for DREAM.

1. ASR: Phoneme Recognition
---------------------------

Extracting MFCC features:

.. code-block:: python

   import librosa
   from dream import DREAM, DREAMConfig
   
   # Load audio
   audio, sr = librosa.load("speech.wav", sr=16000)
   
   # MFCC 39D (13 + 13Δ + 13ΔΔ)
   mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
   delta = librosa.feature.delta(mfcc, order=1)
   delta_delta = librosa.feature.delta(mfcc, order=2)
   features = np.vstack([mfcc, delta, delta_delta]).T
   
   # Normalization
   features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
   features = np.clip(features, -3.0, 3.0)

ASR Model:

.. code-block:: python

   config = DREAMConfig(
       input_dim=39,        # MFCC 39D
       hidden_dim=512,      # Large capacity
       rank=16,
       forgetting_rate=0.01,
       base_plasticity=2.0,
       ltc_enabled=True,
   )
   
   model = DREAMStack(
       input_dim=39,
       hidden_dims=[256, 512, 256],  # 3 layers
       rank=16,
       dropout=0.1,
   )

2. Time Series Prediction
-------------------------

Predicting time series:

.. code-block:: python

   from dream import DREAM
   import torch
   
   # Data: (batch, time, features)
   x = torch.randn(32, 200, 10)  # 10 features, 200 steps
   y = torch.randn(32, 200, 1)   # Predict 1 value
   
   # Model
   model = DREAM(
       input_dim=10,
       hidden_dim=128,
       rank=8,
       ltc_enabled=True,
       ltc_tau_sys=5.0,  # Fast response
   )
   
   head = torch.nn.Linear(128, 1)
   optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
   
   # Training
   for epoch in range(50):
       optimizer.zero_grad()
       output, state = model(x)
       loss = torch.nn.functional.mse_loss(head(output), y)
       loss.backward()
       optimizer.step()
       print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

3. Online Learning
------------------

Learning without gradient descent (only fast weights):

.. code-block:: python

   from dream import DREAM, DREAMConfig
   
   config = DREAMConfig(
       input_dim=39,
       hidden_dim=256,
       rank=16,
       forgetting_rate=0.0,    # No forgetting
       base_plasticity=3.0,    # High plasticity
       ltc_enabled=True,
   )
   
   model = DREAM(**config.__dict__)
   state = model.init_state(batch_size=1)
   
   # Online learning (no .backward())
   for x in data_stream:
       x_tensor = torch.tensor(x).unsqueeze(0)  # (1, input_dim)
       
       # Forward with fast weights update
       h_new, state = model(x_tensor, state)
       
       # Fast weights (U) update via Hebbian learning!
       # Surprise shows pattern novelty

4. Anomaly Detection
--------------------

Detecting anomalies via surprise:

.. code-block:: python

   from dream import DREAM, DREAMConfig
   import torch
   
   # Train on normal data
   model = DREAM(input_dim=10, hidden_dim=128, rank=8)
   
   for epoch in range(20):
       output, state = model(normal_data)
       # Training...
   
   # Anomaly detection
   model.eval()
   state = model.init_state(batch_size=1)
   
   surprises = []
   with torch.no_grad():
       for x in test_data:
           x_tensor = torch.tensor(x).unsqueeze(0)
           h, x_pred, error_norm, surprise = model.cell(x_tensor, state.h)
           surprises.append(surprise.item())
   
   # Anomalies have high surprise
   anomalies = [i for i, s in enumerate(surprises) if s > threshold]

5. Memory Retention Test
------------------------

Testing memorization capability:

.. code-block:: python

   from dream import DREAM
   import torch
   
   model = DREAM(input_dim=39, hidden_dim=256, rank=16, forgetting_rate=0.0)
   
   # Same audio, 5 passes
   state = model.init_state(batch_size=1)
   
   for pass_idx in range(5):
       output, state = model(audio_features, state=state)
       
       # Surprise should decrease each pass!
       avg_surprise = state.avg_surprise.item()
       print(f"Pass {pass_idx+1}: Surprise = {avg_surprise:.4f}")

6. Batch Processing with Padding
--------------------------------

Processing variable-length sequences:

.. code-block:: python

   from dream import DREAM
   import torch
   
   # Variable length data
   features = [
       torch.randn(150, 39),  # 150 steps
       torch.randn(200, 39),  # 200 steps
       torch.randn(180, 39),  # 180 steps
   ]
   
   # Padding
   max_len = max(f.shape[0] for f in features)
   batch_features = torch.zeros(3, max_len, 39)
   lengths = torch.tensor([f.shape[0] for f in features])
   
   for i, f in enumerate(features):
       batch_features[i, :f.shape[0], :] = f
   
   # Model with mask
   model = DREAM(input_dim=39, hidden_dim=256, rank=16)
   
   output, state = model.forward_with_mask(batch_features, lengths)
   
   # Output for padding = 0
   print(f"Output shape: {output.shape}")  # (3, max_len, 256)

7. Multi-Layer DREAM (DREAMStack)
---------------------------------

Deep architecture:

.. code-block:: python

   from dream import DREAMStack
   
   stack = DREAMStack(
       input_dim=39,
       hidden_dims=[128, 256, 128],  # 3 layers
       rank=16,
       dropout=0.1,
       ltc_enabled=True,
   )
   
   x = torch.randn(4, 100, 39)
   output, states = stack(x)
   
   # states[0], states[1], states[2] — states for each layer
   print(f"Output shape: {output.shape}")  # (4, 100, 128)

Tips and Best Practices
-----------------------

**State Initialization:**

.. code-block:: python

   # Always initialize state before use
   state = model.init_state(batch_size)
   
   # For memory preservation between epochs — don't reset state!
   for epoch in range(n_epochs):
       output, state = model(x, state=state)  # state preserved

**Hyperparameter Selection:**

.. code-block:: python

   # For fast adaptation
   config = DREAMConfig(
       base_plasticity=3.0,    # High plasticity
       ltc_tau_sys=5.0,        # Fast response
       forgetting_rate=0.0,    # No forgetting
   )
   
   # For stability
   config = DREAMConfig(
       base_plasticity=0.5,    # Low plasticity
       ltc_tau_sys=20.0,       # Slow integration
       forgetting_rate=0.05,   # Forgetting for homeostasis
   )

**Gradient Clipping:**

.. code-block:: python

   # Always use gradient clipping for RNNs
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

Next Steps
----------

* :doc:`architecture` — Architecture details
* :doc:`api` — Full API documentation
* :doc:`installation` — Installation guide
