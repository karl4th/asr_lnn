Architecture
============

This section describes the internal architecture of DREAM.

Architecture Overview
---------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    DREAM Cell                               │
   ├─────────────────────────────────────────────────────────────┤
   │  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
   │  │  Predictive  │     │   Surprise   │     │    Fast     │ │
   │  │   Coding     │────▶│    Gate      │────▶│   Weights   │ │
   │  │  (C, W, B)   │     │  (τ + habit) │     │   (U, V)    │ │
   │  └──────────────┘     └──────────────┘     └─────────────┘ │
   │         │                    │                    │         │
   │         ▼                    ▼                    ▼         │
   │  ┌─────────────────────────────────────────────────────┐   │
   │  │          Liquid Time-Constant (LTC)                 │   │
   │  │   τ_eff = τ_sys / (1 + surprise × scale)            │   │
   │  │   h_new = (1-α)·h_prev + α·tanh(input_effect)       │   │
   │  └─────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────┘

DREAMCell Components
--------------------

Predictive Coding
~~~~~~~~~~~~~~~~~

Matrices for prediction and error processing:

.. math::

   x_{pred} = \tanh((C + V U^T) h) \cdot \|x\|
   
   error = x - x_{pred}

Where:
- ``C`` — main prediction matrix (input_dim × hidden_dim)
- ``W`` — error projection (hidden_dim × input_dim)
- ``B`` — input projection (hidden_dim × input_dim)

Surprise Gate
~~~~~~~~~~~~~

Computing surprise with habituation:

.. math::

   \text{entropy} = 0.5 \log(2\pi e \cdot \text{error\_var})
   
   \tau_{adaptive} = (1 - r) \cdot \tau_{adaptive} + r \cdot \|error\|
   
   \tau_{effective} = 0.3 \cdot \tau_0 (1 + \alpha \cdot \text{entropy}) + 0.7 \cdot \tau_{adaptive}
   
   \text{surprise} = \sigma\left(\frac{\|error\| - \tau_{effective}}{\gamma}\right)

Parameters:
- ``tau_0`` — base surprise threshold
- ``alpha`` — entropy influence
- ``gamma`` — surprise temperature
- ``habituation_rate`` — habituation speed

Fast Weights (U, V)
~~~~~~~~~~~~~~~~~~~

Low-rank decomposition for efficient learning:

.. math::

   \text{hebbian} = (h_{prev} \otimes error) V
   
   dU = -\lambda (U - U_{target}) + \eta \cdot \text{surprise} \cdot \text{hebbian}
   
   U_{new} = U + dU \cdot dt

Where:
- ``V`` — fixed matrix (input_dim × rank), SVD initialized
- ``U`` — learnable matrix (batch × hidden_dim × rank)
- ``lambda`` — forgetting rate
- ``eta`` — plasticity coefficient

Liquid Time-Constant (LTC)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptive integration with surprise modulation:

.. math::

   \tau_{dynamic} = \frac{\tau_{sys}}{1 + \text{surprise} \cdot \text{scale}}
   
   \tau_{effective} = \text{clamp}(\tau_{dynamic}, \tau_{min}, \tau_{max})
   
   \alpha = \frac{dt}{\tau_{effective} + dt}
   
   h_{new} = (1 - \alpha) h_{prev} + \alpha \tanh(\text{input\_effect})

Parameters:
- ``tau_sys`` — base time constant
- ``ltc_surprise_scale`` — modulation strength from surprise
- ``min_tau``, ``max_tau`` — stability clamps

Sleep Consolidation
~~~~~~~~~~~~~~~~~~~

Memory consolidation during "sleep":

.. math::

   \text{if } \text{avg\_surprise} > S_{min}:
   
   dU_{target} = \zeta_{sleep} \cdot \text{avg\_surprise} \cdot (U - U_{target})
   
   U_{target} = U_{target} + dU_{target}

Where:
- ``sleep_rate`` — consolidation rate
- ``min_surprise_for_sleep`` — activation threshold

State Management
----------------

DREAMState contains:

.. code-block:: python

   @dataclass
   class DREAMState:
       h: torch.Tensor              # Hidden state (batch, hidden_dim)
       U: torch.Tensor              # Fast weights (batch, hidden_dim, rank)
       U_target: torch.Tensor       # Target fast weights (batch, hidden_dim, rank)
       adaptive_tau: torch.Tensor   # Adaptive threshold (batch,)
       error_mean: torch.Tensor     # Error mean (batch, input_dim)
       error_var: torch.Tensor      # Error variance (batch, input_dim)
       avg_surprise: torch.Tensor   # Average surprise (batch,)

Per-Batch U Matrices
~~~~~~~~~~~~~~~~~~~~

Each batch element has **independent memory**:

.. code-block:: python

   # U shape: (batch, hidden_dim, rank)
   # Each example learns independently!

This enables:
- Learning different patterns simultaneously
- No memory mixing between examples
- State preservation between epochs for the same example

Truncated BPTT
~~~~~~~~~~~~~~

For long sequences:

.. code-block:: python

   for start in range(0, seq_len, segment_size):
       segment = features[:, start:start+segment_size, :]
       output, state = model(segment, state=state)
       state = state.detach()  # Graph reset between segments

This saves 5-10x memory.

References
----------

* `Liquid Time-Constant Networks <https://arxiv.org/abs/2006.04439>`_
* `Fast Weights Paper <https://arxiv.org/abs/1610.06251>`_
* `Technical Report <https://github.com/karl4th/dream-nn/blob/main/tech_report.md>`_
