Architecture
============

DREAM Cell Overview
-------------------

.. code-block:: text

   Input → [Predictive Coding] → [Surprise Gate] → [Fast Weights] → [LTC] → Output

Components
----------

Predictive Coding
~~~~~~~~~~~~~~~~~

Predicts next input and computes error:

.. code-block:: python

   x_pred = tanh((C + V @ U.T) @ h) * ||x||
   error = x - x_pred

Surprise Gate
~~~~~~~~~~~~~

Computes surprise from prediction error:

.. code-block:: python

   surprise = sigmoid((||error|| - tau) / gamma)

Fast Weights
~~~~~~~~~~~~

Hebbian learning with surprise modulation:

.. code-block:: python

   dU = -lambda * (U - U_target) + eta * surprise * hebbian

Liquid Time-Constant
~~~~~~~~~~~~~~~~~~~~

Adaptive integration speed:

.. code-block:: python

   tau = tau_sys / (1 + surprise * scale)
   h_new = (1 - alpha) * h_prev + alpha * tanh(input)

State Structure
---------------

.. code-block:: python

   DREAMState:
       h              # Hidden state
       U              # Fast weights
       U_target       # Target for sleep
       adaptive_tau   # Surprise threshold
       error_mean     # Error statistics
       error_var      
       avg_surprise   
