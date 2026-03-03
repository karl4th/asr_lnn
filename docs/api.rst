API Reference
=============

Complete DREAM API documentation.

.. module:: dream

Core Components
---------------

DREAMConfig
~~~~~~~~~~~

.. autoclass:: DREAMConfig
   :members:
   :member-order: bysource
   :show-inheritance:
   :undoc-members:

DREAMState
~~~~~~~~~~

.. autoclass:: DREAMState
   :members:
   :member-order: bysource
   :show-inheritance:
   :undoc-members:

DREAMCell
~~~~~~~~~

.. autoclass:: DREAMCell
   :members: __init__, forward, init_state, forward_sequence, surprise_gate, update_fast_weights, compute_ltc_update, sleep_consolidation
   :member-order: bysource
   :show-inheritance:
   :undoc-members:

High-Level API
--------------

DREAM
~~~~~

.. autoclass:: DREAM
   :members: __init__, forward, init_state, forward_sequence, forward_with_mask
   :member-order: bysource
   :show-inheritance:
   :undoc-members:

DREAMStack
~~~~~~~~~~

.. autoclass:: DREAMStack
   :members: __init__, forward, init_state
   :member-order: bysource
   :show-inheritance:
   :undoc-members:

Utilities
---------

RunningStatistics
~~~~~~~~~~~~~~~~~

.. autoclass:: RunningStatistics
   :members: __init__, update, reset, forward
   :member-order: bysource
   :show-inheritance:
   :undoc-members:

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
