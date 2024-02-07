.. _python_api_loaders:

Loaders
=======

Uncompressed File Loaders
-------------------------

.. autoclass:: pysvs.VectorDataLoader
   :members:

   .. automethod:: __init__

.. _python_api_lvq_loader:

LVQ Loader
----------

The LVQ loader provides lazy compression of uncompressed data and reloading of previously
saved LVQ data.

.. autoclass:: pysvs.LVQLoader
   :members:

   .. automethod:: __init__

Strategy Selection
******************

The strategy argument of the LVQ loader provides a way of overriding the default selection
of the packing strategy used by a LVQ backend.

Note that overriding the default strategy requires the corresponding backend to
be compiled in the `pysvs` shared library component.

.. autoclass:: pysvs.LVQStrategy

LeanVecLoader
-------------

The LeanVec loader provides a way to use dimensionality reduction to improve
performance on high dimensional datasets.

Internally, a LeanVec dataset consists of the dimensionality reduced primary dataset
(over which the bulk of the index search is conducted) and a full dimensional secondary
dataset used to rerank and refine candidates returned from the initial search.

`pysvs` allows selection of the storage format using the :py:class:`pysvs.LeanVecKind` enum,
enabling `float16` and `lvq` compression for either of the primary and secondary datasets.

.. autoclass:: pysvs.LeanVecLoader
   :members:

   .. automethod:: __init__

.. autoclass:: pysvs.LeanVecKind
