.. Copyright (C) 2023 Intel Corporation
..
.. This software and the related documents are Intel copyrighted materials,
.. and your use of them is governed by the express license under which they
.. were provided to you ("License"). Unless the License provides otherwise,
.. you may not use, modify, copy, publish, distribute, disclose or transmit
.. this software or the related documents without Intel's prior written
.. permission.
..
.. This software and the related documents are provided as is, with no
.. express or implied warranties, other than those that are expressly stated
.. in the License.

.. _python_api_loaders:

Loaders
=======

Uncompressed File Loaders
-------------------------

.. autoclass:: svs.VectorDataLoader
   :members:

   .. automethod:: __init__

.. _python_api_lvq_loader:

LVQ Loader
----------

The LVQ loader provides lazy compression of uncompressed data and reloading of previously
saved LVQ data.

.. autoclass:: svs.LVQLoader
   :members:

   .. automethod:: __init__

Strategy Selection
******************

The strategy argument of the LVQ loader provides a way of overriding the default selection
of the packing strategy used by a LVQ backend.

Note that overriding the default strategy requires the corresponding backend to
be compiled in the `svs` shared library component.

.. autoclass:: svs.LVQStrategy

LeanVecLoader
-------------

The LeanVec loader provides a way to use dimensionality reduction to improve
performance on high dimensional datasets.

Internally, a LeanVec dataset consists of the dimensionality reduced primary dataset
(over which the bulk of the index search is conducted) and a full dimensional secondary
dataset used to rerank and refine candidates returned from the initial search.

`svs` allows selection of the storage format using the :py:class:`svs.LeanVecKind` enum,
enabling `float16` and `lvq` compression for either of the primary and secondary datasets.

.. autoclass:: svs.LeanVecLoader
   :members:

   .. automethod:: __init__

.. autoclass:: svs.LeanVecKind
