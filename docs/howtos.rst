.. _howtos:

How-Tos
##########

.. contents::
   :local:
   :depth: 1

.. _how_to_run_dynamic_indexing:

How to Do Dynamic Indexing
===========================
This tutorial will show you how to create a dynamic index, add and remove vectors, search the index, save and reload it.

Generating test data
********************

We generate a sample dataset using the :py:func:`pysvs.generate_test_dataset` generation function.
This function generates a data file, a query file, and the ground truth. Note that this is randomly generated data,
with no semantic meaning for the elements within it.

We first load pysvs and other modules required for this example.

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [imports]
   :end-before: [imports]

Then proceed to generate the test dataset.

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [generate-dataset]
   :end-before: [generate-dataset]
   :dedent: 4

Building the Dynamic Index
**************************
To construct the index we first need to define the hyper-parameters for the graph construction
(see :ref:`graph-build-param-setting` for details).

**In Python**

This is done by creating an instance of :py:class:`pysvs.VamanaBuildParameters`.

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [build-parameters]
   :end-before: [build-parameters]
   :dedent: 4

Now that we've established our hyper-parameters, it is time to construct the index. For this, we load the data and
build the dynamic index with the first 9k vectors of the dataset.

**In Python**

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [build-index]
   :end-before: [build-index]
   :dedent: 4

Updating the index
******************
Once we've built the initial dynamic index, we can add and remove vectors.

**In Python**

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [add-vectors]
   :end-before: [add-vectors]
   :dedent: 4

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [remove-vectors]
   :end-before: [remove-vectors]
   :dedent: 4

Deletions are performed in a lazy fashion to avoid an excessive compute overhead. When a vector is deleted, it is added
to a list of deleted elements but not immediately removed from the index. At search time, it is used during graph
traversal but it is filtered out from the nearest neighbors result.
Once a sufficient number of deletions is accumulated the ``consolidate()`` and ``compact()`` functions should be ran to
effectively remove the vectors from the index.

**In Python**

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [consolidate-index]
   :end-before: [consolidate-index]
   :dedent: 4

Searching the Index
********************

First, we load the queries and the computed ground truth for our example dataset.

**In Python**

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [load-aux]
   :end-before: [load-aux]
   :dedent: 4

Then, run the search in the same fashion as for the static graph.

**In Python**

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [perform-queries]
   :end-before: [perform-queries]
   :dedent: 4

Saving the Index
****************

If you are satisfied with the performance of the generated index, you can save it to disk to avoid rebuilding it in the future.

**In Python**

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [saving-results]
   :end-before: [saving-results]
   :dedent: 4

.. note::

    The save index function currently uses three folders for saving.
    All three are needed to be able to reload the index.

    * One folder for the graph.
    * One folder for the data.
    * One folder for metadata.

    This is subject to change in the future.

Reloading a Saved Index
***********************

To reload the index from file, use the corresponding constructor with the three folder names used to save the index.
Performing queries is identical to before.

**In Python**

.. literalinclude:: ../examples/python/example_vamana_dynamic.py
   :language: python
   :start-after: [loading]
   :end-before: [loading]
   :dedent: 4

Note that the second argument, the one corresponding to the file for the data, requires a :py:class:`pysvs.VectorDataLoader` and
the corresponding data type.

|

.. _graph-build-param-setting:

How to Choose Graph Building Hyper-parameters
=============================================
The optimal values for the graph building hyper-parameters depend on the dataset and on the trade-off between performance and
accuracy that is required. We suggest here commonly used values and provide some guidance on how to adjust them. See
:ref:`graph-building-details` for more details about graph building.

* ``graph_max_degree``: Maximum out-degree of the graph. A larger ``graph_max_degree`` implies more distance computations
  per hop, but potentially a shorter graph traversal path, so it can lead to higher search performance. High-dimensional
  datasets or datasets with a large number of points usually require a larger ``graph_max_degree`` to reach very high search accuracy.
  Keep in mind that the graph size in bytes is given by 4 times ``graph_max_degree``
  (each neighbor id in the graph adjacency lists is represented with 4 bytes) times the number of points in the dataset,
  so a larger ``graph_max_degree`` will have a larger memory footprint. Commonly used values for ``graph_max_degree`` are 32, 64 or 128.

* ``alpha``: Threshold for the graph adjacency lists :ref:`pruning rule <graph-pruning>` during the second pass over the dataset. For
  distance types favoring minimization, set this to a number greater than 1.0 to build a denser graph (typically, 1.2 is sufficient).
  For distance types preferring maximization, set to a value less than 1.0 to build a denser graph (such as 0.95).

* ``window_size``: Sets the ``search_window_size`` for the graph search conducted to add new points to the graph. This
  parameter controls the quality of :ref:`graph construction <graph-building-pseudocode>`. A larger window size will yield a higher-quality
  index at the cost of longer construction time. Should be larger than ``graph_max_degree``.

* ``max_candidate_pool_size``: Limit on the number of candidates to consider for the graph adjacency lists :ref:`pruning rule <graph-pruning>`.
  Should be larger than ``window_size``.

* ``num_threads``: The number of threads to use for index construction. The indexing process is highly parallelizable, so
  using as many ``num_threads`` as possible is usually better.

.. _search-window-size-setting:

How to Set the Search Window Size
==================================
The ``search_window_size`` is the knob controlling the trade-off between performance and accuracy for the graph search.
A larger ``search_window_size`` implies exploring more vectors, improving the accuracy at the cost of a longer search path.
See :ref:`graph-search-details` for more details about graph search. One simple way to set the ``search_window_size`` is to run
searches with multiple values of the parameter and print the recall to identify the required ``search_window_size`` for
the chosen accuracy level.

**In Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [search-window-size]
   :end-before: [search-window-size]
   :dedent: 4

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Search Window Size]
       :end-before: [Search Window Size]
       :dedent: 4

|

.. _compression-setting:

How to Choose Compression Parameters
=====================================

Number of bits per level
************************
LVQ compression [ABHT23]_ comes in two flavors: one or two levels. One level LVQ, or LVQ-B, uses B bits to encode each vector
component using a scalar quantization with per-vector scaling factors. Two level LVQ, or LVQ-B1xB2, uses LVQ-B1 to encode
the vectors and a modification of LVQ to encode the residuals using B2 bits.

Whether using one or two levels, and the number of bits, depends on the dataset and the trade-off between performance and
accuracy that needs to be achieved.

When using **two-level LVQ**, the graph search is conducted using vectors compressed with LVQ-B1 and a final re-ranking
step is performed using the residuals compressed with B2 bits to improve the search recall.
This decoupled strategy is particularly beneficial for **high dimensional datasets** (>200 dimensions) as LVQ achieves up to
~8x bandwidth reduction (B1=4) compared to a float32-valued vector. The number of bits for the residuals (4 or 8) should
be chosen depending on the desired search accuracy. Suggested configurations for high dimensional vectors are LVQ-4x8 or
LVQ-4x4 depending on the desired accuracy.

For **lower dimensional datasets** (<200 dimensions), **one-level** LVQ-8 is often a good choice. If higher recall is required, and a
slightly larger memory footprint is allowed, then LVQ-8x4 or LVQ-8x8 should be used.

These are general guidelines, but the best option will depend on the dataset. If willing to optimize the search for a
particular dataset and use case, we suggest trying different LVQ options. See :ref:`SVS + Vector compression (large scale
datasets) <benchs-compression-evaluation>` and
:ref:`SVS + Vector compression (small scale datasets) <benchs-compression-evaluation_small_scale>` for benchmarking results of
the different LVQ settings in standard datasets.

.. _lvq_strategy:

LVQ implementation strategy
***************************
The ``strategy`` argument in the :py:class:`pysvs.LVQLoader` is of type :py:class:`pysvs.LVQStrategy`
and defines the low level implementation strategy for LVQ, whether it is Turbo or Sequential. Turbo is an
optimized implementation that brings further performance over the default (Sequential) implementation [AHBW24]_. Turbo can be used
when using 4 bits for the primary LVQ level and it is enabled by default for that setting.

Padding
*******
LVQ-compressed vectors can be padded to a multiple of 32 or 64 bytes to be aligned with half or full cache lines.
This improves search performance and has a low impact on the overall memory footprint cost (e.g., 5% and 12% larger
footprint for `Deep <http://sites.skoltech.ru/compvision/noimi/>`_ with ``graph_max_degree`` = 128 and 32, respectively).
A value of 0 (default) implies no special alignment.

For details on the C++ implementation see :ref:`cpp_quantization_lvq`.