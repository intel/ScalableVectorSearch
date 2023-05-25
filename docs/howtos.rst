.. _howtos:

How-Tos
##########

.. contents::
   :local:
   :depth: 1

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
LVQ compression [ABHT23]_ comes in two flavors: one or two levels. One level LVQ, or LVQ-B, uses B bits to encode each vector
component using a scalar quantization with per-vector scaling factors. Two level LVQ, or LVQ-B1xB2, uses LVQ-B1 to encode
the vectors and a modification of LVQ to encode the residuals using B2 bits. The currently supported combinations are:

* :py:class:`LVQ-8 <pysvs.LVQ8>`
* :py:class:`LVQ-4 <pysvs.LVQ4>`
* :py:class:`LVQ-4x4 <pysvs.LVQ4x4>`
* :py:class:`LVQ-4x8 <pysvs.LVQ4x8>`
* :py:class:`LVQ-8x8 <pysvs.LVQ8x8>`

For an updated list see :ref:`python_api_compressed_loaders`. For details on which LVQ configurations are supported for each
index type, study their respective signatures (:py:class:`pysvs.Vamana`, :py:class:`pysvs.Flat`).

For details on the C++ implementation see :ref:`cpp_quantization_lvq`.

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

LVQ-compressed vectors can be padded to a multiple of 32 or 64 bytes to be aligned with half or full cache lines.
This improves search performance and has a low impact on the overall memory footprint cost (e.g., 5% and 12% larger
footprint for `Deep <http://sites.skoltech.ru/compvision/noimi/>`_ with ``graph_max_degree`` = 128 and 32, respectively).
A value of 0 (default) implies no special alignment.

These are general guidelines, but the best option will depend on the dataset. If willing to optimize the search for a
particular dataset and use case, we suggest trying different LVQ options. See :ref:`SVS + Vector compression (large scale
datasets) <benchs-compression-evaluation>` and
:ref:`SVS + Vector compression (small scale datasets) <benchs-compression-evaluation_small_scale>` for benchmarking results of
the different LVQ settings in standard datasets.
