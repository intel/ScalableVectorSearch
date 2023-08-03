.. _vamana.h: ../../bindings/python/src/vamana.h
.. _flat.cpp: ../../bindings/python/src/flat.cpp

.. _features:

Library Features
################
Here we present the main library features, including the supported index types, distance functions and data types.

.. contents::
   :local:
   :depth: 1

.. _index_constructors:

Indexes
==================
SVS supports two index types: the Vamana graph (:ref:`in Python <vamana_api>`, :ref:`in C++ <vamana>`), to run fast
:ref:`graph-based <graph-search>` similarity search with high accuracy, and the flat index
(:ref:`in Python <flat_api>`, :ref:`in C++ <flat>`), to run exhaustive search (e.g., useful to efficiently compute
the ground-truth nearest neighbors for a dataset).

.. _graph-search:

Graph-based similarity search
-------------------------------
Graph-based methods use proximity graphs, where nodes represent data vectors and two nodes are connected if they fulfill
a defined property or neighborhood criterion, building on the structure inherent in the data. Search involves starting
at a designated entry point and traversing the graph to get closer and closer to the nearest neighbor with each hop. We
follow the Vamana [SDSK19]_ algorithm for graph building and search.

.. _graph-search-details:

How does the graph search work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The simplest way to traverse the graph to find 1 approximate nearest neighbor is to do a **greedy search**. At each hop,
the distances from the query to all the neighbors of the current node (i.e., vectors in the current node's adjacency
list) are computed and the closest point is chosen as the next point to be explored. The search ends when the distance to
the query cannot be further reduced by jumping to any of the neighbors of the current node.

**How do we find k neighbors?** To improve the search accuracy and be able to find k nearest neighbors, this greedy search is combined with a **priority
queue**. While traversing the graph, we keep track of the distance from the query to the ``search_window_size``
closest points seen so far (where ``search_window_size`` is the length of the priority queue). At each hop, we choose to
explore next the closest point in the priority queue that has not been visited yet. The search ends when all the
neighbors of the current node are further from the query than the furthest point in the priority queue. This prevents
the search path to diverge too far from the query. A larger ``search_window_size`` implies exploring a larger volume,
improving the accuracy at the cost of a longer search path.

.. _graph-building-details:

How does graph building work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, we :ref:`set the hyper-parameters <graph-build-param-setting>` required to build the graph: ``alpha``,
``graph_max_degree``, ``window_size``, ``max_candidate_pool_size`` (see :py:class:`pysvs.VamanaBuildParameters` and
:cpp:class:`svs::index::vamana::VamanaBuildParameters` for more details).

.. _graph-building-pseudocode:

Then, the graph is built following the Vamana indexing algorithm [SDSK19]_ as follows:

#. Start from an uninitialized graph **G**.
#. Iterate through all nodes in a random order.

   a. Run the search for node **x** on the current **G**, with the search window size set to ``window_size``, and save the list of visited nodes **C**.
   b. Update **G** by :ref:`pruning <graph-pruning>` **C** to determine the new set of **x**'s neighbors.
   c. Add backward edges (**x**, **x***) for all **x*** in **x**'s out neighbors and prune **x***' edges.

#. Make two passes over the dataset, the first one with the pruning parameter `alpha` =1 and the second one with `alpha` = ``alpha``.
#. Return graph **G** to be used by the search algorithm.

The **pruning rule** limits **x**'s out-neighbors **N** to a maximum of ``graph_max_degree`` as follows:

.. _graph-pruning:

#. Set the list of neighbors candidates **C** = **C** U **N** \\ { **x** }
#. Sort **C** in ascending distance from **x**, and limit **C** to the closest ``max_candidate_pool_size`` neighbors.
#. Initialize **N** to null
#. While **C** is not empty do:

   a. Find **x*** the closest point to **x** in **C**.
   b. Add **x*** to **x**'s out-neighbors list **N**.
   c. If *length* ( **N** ) > ``graph_max_degree`` then break; else remove all points from **C** that are closer to **x*** than **x** by a factor `alpha`.

.. _supported_distance_functions:

Distance functions
===================
SVS supports the distance functions listed in :ref:`cpp_core_distance` (see :py:class:`pysvs.DistanceType` for the corresponding
Python classes). The distance function is specified when the index is created by the corresponding :ref:`index constructors <index_constructors>`. In the
case of the Vamana index, it must also be specified when the graph is built (see :py:class:`pysvs.Vamana.build` and
:cpp:func:`svs::Vamana::build` for details).

.. _supported_data_types:

Data types
==========
The supported data types are: *float32*, *float16*, *int8* and *uint8*. Other data types might work but performance has not been tested.

The data type can be set **independently** for the **database vectors** and the **query vector**. For example, one could compress
the database vectors to float16, which allows for a 2x storage reduction often with negligible accuracy loss, and keep
the query in float32.

**In Python**

The data type for the **database vectors** is specified by the ``data_type`` argument when the vectors are loaded with
:py:class:`pysvs.VectorDataLoader`. The data type for the
**query vectors** is specified in the ``query_type`` argument for the corresponding index constructors
(:py:class:`pysvs.Vamana`, :py:class:`pysvs.Flat`).

**In C++**

.. collapse:: Click to display

    The database vectors data type is specified in the template argument of the :cpp:class:`svs::VectorDataLoader`.

    .. code-block:: cpp

        svs::VectorDataLoader<float>("data_f32.svs")

    For details on setting the query vectors data type see :ref:`cpp_orchestrators_vamana` and :ref:`cpp_orchestrators_flat`.

|

.. warning::

    This will not perform any dataset conversion. If a dataset was saved to disk as float16 data, for example,
    then it must be loaded with ``data_type = pysvs.DataType.float16`` in Python or
    ``svs::Float16`` in C++.

The supported data type combinations for (*queries*, *database vectors*) are: (*float32*, *float32*), (*float32*, *float16*),
(*uint8*, *uint8*), (*int8*, *int8*), among others.

.. _vector_compression:

Vector compression
==================
The memory footprint can be reduced and the search performance improved by combining the graph-search with the
Locally-adaptive Vector Quantization (LVQ) [ABHT23]_ approach.
The library has experimental support for performing online vector compression with LVQ.
See :ref:`compression-setting` for more details about LVQ and how to set LVQ parameters.

.. _enabling_vector_compression:

Enabling Vector Compression Support
-----------------------------------

Vector compression support needs to be enabled at compile time for each vector dimensionality.
To add support to the pysvs module for LVQ compression for the :ref:`Vamana graph index <vamana_api>`:

1. Define the desired dimensionality specialization in the vamana.h_ file by adding the corresponding line to the ``compressed_specializations`` template
   indicating the desired distance type (Euclidean distance and inner product are currently supported), dimensionality
   and whether graph building with compressed vectors wants to be
   enabled for that setting.

   For example, to add LVQ support for the 96-dimensional dataset `Deep <http://sites.skoltech.ru/compvision/noimi/>`_,
   for Euclidean distance, with graph building enabled, add the following line:

.. code-block:: cpp

   X(DistanceL2, 96, true);

2. :ref:`Install pysvs <install_pysvs>`.

For the :ref:`Flat index <flat_api>` follow the same procedure with the flat.cpp_ file.

Search with compressed vectors
------------------------------
See :ref:`search_with_compression` for details on how to use LVQ for search.

.. _building_with_compressed_vectors:

Graph building with compressed vectors
--------------------------------------
LVQ-compressed vectors can be used to build the graph, thus reducing the memory footprint not only for search but also
for indexing. Depending on the dataset, the search accuracy can be almost unchanged even when the graph is built with
highly compressed vectors using LVQ-4 or LVQ-8, that is, using only 4 or 8 bits per vector component [ABHT23]_.

To build the graph using compressed vectors we just need to follow the same procedure we did for :ref:`graph building <graph-building-code>`.
First, define the building parameters,

**In Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [build-parameters]
   :end-before: [build-parameters]
   :dedent: 4

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Build Parameters]
       :end-before: [Build Parameters]
       :dedent: 4

|

then load the dataset, using the loader (:ref:`details for Python <python_api_loaders>`,
:ref:`details for C++ <cpp_quantization_lvq>`) corresponding to the chosen compression type,

**In Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [search-compressed-loader]
   :end-before: [search-compressed-loader]
   :dedent: 4

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Compressed Loader]
       :end-before: [Compressed Loader]
       :dedent: 4

|

and finally call the build function with the chosen compression loader

**In Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [build-index-compressed]
   :end-before: [build-index-compressed]
   :dedent: 4

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Build Index Compressed]
       :end-before: [Build Index Compressed]
       :dedent: 4

|

.. rubric:: References

.. [SDSK19] Subramanya, S.J.; Devvrit, F.; Simhadri, H.V.; Krishnawamy, R.; Kadekodi, R..:Diskann: Fast accurate billion-point nearest neighbor search on a single node. In: Advances in Neural Information Processing Systems 32 (2019).
.. [ABHT23] Aguerrebere, C.; Bhati I.; Hildebrand M.; Tepper M.; Willke T..:Similarity search in the blink of an eye with compressed indices. In: Proceedings of the VLDB Endowment, 16, 11, 3433 - 3446. (2023)
.. [MaYa18] Malkov, Y. A. and Yashunin, D. A..: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. In: IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.
.. [JoDJ19] Johnson, J.; Douze, M.; Jégou, H..: Billion-scale similarity search with GPUs. In: IEEE Transactions on Big Data 7, 3 (2019), 535–547.
.. [GSLG20] Guo, R.; Sun, P.; Lindgren, E.; Geng, Q.; Simcha, D.; Chern, F.; Kumar, S..: Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning. PMLR, 3887-3896 (2020)
.. [IwMi18] Iwasaki, M. and Miyazaki, D..: Nearest Neighbor Search with Neighborhood Graph and Tree for High-dimensional Data. https://github.com/yahoojapan/NGT (2018)
.. [AuBF20] Aumüller, M.; Bernhardsson, E.; Faithfull, A..: ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms. In: Information Systems 87 (2020), 101374. https://doi.org/10.1016/j.is.2019.02.006
.. [KOML20] Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; Yih, W..: Dense Passage Retrieval for Open-Domain Question Answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 6769–6781. (2020)
.. [RSRL20] Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; Liu, P.J.: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. In: The Journal of Machine Learning Research 21,140:1–140:67.(2020)