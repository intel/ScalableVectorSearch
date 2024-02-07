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

Index Types
===========
SVS supports the following index types:

* :ref:`Graphs for static datasets <static_graph_index>`
* :ref:`Graphs for streaming data <dynamic_graph_index>`
* :ref:`Flat index <flat_index>`

.. _static_graph_index:

Graphs for static datasets
---------------------------
The Vamana graph (:ref:`in Python <vamana_api>`, :ref:`in C++ <vamana>`) enables fast in-memory
:ref:`graph-based <graph-search>` similarity search with high accuracy for static databases, where the database
is fixed and never updated.

.. _dynamic_graph_index:

Graphs for streaming data
-------------------------
The DynamicVamana graph (:ref:`in Python <dynamic_vamana_api>`) enables fast in-memory
:ref:`graph-based <graph-search>` similarity search with high accuracy for streaming data, where the database is built
dynamically by adding and removing vectors.

.. _flat_index:

Flat Index
----------
The flat index (:ref:`in Python <flat_api>`, :ref:`in C++ <flat>`) can be used to run exhaustive search, e.g., useful to compute
the ground-truth nearest neighbors for a dataset.

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
The library has support for :ref:`performing online vector compression with LVQ <search_with_compression>` or
for :ref:`loading an index with a previously compressed dataset <loading_compressed_indices>`.
See :ref:`compression-setting` for more details about LVQ and how to set its parameters.

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
.. [AHBW24] Aguerrebere, C.; Hildebrand M.; Bhati I.; Willke T.; Tepper M..:Locally-adaptive Quantization for Streaming Vector Search. In: arxiv.
.. [MaYa18] Malkov, Y. A. and Yashunin, D. A..: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. In: IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.
.. [JoDJ19] Johnson, J.; Douze, M.; Jégou, H..: Billion-scale similarity search with GPUs. In: IEEE Transactions on Big Data 7, 3 (2019), 535–547.
.. [GSLG20] Guo, R.; Sun, P.; Lindgren, E.; Geng, Q.; Simcha, D.; Chern, F.; Kumar, S..: Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning. PMLR, 3887-3896 (2020)
.. [IwMi18] Iwasaki, M. and Miyazaki, D..: Nearest Neighbor Search with Neighborhood Graph and Tree for High-dimensional Data. https://github.com/yahoojapan/NGT (2018)
.. [AuBF20] Aumüller, M.; Bernhardsson, E.; Faithfull, A..: ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms. In: Information Systems 87 (2020), 101374. https://doi.org/10.1016/j.is.2019.02.006
.. [KOML20] Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; Yih, W..: Dense Passage Retrieval for Open-Domain Question Answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 6769–6781. (2020)
.. [RSRL20] Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; Liu, P.J.: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. In: The Journal of Machine Learning Research 21,140:1–140:67.(2020)