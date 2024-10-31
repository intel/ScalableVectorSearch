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
The Vamana graph (:ref:`in Python <vamana_api>`, :ref:`in C++ <cpp_orchestrators_vamana>`) enables fast in-memory
:ref:`graph-based <graph-search>` similarity search with high accuracy for static databases, where the database
is fixed and never updated.

.. _dynamic_graph_index:

Graphs for streaming data
-------------------------
The DynamicVamana graph (:ref:`in Python <dynamic_vamana_api>`, :ref:`in C++ <cpp_orchestrators_dynamic_vamana>`) enables fast in-memory
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
SVS supports the distance functions listed in :ref:`cpp_core_distance` (see :py:class:`svs.DistanceType` for the corresponding
Python classes). The distance function is specified when the index is created by the corresponding :ref:`index constructors <index_constructors>`. In the
case of the Vamana index, it must also be specified when the graph is built (see :py:class:`svs.Vamana.build` and
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
:py:class:`svs.VectorDataLoader`. The data type for the
**query vectors** is specified in the ``query_type`` argument for the corresponding index constructors
(:py:class:`svs.Vamana`, :py:class:`svs.Flat`).

**In C++**

.. collapse:: Click to display

    The database vectors data type is specified in the template argument of the :cpp:class:`svs::VectorDataLoader`.

    .. code-block:: cpp

        svs::VectorDataLoader<float>("data_f32.svs")

    For details on setting the query vectors data type see :ref:`cpp_orchestrators`.

|

.. warning::

    This will not perform any dataset conversion. If a dataset was saved to disk as float16 data, for example,
    then it must be loaded with ``data_type = svs.DataType.float16`` in Python or
    ``svs::Float16`` in C++.

The supported data type combinations for (*queries*, *database vectors*) are: (*float32*, *float32*), (*float32*, *float16*),
(*uint8*, *uint8*), (*int8*, *int8*), among others.

.. rubric:: References

.. [SDSK19] Subramanya, S.J.; Devvrit, F.; Simhadri, H.V.; Krishnawamy, R.; Kadekodi, R..: Diskann: Fast accurate billion-point nearest neighbor search on a single node. In: Advances in Neural Information Processing Systems 32 (2019).
.. [ABHT23] Aguerrebere, C.; Bhati I.; Hildebrand M.; Tepper M.; Willke T..: Similarity search in the blink of an eye with compressed indices. In: Proceedings of the VLDB Endowment, 16, 11, 3433 - 3446. (2023)
.. [AHBW24] Aguerrebere, C.; Hildebrand M.; Bhati I.; Willke T.; Tepper M..: Locally-adaptive Quantization for Streaming Vector Search. In: arxiv preprint arXiv:2402.02044 (2024)
.. [MaYa18] Malkov, Y. A. and Yashunin, D. A..: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. In: IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.
.. [JoDJ19] Johnson, J.; Douze, M.; Jégou, H..: Billion-scale similarity search with GPUs. In: IEEE Transactions on Big Data 7, 3 (2019), 535–547.
.. [GSLG20] Guo, R.; Sun, P.; Lindgren, E.; Geng, Q.; Simcha, D.; Chern, F.; Kumar, S..: Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning. PMLR, 3887-3896 (2020)
.. [IwMi18] Iwasaki, M. and Miyazaki, D..: Nearest Neighbor Search with Neighborhood Graph and Tree for High-dimensional Data. https://github.com/yahoojapan/NGT (2018)
.. [AuBF20] Aumüller, M.; Bernhardsson, E.; Faithfull, A..: ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms. In: Information Systems 87 (2020), 101374. https://doi.org/10.1016/j.is.2019.02.006
.. [QDLL21] Qu, Y.; Ding, Y.; Liu, J.; Liu, K.; Ren, R.; Zhao, W. X.; Dong, D.; Wu, H. and Wang, H..: RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. In: Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 5835–5847. (2021)
.. [SSKS21] Singh, A.; Subramanya, S.J.; Krishnaswamy, R.; Simhadri, H.V..: FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search. In: arxiv preprint arXiv:2105.09613 (2021)
.. [DGDJ24] Douze, M.; Guzhva, A.; Deng, C.; Johnson, J.; Szilvasy, G.; Mazaré, P.E.; Lomeli, M.; Hosseini, L.; Jégou, H.: The Faiss library. In: arxiv preprint arXiv:2401.08281 (2024)
.. [TBAH24] Tepper M.; Bhati I.; Aguerrebere, C.; Hildebrand M.; Willke T.: LeanVec: Search your vectors faster by making them fit. arXiv preprint arXiv:2312.16335 (2024)
