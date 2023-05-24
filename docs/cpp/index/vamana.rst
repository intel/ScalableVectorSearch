.. _cpp_orchestrators_vamana:

Vamana Index
============

Documentation for the type-erased version of the :cpp:class:`svs::index::vamana::VamanaIndex`.

.. doxygenclass:: svs::Vamana
   :project: SVS
   :members:

Helper Classes
--------------

.. doxygenstruct:: svs::index::vamana::VamanaBuildParameters
   :project: SVS
   :members:

Example
-------

This example will cover the following topic:

* Building a :cpp:class:`svs::Vamana` index orchestrator.
* Performing queries to retrieve neighbors from the :cpp:class:`svs::Vamana` index.
* Saving the index to disk.
* Loading a :cpp:class:`svs::Vamana` index orchestrator from disk.
* Compressing an on-disk dataset and loading/searching with a compressed
  :cpp:class:`svs::Vamana`.

The complete example is included at the end of this file.

Preamble
^^^^^^^^

First, We need to include some headers.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Includes]
   :end-before: [Includes]

Then, we need to include some helpful utilities.
@snippet vamana.cpp Helper Utilities

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Helper Utilities]
   :end-before: [Helper Utilities]

The function ``run_recall`` sets the search window size of the :cpp:class:`svs::Vamana`
index performs a search over a given set of queries, and computes the ``k`` recall at ``k``
where ``k`` is the number of returned neighbors.

The function ``check`` compares the recall with an expected recall and throws an
exception if they differ by too much (this is mainly to allow automated testing of this
example).

Finally, we do some argument decoding.
@snippet vamana.cpp Argument Extraction

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Argument Extraction]
   :end-before: [Argument Extraction]
   :dedent: 4

It is expected that the pysvs function :py:func:`pysvs.generate_test_dataset` was used
to generate the data, graph, and metdata files.

Building and Index
^^^^^^^^^^^^^^^^^^

The first step is to construct an instance of :cpp:class:`svs::index::vamana::VamanaBuildParameters` to describe the hyper-parameters of the graph we wish to construct.
Don't worry too much about selecting the correct values for these hyper-parameters right now.
This usually involves a bit of experimentation and is dataset dependent.

Now that we've established our parameters, it is time to construct the index.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Index Build]
   :end-before: [Index Build]
   :dedent: 4

There are several things to note about about this construction.

* The type parameter ``float`` passed to :cpp:func:`svs::Vamana::build` indicates the element types
  of the queries that will be supported by the resulting index.

  Using queries with a different element type will result in a run-time error.
  This is due to limits in type-erasure and dynamic function calls.
* The data path is wrapped in a :cpp:class:`svs::VectorDataLoader` with the correct element type.
  In this case, we're explicitly using the dynamic sizing for the data. If static
  dimensionality was desired, than the second value parameter for :cpp:class:`svs::VectorDataLoader`
  could be used.
* An instance of the :cpp:class:`svs::distance::DistanceL2` functor is passed directly.

Searching the Index
^^^^^^^^^^^^^^^^^^^

The graph is now built and we can perform queries over the graph.
First, we load the queries and the computed ground truth for our example dataset using :cpp:func:`svs::io::auto_load`.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Load Aux]
   :end-before: [Load Aux]
   :dedent: 4

Performing queries is easy.
First establish a base-line search window size (:cpp:func:`svs::Vamana::set_search_window_size`).
This provides a parameter by which performance and accuracy can be traded.
The larger ``search_window_size`` is, the higher the accuracy but the lower the performance.
Note that ``search_window_size`` must be at least as large as the desired number of neighbors.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Perform Queries]
   :end-before: [Perform Queries]
   :dedent: 4

We use :cpp:func:`svs::Vamana::search` to find the 10 approximate nearest neighbors to each query in the form of a :cpp:type:`svs::QueryResult`.
Then, we use :cpp:func:`svs::k_recall_at_n` to compute the 10-recall at 10 of the returned neighbors, checking to confirm the accuracy.

The code snippet below demonstrates how to vary the search window size to change the achieved recall.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Search Window Size]
   :end-before: [Search Window Size]
   :dedent: 4

Saving the Index
^^^^^^^^^^^^^^^^

If you are satisfied with the performance of the generated index, you can save it to disk to avoid rebuilding it in the future.
This is done with :cpp:func:`svs::Vamana::save`.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: python
   :start-after: [Saving]
   :end-before: [Saving]
   :dedent: 4

Reloading a Saved Index
^^^^^^^^^^^^^^^^^^^^^^^

To reload the index from file, use :cpp:func:`svs::Vamana::load`.
This informs the dispatch mechanisms that we're loading an uncompressed data file from disk.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Loading]
   :end-before: [Loading]
   :dedent: 4

Performing queries is identical to before.

Using Vector Compression
^^^^^^^^^^^^^^^^^^^^^^^^

The library has experimental support for performing online vector compression.
The second argument to :cpp:func:`svs::Vamana::load` can be one of the :ref:`compressed loaders <cpp_quantization_lvq>`, which will compress an uncompressed dataset on the fly.

Specifying the loader is all that is required to use vector compression.
Note that vector compression is usually accompanied by an accuracy loss for the same search window size and may require increasing the window size to compensate.
The example below shows a :cpp:class:`svs::quantization::lvq::OneLevelWithBias` quantization scheme with a static dimensionality of 128.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Compressed Loader]
   :end-before: [Compressed Loader]

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Search Compressed]
   :end-before: [Search Compressed]

Building using Vector Compression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Index building using LVQ is very similar to index building using standard uncompressed vectors, though it may not be supported by all compression techniques.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Build Index Compressed]
   :end-before: [Build Index Compressed]

Entire Example
^^^^^^^^^^^^^^

This ends the example demonstrating the features of the :cpp:class:`svs::Vamana` index.
The entire executable code is shown below.
Please reach out with any questions.

.. literalinclude:: ../../../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Example All]
   :end-before: [Example All]

