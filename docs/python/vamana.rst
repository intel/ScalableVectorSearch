.. _vamana_api:

Vamana Graph Index
==================

In this section, we cover the API and usage of the Vamana graph-based index.

.. autoclass:: pysvs.Vamana
   :members:

   .. automethod:: __init__

.. autoclass:: pysvs.VamanaBuildParameters
   :members:

   .. automethod:: __init__


Example
-------

This example will cover the following topics:

    * Building a :py:class:`pysvs.Vamana` index from a dataset.
    * Performing queries to retrieve neighbors from a :py:class:`pysvs.Vamana`.
    * Saving the index to disk.
    * Loading a :py:class:`pysvs.Vamana` from disk.
    * Compressing an on-disk dataset and loading/searching with a compressed
      :py:class:`pysvs.Vamana`.

The complete example is included at the end of this file.

Preamble
^^^^^^^^

First, it is assumed that ``pysvs`` is installed and available to Python.
We first need to perform a couple of imports.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [imports]
   :end-before: [imports]

Then, we generate a sample dataset using the :py:func:`pysvs.generate_test_dataset` generation function.
This function generates a data file, a query file, and the ground truth.

.. note::

    The :py:func:`pysvs.generate_test_dataset` function generates datasets randomly
    with no semantic meaning for the elements within it.

    Recall values for this dataset are usually lower than for real datasets.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [generate-dataset]
   :end-before: [generate-dataset]
   :dedent: 4

Building and Index
^^^^^^^^^^^^^^^^^^

Now that data has been generated, we need to construct an index over that data.
The index is a graph connecting related data vectors in such a way that searching for nearest neighbors yields good results.
The first step is to construct an instance of :py:class:`pysvs.VamanaBuildParameters` to describe the hyper-parameters of the graph we wish to construct.
Don't worry too much about selecting the correct values for these hyper-parameters right now.
This usually involves a bit of experimentation and is dataset dependent.

Now that we've established our parameters, it is time to construct the index.
Note the use of :py:class:`pysvs.VectorDataLoader` to indicate both the file path and the element type of the ``fvecs`` file on disk.
Passing the ``dims`` is optional, but may yield performance benefits if given.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [build-index]
   :end-before: [build-index]
   :dedent: 4


Searching the Index
^^^^^^^^^^^^^^^^^^^

The graph is now built and we can perform queries over the graph.
First, we load the queries and the computed ground truth for our example dataset using :py:func:`pysvs.read_vecs`.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [load-aux]
   :end-before: [load-aux]
   :dedent: 4

Performing queries is easy.
First establish a base-line search window size (:py:attr:`pysvs.Vamana.search_window_size`).
This provides a parameter by which performance and accuracy can be traded.
The larger ``search_window_size`` is, the higher the accuracy but the lower the performance.
Note that ``search_window_size`` must be at least as large as the desired number of neighbors.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [perform-queries]
   :end-before: [perform-queries]
   :dedent: 4

We use :py:func:`pysvs.Vamana.search` to find the 10 approximate nearest neighbors to each query.
Then, we use :py:func:`pysvs.k_recall_at` to compute the 10-recall at 10 of the returned neighbors, checking to confirm the accuracy.

The code snippet below demonstrates how to vary the search window size to change the achieved recall.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [perform-queries]
   :end-before: [perform-queries]
   :dedent: 4

Saving the Index
^^^^^^^^^^^^^^^^

If you are satisfied with the performance of the generated index, you can save it to disk to avoid rebuilding it in the future.
This is done with :py:func:`pysvs.Vamana.save`.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [saving-results]
   :end-before: [saving-results]
   :dedent: 4

.. note::

    The :py:class:`pysvs.Vamana` index currently uses three files for saving.
    All three are needed to be able to reload the index.

    * One file for the graph.
    * One file for the data (in a different form from ``fvecs``).
    * One small metadata file.

    This is subject to change in the future.

Reloading a Saved Index
^^^^^^^^^^^^^^^^^^^^^^^

To reload the index from file, use :py:func:`pysvs.Vamana.__init__` with a :py:class:`pysvs.VectorDataLoader` for the second argument.
This informs the dispatch mechanisms that we're loading an uncompressed data file from disk.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [loading]
   :end-before: [loading]
   :dedent: 4

Performing queries is identical to before.

Using Vector Compression
^^^^^^^^^^^^^^^^^^^^^^^^

The library has experimental support for performing online vector compression.
The second argument to :py:func:`pysvs.Vamana.__init__` can be one of the compressed loaders (:ref:`python_api_loaders`), which will compress an uncompressed dataset on the fly.
To see which loaders are applicable to which methods, study the signature in :py:func:`pysvs.vamana.__init__` carefully.

Specifying the loader is all that is required to use vector compression.
Note that vector compression is usually accompanied by an accuracy loss for the same search window size and may require increasing the window size to compensate.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python
   :start-after: [search-compressed]
   :end-before: [search-compressed]

Entire Example
^^^^^^^^^^^^^^

This ends the example demonstrating the features of the :py:class:`pysvs.Vamana` index.
The entire executable code is shown below.
Please reach out with any questions.

.. literalinclude:: ../../examples/python/example_vamana.py
   :language: python

