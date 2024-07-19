.. _start:

Getting started with Python
###########################
This tutorial will show you how to install SVS and run your first search with it! Tutorials for running
:ref:`dynamic indexing <how_to_run_dynamic_indexing>`, setting :ref:`index <graph-build-param-setting>` and
:ref:`search<search-window-size-setting>` parameters, :ref:`using vector compression <compression-setting>`, as well as
more :ref:`advanced installation options <build>` are also available.

.. contents::
   :local:
   :depth: 2

.. _install_svs:

Installation
************

Building and installing SVS should be relatively straight-forward. We test on Ubuntu 22.04 LTS, but any Linux distribution should work.

Prerequisites
=============

* Python >= 3.9

.. include:: prerequisites.rst


Building and installing
=======================

To build and install the SVS Python module, ``svs``, clone the repo and run the following pip install command.

.. code-block:: sh

    # Clone the repository
    git clone https://github.com/IntelLabs/ScalableVectorSearch
    cd ScalableVectorSearch

    # Install svs using pip
    pip install bindings/python

If you encounter any issues with the pip install command, please follow the :ref:`advanced installation instructions <building_python_library>`.


Verifying the installation
==========================

Run the following command to verify that SVS was successfully installed. It should print ``['native']``.

.. code-block:: sh

    python3 -c "import svs; print(svs.available_backends())"


SVS search example
******************

In this tutorial we will showcase the most important features of SVS. The :ref:`full example <entire_example>` is available at the end of this tutorial.
You can run it with the following commands:

.. code-block:: sh

    cd examples/python
    python3 example_vamana.py


Generating test data
********************

We generate a sample dataset using the :py:func:`svs.generate_test_dataset` generation function.
This function generates a data file, a query file, and the ground truth. Note that this is randomly generated data,
with no semantic meaning for the elements within it.

We first load svs and the os module also required for this example.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [imports]
   :end-before: [imports]

Then proceed to generate the test dataset.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [generate-dataset]
   :end-before: [generate-dataset]
   :dedent: 4

.. _graph-building-code:

Building the index
******************

Now that data has been generated, we need to construct an index over that data.
The index is a graph connecting related data vectors in such a way that searching for nearest neighbors yields good results.
The first step is to define the hyper-parameters of the graph we wish to construct.
Don't worry too much about selecting the correct values for these hyper-parameters right now.
This usually involves a bit of experimentation and is dataset dependent. See :ref:`graph-build-param-setting` for details.

This is done by creating an instance of :py:class:`svs.VamanaBuildParameters`.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [build-parameters]
   :end-before: [build-parameters]
   :dedent: 4

Now that we've established our hyper-parameters, it is time to construct the index.
Passing the ``dims`` is optional, but may :ref:`yield performance benefits if given <static-dim>`.

We can build the index directly from the dataset file on disk

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [build-index]
   :end-before: [build-index]
   :dedent: 4

or from a Numpy array

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [build-index-fromNumpyArray]
   :end-before: [build-index-fromNumpyArray]
   :dedent: 4

Note the use of :py:class:`svs.VectorDataLoader` to indicate both the file path and the :ref:`data type <supported_data_types>`
of the ``fvecs`` file on disk (see :ref:`io` for supported file formats). See :py:class:`svs.Vamana.build`
for details about the build function.

.. note::

   :py:class:`svs.Vamana.build` supports building from Numpy arrays with dtypes float32, float16, int8 and uint8.


Searching the index
********************

The graph is now built and we can perform queries over the graph.
First, we load the queries for our example dataset.
After searching, we compare the search results with ground truth results which
we also load from the dataset.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [load-aux]
   :end-before: [load-aux]
   :dedent: 4

Performing queries is easy.
First establish a base-line search window size. This provides a parameter by which performance and accuracy can be traded.
The larger ``search_window_size`` is, the higher the accuracy but the lower the performance.
Note that ``search_window_size`` must be at least as large as the desired number of neighbors.
See :ref:`search-window-size-setting` for details.

We use the search function to find the 10 approximate nearest neighbors to each query.
Then, we compute the 10-recall at 10 of the returned neighbors, checking to confirm
the accuracy.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [perform-queries]
   :end-before: [perform-queries]
   :dedent: 4

See :py:class:`svs.Vamana.search` for details about the search function.


.. _index_saving:

Saving the index
****************

If you are satisfied with the performance of the generated index, you can save it to disk to avoid rebuilding it in the future.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [saving-results]
   :end-before: [saving-results]
   :dedent: 4

See :py:func:`svs.Vamana.save` for details about the save function.

.. note::

    The save index function currently uses three folders for saving.
    All three are needed to be able to reload the index.

    * One folder for the graph.
    * One folder for the data.
    * One folder for metadata.

    This is subject to change in the future.


.. _index_loading:

Reloading a saved index
***********************

To reload the index from file, use the corresponding constructor with the three folder names used to save the index.
Performing queries is identical to before.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [loading]
   :end-before: [loading]
   :dedent: 4

Note that the second argument, the one corresponding to the file for the data, requires a :py:class:`svs.VectorDataLoader` and
the corresponding data type.


.. _search_with_compression:

Search using vector compression
*******************************

:ref:`Vector compression <vector_compression>` can be used to speed up the search. It can be done on the fly by loading
the index with a :py:class:`LVQLoader <svs.LVQLoader>` (:ref:`details for Python <python_api_loaders>`)
or by :ref:`loading an index with a previously compressed dataset <loading_compressed_indices>`.

See :ref:`compression-setting` for details on setting the compression parameters.

First, specify the compression loader. Specifying ``dims`` in :py:class:`svs.VectorDataLoader` is optional and
:ref:`can boost performance considerably <static-dim>` (:ref:`see <static-dim-for-lvq>` for details on how to enable
this functionality).

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [search-compressed-loader]
   :end-before: [search-compressed-loader]
   :dedent: 4

Then load the index and run the search as usual.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [search-compressed]
   :end-before: [search-compressed]
   :dedent: 4

.. note::
   Vector compression is usually accompanied by an accuracy loss for the same search window size and may require
   increasing the window size to compensate.


.. _loading_compressed_indices:


Saving an index with compressed vectors
=======================================

SVS has support to save and load indices with a previously compressed dataset.
The saving and loading procedures are the same as with uncompressed vectors.


.. _entire_example:

Entire example
**************

This ends the example demonstrating the features of the Vamana index.
The entire executable code is shown below.
Please reach out with any questions.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
