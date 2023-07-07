.. _start:

Getting Started
################
This tutorial will show you how to install SVS and run your first search with it! Tutorials for more :ref:`advanced
installation options <build>`, as well as :ref:`data indexing <graph-build-param-setting>`, :ref:`search options <search-window-size-setting>`
and :ref:`using vector compression <compression-setting>` are also available.

.. contents::
   :local:
   :depth: 2

.. _install_pysvs:

Installation
************

Building and installing SVS should be relatively straight-forward.

Prerequisites
=============

* A C++20 capable compiler:

  * GCC >= 11.0
  * Clang >= 13.0

To install the Python module you'll also need:

* Python >= 3.7
* A working internet connection

Python build
============

To build and install the Python module, pysvs, clone the repo and run the following pip install command.

.. code-block:: sh

    # Clone the repository
    git clone https://github.com/IntelLabs/ScalableVectorSearch.git
    cd ScalableVectorSearch

    # Install pysvs using pip
    CC=gcc-11 CXX=g++-11 pip install bindings/python

If you encounter any issues with the pip install command, we suggest you follow an alternative installation procedure using
`cibuildwheel <https://cibuildwheel.readthedocs.io/en/stable/>`_. To generate a wheel using your current version of
Python you will need to have cibuildwheel installed as well as `docker <https://www.docker.com/>`_.
Once those are installed, follow these steps:

1. Navigate to the root directory of the source and, if the ``bindings/python/_skbuild`` folder exists, remove it.

2. From the root directory of the source run

.. code-block:: sh

    cibuildwheel --only $(python tools/pybuild.py) bindings/python

3. Then simply run

.. code-block:: sh

    pip install ./wheelhouse/pysvs*.whl

For more details see :ref:`building_python_library`.

C++ build
==========
SVS provides a cmake target to enable source builds against the library. See :ref:`cpp_cmake_support` for details.

Generating test data
********************

We generate a sample dataset using the :py:func:`pysvs.generate_test_dataset` generation function.
This function generates a data file, a query file, and the ground truth. Note that this is randomly generated data,
with no semantic meaning for the elements within it.

We first load pysvs and the os module also required for this example.

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

Building the Index
******************

Now that data has been generated, we need to construct an index over that data.
The index is a graph connecting related data vectors in such a way that searching for nearest neighbors yields good results.
The first step is to define the hyper-parameters of the graph we wish to construct.
Don't worry too much about selecting the correct values for these hyper-parameters right now.
This usually involves a bit of experimentation and is dataset dependent. See :ref:`graph-build-param-setting` for details.

**In Python**

This is done by creating an instance of :py:class:`pysvs.VamanaBuildParameters`.

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [build-parameters]
   :end-before: [build-parameters]
   :dedent: 4

**In C++**

.. collapse:: Click to display

    This is done by creating an instance of :cpp:class:`svs::index::vamana::VamanaBuildParameters`.

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Build Parameters]
       :end-before: [Build Parameters]
       :dedent: 4

|

Now that we've established our hyper-parameters, it is time to construct the index.
Passing the ``dims`` is optional, but may :ref:`yield performance benefits if given <static-dim>`.

**In Python**

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

Note the use of :py:class:`pysvs.VectorDataLoader` to indicate both the file path and the :ref:`data type <supported_data_types>`
of the ``fvecs`` file on disk (see :ref:`io` for supported file formats). See :py:class:`pysvs.Vamana.build`
for details about the build function.

.. note::

   :py:class:`pysvs.Vamana.build` only supports building from Numpy arrays with dtypes float32, int8 and uint8.


**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Index Build]
       :end-before: [Index Build]
       :dedent: 4

    Note the use of :cpp:class:`svs::VectorDataLoader` to indicate both the path to the dataset file ``data_vecs`` and the
    :ref:`data type <supported_data_types>` of the file on disk (see :ref:`io` for supported file formats).
    See :cpp:func:`svs::Vamana::build` for details about the build function.

|

Searching the Index
********************

The graph is now built and we can perform queries over the graph.
First, we load the queries and the computed ground truth for our example dataset.

**In Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [load-aux]
   :end-before: [load-aux]
   :dedent: 4

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Load Aux]
       :end-before: [Load Aux]
       :dedent: 4

|

Performing queries is easy.
First establish a base-line search window size. This provides a parameter by which performance and accuracy can be traded.
The larger ``search_window_size`` is, the higher the accuracy but the lower the performance.
Note that ``search_window_size`` must be at least as large as the desired number of neighbors.
See :ref:`search-window-size-setting` for details.

We use the search function to find the 10 approximate nearest neighbors to each query.
Then, we compute the 10-recall at 10 of the returned neighbors, checking to confirm
the accuracy.

**In Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [perform-queries]
   :end-before: [perform-queries]
   :dedent: 4

See :py:class:`pysvs.Vamana.search` for details about the search function.

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Perform Queries]
       :end-before: [Perform Queries]
       :dedent: 4

    See :cpp:func:`svs::Vamana::search` for details about the search function.

|

.. _index_saving:

Saving the Index
****************

If you are satisfied with the performance of the generated index, you can save it to disk to avoid rebuilding it in the future.

**In Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [saving-results]
   :end-before: [saving-results]
   :dedent: 4

See :py:func:`pysvs.Vamana.save` for details about the save function.

**In C++**

.. collapse:: Click to display

    See :cpp:func:`svs::Vamana::save` for details about the save function.

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Saving]
       :end-before: [Saving]
       :dedent: 4

|

.. note::

    The save index function currently uses three folders for saving.
    All three are needed to be able to reload the index.

    * One folder for the graph.
    * One folder for the data.
    * One folder for metadata.

    This is subject to change in the future.

.. _index_loading:

Reloading a Saved Index
***********************

To reload the index from file, use the corresponding constructor with the three folder names used to save the index.
Performing queries is identical to before.

**In Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [loading]
   :end-before: [loading]
   :dedent: 4

Note that the second argument, the one corresponding to the file for the data, requires a :py:class:`pysvs.VectorDataLoader` and
the corresponding data type.

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Loading]
       :end-before: [Loading]
       :dedent: 4

    Note that the second argument, the one corresponding to the file for the data, requires a :cpp:class:`svs::VectorDataLoader` and
    the corresponding data type. See the :ref:`entire executable code <entire_cpp_example>` for details about the ``run_recall`` function.

|

.. _search_with_compression:

Search using Vector Compression
*******************************

The library has experimental support for performing online :ref:`vector compression <vector_compression>`, which
:ref:`needs to be enabled at compile time for each vector dimensionality <enabling_vector_compression>`.

Once enabled, vector compression can be used for search by choosing a compressed file loader
(:ref:`details for Python <python_api_loaders>`, :ref:`details for C++ <cpp_quantization_lvq>`) when loading the graph.
The loader will compress an uncompressed dataset on the fly. :ref:`Loading an index with a previously compressed dataset
<loading_compressed_indices>` is also supported.

Specifying the loader is all that is required to use vector compression. The loader takes as argument a ``VectorDataLoader``
structure (:py:class:`pysvs.VectorDataLoader`, :ref:`VectorDataLoader for C++ <cpp_core_data>`) with the path to the file containing
the uncompressed dataset (**only the** ``.svs`` **format is currently supported**) and its corresponding data type.
Padding is an optional parameter that specifies the value (in bytes) to align the beginning of each compressed vector.
Values of 32 or 64 (half or full cache lines alignment) may offer the best performance at the cost of a lower compression ratio.
A value of 0 (default) implies no special alignment.

See :ref:`compression-setting` for details on choosing the best compression loader.
Performing queries is identical to before.

**In Python**

Specifying ``dims`` in :py:class:`pysvs.VectorDataLoader` is **not** optional when it will be used in combination with a
vector compression loader as in this example

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [search-compressed-loader]
   :end-before: [search-compressed-loader]
   :dedent: 4

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python
   :start-after: [search-compressed]
   :end-before: [search-compressed]
   :dedent: 4

To see which loaders are applicable to which methods, study the signature in :py:func:`pysvs.Vamana.__init__` carefully.

**In C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Compressed Loader]
       :end-before: [Compressed Loader]
       :dedent: 4

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp
       :start-after: [Search Compressed]
       :end-before: [Search Compressed]
       :dedent: 4

    See the :ref:`entire executable code <entire_cpp_example>` for details about the ``run_recall`` function.

|

.. warning::

   Remember to enable vector compression for the desired dimensionality following these :ref:`steps <enabling_vector_compression>`.

.. note::
   Vector compression is usually accompanied by an accuracy loss for the same search window size and may require
   increasing the window size to compensate.


.. _loading_compressed_indices:

Saving an Index with Compressed Vectors
=======================================

SVS has support to save and load indices with a previously compressed dataset.
Just follow the same procedure for :ref:`saving <index_saving>` and :ref:`loading <index_loading>` indices with
uncompressed vectors.

.. note::

    Saving padded datasets is not supported but it is planned for a near future release.

Entire Example
**************

This ends the example demonstrating the features of the Vamana index.
The entire executable code is shown below.
Please reach out with any questions.

**Entire code in Python**

.. literalinclude:: ../examples/python/example_vamana.py
   :language: python

.. _entire_cpp_example:

**Entire code in C++**

.. collapse:: Click to display

    .. literalinclude:: ../examples/cpp/vamana.cpp
       :language: cpp

|