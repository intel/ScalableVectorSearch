.. _start:

Getting Started
################
This tutorial will show you how to install SVS and run your first search with it! Tutorials for running
:ref:`dynamic indexing <how_to_run_dynamic_indexing>`, setting :ref:`index <graph-build-param-setting>` and
:ref:`search<search-window-size-setting>` parameters, :ref:`using vector compression <compression-setting>`, as well as
more :ref:`advanced installation options <build>` are also available.

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

* `OneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_ (:ref:`installation details <one-mkl-install>`)

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

.. _one-mkl-install:

OneMKL installation
===================
`OneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_ can be installed as part of the
`Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html#gs.8u2swh>`_ by following one of the methods indicated in the `oneAPI docs <https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-1/installation.html>`_ .

For example, the following commands show how to install the OneMKL component of the Intel oneAPI Base Toolkit on a Linux
system using the `offline installer <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_:

.. code-block:: sh

    wget [link to the offline installer]
    sudo sh [downloaded installer script] -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s
    source /opt/intel/oneapi/setvars.sh


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

   :py:class:`pysvs.Vamana.build` supports building from Numpy arrays with dtypes float32, float16, int8 and uint8.


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

:ref:`Vector compression <vector_compression>` can be used to speed up the search. It can be done on the fly by loading
the index with a :py:class:`LVQLoader <pysvs.LVQLoader>` (:ref:`details for Python <python_api_loaders>`, :ref:`details for C++ <cpp_quantization_lvq>`)
or by :ref:`loading an index with a previously compressed dataset <loading_compressed_indices>`.

See :ref:`compression-setting` for details on setting the compression parameters.

**In Python**

First, specify the compression loader. Specifying ``dims`` in :py:class:`pysvs.VectorDataLoader` is optional and
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

.. note::
   Vector compression is usually accompanied by an accuracy loss for the same search window size and may require
   increasing the window size to compensate.


.. _loading_compressed_indices:

Saving an Index with Compressed Vectors
=======================================

SVS has support to save and load indices with a previously compressed dataset.
Just follow the same procedure for :ref:`saving <index_saving>` and :ref:`loading <index_loading>` indices with
uncompressed vectors.


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