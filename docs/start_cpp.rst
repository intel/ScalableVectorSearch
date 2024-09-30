.. Copyright (C) 2024 Intel Corporation
..
.. This software and the related documents are Intel copyrighted materials,
.. and your use of them is governed by the express license under which they
.. were provided to you ("License"). Unless the License provides otherwise,
.. you may not use, modify, copy, publish, distribute, disclose or transmit
.. this software or the related documents without Intel's prior written
.. permission.
..
.. This software and the related documents are provided as is, with no
.. express or implied warranties, other than those that are expressly stated
.. in the License.

.. _start_cpp:

Getting started with C++
########################
This tutorial will show you how to install SVS and run your first search with it!

.. contents::
   :local:
   :depth: 2

.. _install_svs:

Building
********

Building SVS should be relatively straight-forward. We test on Ubuntu 22.04 LTS, but any Linux distribution should work.

Prerequisites
=============

.. include:: prerequisites.rst

CMake build
===========

To build SVS and the included examples, use the following:

.. code-block:: sh

    mkdir build && cd build
    cmake .. -DSVS_BUILD_EXAMPLES=YES
    cmake --build . -j$(nproc)

Verifying the build
====================

Run the following command to verify that SVS was successfully installed. It should print some types, like ``float32``.

.. code-block:: sh

    examples/cpp/types


SVS search example
******************

In this tutorial we will showcase the most important features of SVS. The
:ref:`full example <entire_example_cpp>` is available at the end of this
tutorial. You can run it with the following command:

.. code-block:: sh

   examples/cpp/vamana ../data/test_dataset/data_f32.fvecs ../data/test_dataset/queries_f32.fvecs ../data/test_dataset/groundtruth_euclidean.ivecs


Example data
************

We will use the random dataset included in SVS for testing in ``data/test_dataset``.


Building the index
******************

Before searching, we need to construct an index over that data.
The index is a graph connecting related data vectors in such a way that searching for nearest neighbors yields good results.
The first step is to define the hyper-parameters of the graph we wish to construct.
Don't worry too much about selecting the correct values for these hyper-parameters right now.
This usually involves a bit of experimentation and is dataset dependent. See :ref:`graph-build-param-setting` for details.

This is done by creating an instance of :cpp:class:`svs::index::vamana::VamanaBuildParameters`.

.. literalinclude:: ../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Build Parameters]
   :end-before: [Build Parameters]
   :dedent: 4

Now that we have established our hyper-parameters, it is time to build the index.
Passing the ``dims`` is optional, but may :ref:`yield performance benefits if given <static-dim>`.


.. literalinclude:: ../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Index Build]
   :end-before: [Index Build]
   :dedent: 4

Note the use of :cpp:class:`svs::VectorDataLoader` to indicate both the path to the dataset file ``data_vecs`` and the
:ref:`data type <supported_data_types>` of the file on disk (see :ref:`io` for supported file formats).
See :cpp:func:`svs::Vamana::build` for details about the build function.


Searching the index
*******************

The graph is now built and we can perform queries over the graph.
First, we load the queries from a file on disk.

After searching, we compare the search results with ground truth results which
we also load from the SVS test dataset.

.. literalinclude:: ../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Load Aux]
   :end-before: [Load Aux]
   :dedent: 4

Performing queries is easy.
First establish a base-line search window size. This provides a parameter by which performance and accuracy can be traded.
The larger ``search_window_size`` is, the higher the accuracy but the lower the performance.
Note that ``search_window_size`` must be at least as large as the desired number of neighbors.
See :ref:`search-window-size-setting` for details.

We use the search function to find the 10 approximate nearest neighbors to each query.
See :cpp:func:`svs::Vamana::search` for details about the search function.
Then, we compute the 10-recall at 10 of the returned neighbors, checking to confirm
the accuracy against the ground truth.


.. literalinclude:: ../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Perform Queries]
   :end-before: [Perform Queries]
   :dedent: 4


Saving the index
****************

If you are satisfied with the performance of the generated index, you can save it to disk to avoid rebuilding it in the future.

See :cpp:func:`svs::Vamana::save` for details about the save function.

.. literalinclude:: ../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Saving]
   :end-before: [Saving]
   :dedent: 4

.. note::

    The save index function currently uses three folders for saving.
    All three are needed to be able to reload the index.

    * One folder for the graph.
    * One folder for the data.
    * One folder for metadata.

    This is subject to change in the future.


Reloading a saved index
***********************

To reload the index from file, use the corresponding constructor with the three folder names used to save the index.
Note that the second argument, the one corresponding to the file for the data, requires a :cpp:class:`svs::VectorDataLoader` and
the corresponding data type. 

After reloading, performing queries is identical to before.
See the :ref:`entire example code <entire_example_cpp>` for details about
the ``run_recall`` function used to check the search results against ground
truth.

.. literalinclude:: ../examples/cpp/vamana.cpp
   :language: cpp
   :start-after: [Loading]
   :end-before: [Loading]
   :dedent: 4


Search using vector compression
*******************************

:ref:`Vector compression <vector_compression>` can be used to speed up search. It can be done with a :cpp:class:`svs::quantization::lvq::LVQDataset`.

See :ref:`compression-setting` for details on setting the compression parameters.

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

.. note::
   Vector compression is usually accompanied by an accuracy loss for the same search window size and may require
   increasing the window size to compensate.


Saving an index with compressed vectors
=======================================

SVS has support to save and load indices with a previously compressed dataset.
The saving and loading procedures are the same as with uncompressed vectors.

.. _entire_example_cpp:

Entire example
**************

This ends the example demonstrating the features of the Vamana index.
The entire executable code is shown below.
Please reach out with any questions.

.. literalinclude:: ../examples/cpp/vamana.cpp
   :language: cpp
