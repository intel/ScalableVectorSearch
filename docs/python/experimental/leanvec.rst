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

.. _python_example_leanvec_vamana:

Using LeanVec
#############

.. warning::

   This data structure in its current state is experimental.
   It can and will change and/or be removed without deprecations.
   Use at your own risk.
   If you are interested in robust support, please contact the library authors or maintainers.

`LeanVec <https://arxiv.org/abs/2312.16335>`_ is a performance acceleration strategy that
uses linear transformations to reduce the dimensionality of a given dataset.

For graph search, the LeanVec implementation uses two datasets:

* Primary: A version of the data with reduced dimensionality.
  Graph searches are done over this primary dataset.
  Because of the reduced dimensionality, distance computations are faster and memory bandwidth requirements are decreased.

* Secondary: A version of the data with full dimensionality.
  The primary graph search yields a collection of candidate nearest neighbors.
  These candidates are then refined using the full-precision secondary dataset.

Preamble
========

We first need to setup the example environment.
Following the :ref:`basic Vamana example <python_vamana_inline_example>`, we import ``svs`` and other required modules.

.. literalinclude:: ../../../examples/python/example_vamana_leanvec.py
   :language: python
   :start-after: [imports]
   :end-before: [imports]

Next, we create an example dataset.

.. literalinclude:: ../../../examples/python/example_vamana_leanvec.py
   :language: python
   :start-after: [generate-dataset]
   :end-before: [generate-dataset]
   :dedent: 4

Constructing a LeanVec Loader
=============================

As with the :py:class:`svs.LVQLoader`, the :py:class:`svs.LeanVecLoader` can perform dynamic compression of uncompressed vectors.
An example is shown below.

.. literalinclude:: ../../../examples/python/example_vamana_leanvec.py
   :language: python
   :start-after: [create-loader]
   :end-before: [create-loader]
   :dedent: 4

In this example, we construct a :py:class:`svs.LeanVecLoader` with a reduced dimensionality of 128.
This means that the bulk of the graph search will be done using a 128-dimensional transformation of the 256 dimensional dataset we just generated.
Furthermore, we can choose the encodings of the primary and secondary dataset.
The example demonstrates using LVQ8 for the primary dataset and ``float16`` for the full-dimensional secondary dataset.

Index Building and Searching
============================

Index construction and search are done exactly as :ref:`before <python_vamana_inline_example>`, with the minor caveat of using an alpha value less than 1 since we are using the inner product distance function.

.. literalinclude:: ../../../examples/python/example_vamana_leanvec.py
   :language: python
   :start-after: [build-and-search-index]
   :end-before: [build-and-search-index]
   :dedent: 4

Entire Example
==============

The entire code for this example is shown below.

.. literalinclude:: ../../../examples/python/example_vamana_leanvec.py
   :language: python