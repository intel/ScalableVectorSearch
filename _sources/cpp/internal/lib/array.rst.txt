.. Copyright (C) 2023 Intel Corporation
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

.. _cpp_lib_array:

Multi-dimensional Array
=======================

Many of the data structures in the library are backed by a multi-dimensional array class.

.. doxygenclass:: svs::DenseArray
   :project: SVS
   :members:

Type-deducing constructors
--------------------------

The family of ``svs::make_dense_array`` methods assist in array construction.

.. doxygengroup:: make_dense_array_family
   :project: SVS
   :members:
   :content-only:
