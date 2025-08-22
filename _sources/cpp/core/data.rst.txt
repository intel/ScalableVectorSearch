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

.. _cpp_core_data:

In-Memory Representation of Data
================================

Vector data is at the core of the SVS similarity search library.
Several specific classes are provided to implement in-memory vector datasets with different semantics.

* :cpp:class:`svs::data::SimpleData` - General allocator-aware dense representation of embedding vectors.
* :cpp:type:`svs::data::SimpleDataView` - Non-owning view over a dense representation of embedding vectors.
* :cpp:type:`svs::data::ConstSimpleDataView` - Constant version of :cpp:type:`svs::data::SimpleDataView`.
  Useful for crossing virtual function boundaries where templates can't be used.

Detailed documentation for these classes is given below.

.. doxygenclass:: svs::data::SimpleData
   :project: SVS
   :members:

.. doxygentypedef:: svs::data::ConstSimpleDataView
   :project: SVS

.. doxygentypedef:: svs::data::SimpleDataView
   :project: SVS

Data Loading
------------

The :cpp:class:`svs::VectorDataLoader` class provides a way to instantiate a :cpp:class:`svs::data::SimplePolymorphicData` object from multiple different kinds of file types.

.. doxygenclass:: svs::VectorDataLoader
   :project: SVS
   :members:


.. NOTE::

   The various data implementations given above are all instances of the more general concept :cpp:concept:`svs::data::ImmutableMemoryDataset`.
   Where possible, this concept is use to constrain template arguments, allowing for future custom implementations.
