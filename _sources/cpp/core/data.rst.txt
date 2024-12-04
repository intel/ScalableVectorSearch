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
