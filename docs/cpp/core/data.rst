.. _cpp_core_data:

In-Memory Representation of Data
================================

Vector data is at the core of the SVS similarity search library.
Several specific classes are provided to implement in-memory vector datasets with different semantics.

* :cpp:class:`svs::data::SimplePolymorphicData` - General dense representation of embedding vectors using type erasure for the backend allocator.
  This allows multiple different allocators to be used without changing the type of this class.
* :cpp:class:`svs::data::SimpleData` - Special version of :cpp:class:`svs::data::SimplePolymorphicData` where the allocator is propagated as a type parameter.
  Is slightly more efficient than the polymorphic container but less flexible.
* :cpp:class:`svs::data::SimpleDataView` - Non-owning view over a dense representation of embedding vectors.
* :cpp:class:`svs::data::ConstSimpleDataView` - Constant version of :cpp:class:`svs::data::SimpleDataView`.
  Useful for crossing virtual function boundaries where templates can't be used.

Detailed documentation for these classes is given below.

.. doxygenclass:: svs::data::SimplePolymorphicData
   :project: SVS
   :members:

.. doxygenclass:: svs::data::SimpleData
   :project: SVS
   :members:

.. doxygenclass:: svs::data::SimpleDataView
   :project: SVS
   :members:

.. doxygenclass:: svs::data::ConstSimpleDataView
   :project: SVS
   :members:

Data Loading
------------

The :cpp:class:`svs::VectorDataLoader` class provides a way to instantiate a :cpp:class:`svs::data::SimplePolymorphicData` object from multiple different kinds of file types.

.. doxygenclass:: svs::VectorDataLoader
   :project: SVS
   :members:


.. NOTE::

   The various data implementations given above are all instances of the more general concept :cpp:concept:`svs::data::ImmutableMemoryDataset`.
   Where possible, this concept is use to constrain template arguments, allowing for future custom implementations.
