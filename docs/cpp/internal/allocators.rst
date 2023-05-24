.. _allocators:

High Level Allocators
=====================

Many of the large data structures used in the library to store graphs or large blocks of dense data support using different allocators to provide their memory.
This enables easy use of hugepages for large allocations or for direct memory mapping of files if the on-disk representation of the file is suitable.

Algorithms designed to accomodate multiple allocators will either be templated on the allocator type, or accept a predefined list of standard allocators.

.. doxygentypedef:: svs::StandardAllocators

Furthermore, the smart pointers returned by these allocators can all be converted to the :cpp:class:`svs::lib::PolymorphicPointer`.

Standard Allocators
-------------------

.. doxygengroup:: core_allocators_entry
   :project: SVS
   :members:
   :content-only:

Helpers
-------

.. doxygengroup:: core_allocators_public
   :project: SVS
   :members:
   :content-only:

