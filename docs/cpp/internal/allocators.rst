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

.. _allocators:

High Level Allocators
=====================

Many of the large data structures used in the library to store graphs or large blocks of dense data support using different allocators to provide their memory.
This enables easy use of hugepages for large allocations or for direct memory mapping of files if the on-disk representation of the file is suitable.

Algorithms designed to accomodate multiple allocators are templated on the allocator type.

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

