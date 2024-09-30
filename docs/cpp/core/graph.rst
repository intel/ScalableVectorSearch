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

.. _cpp_core_graph:

Graphs
======

.. doxygenclass:: svs::graphs::SimpleGraph
   :project: SVS
   :members:

Graph Loading
-------------

.. doxygenstruct:: svs::GraphLoader
   :project: SVS
   :members:

.. NOTE::

   The various graph implementations given above are all instances of the more general concept :cpp:concept:`svs::graphs::ImmutableMemoryGraph`.
   Where possible, this concept is use to constrain template arguments, allowing for future custom implementations.
