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

.. _graph_concept:

Abstract Graphs
===============

Like :ref:`abstract datasets <data_concept>`, the graph concepts :cpp:concept:`svs::graphs::ImmutableMemoryGraph` and :cpp:concept:`svs::graphs::MemoryGraph` are used to model the expected behavior of graphs.
Concrete implementations can be found :ref:`here <cpp_core_graph>`.

Main Concepts
^^^^^^^^^^^^^

.. doxygenconcept:: svs::graphs::ImmutableMemoryGraph
   :project: SVS

.. doxygenconcept:: svs::graphs::MemoryGraph
   :project: SVS

Public API
^^^^^^^^^^

.. doxygengroup:: graph_concept_public
   :project: SVS
   :members:
   :content-only:
