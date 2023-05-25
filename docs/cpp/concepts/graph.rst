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
