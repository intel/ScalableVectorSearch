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
