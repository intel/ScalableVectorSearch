.. _vamana:

Vamana Index
============

Main Vamana Class
-----------------

.. doxygenclass:: svs::index::vamana::VamanaIndex
   :project: SVS
   :members:

Vamana Entry Point
------------------

Instantiating an instance of the VamanaIndex can be tricky and has many different pieces that need to come together correctly.
To assist with this operation, the factor methods are supplied that can handle many different (both documented and documented) combinations of data set types, graph types, and distances.
These methods are documented below.

.. doxygenfunction:: svs::index::vamana::auto_assemble
   :project: SVS

.. doxygenfunction:: svs::index::vamana::auto_build
   :project: SVS

Vamana Utility Classes
----------------------

Miscellaneous helper classes and functions to be used with the VamanaIndex are documented in this section.

.. doxygenstruct:: svs::index::vamana::NoPostOp
   :project: SVS
   :members:

