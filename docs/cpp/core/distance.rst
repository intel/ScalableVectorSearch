.. _cpp_core_distance:

Built-In Distance Functors
==========================

This section describes concrete implementations of distance functors.
For utilities and concepts related to abstract distance computation, refer to :ref:`distance concepts <distance_concept>`.

.. doxygenstruct:: svs::distance::DistanceL2
   :project: SVS
   :members:

.. doxygenstruct:: svs::distance::DistanceIP
   :project: SVS
   :members:

.. doxygenstruct:: svs::distance::DistanceCosineSimilarity
   :project: SVS
   :members:

Built-in Distance Overloads
---------------------------

.. doxygengroup:: distance_overload
   :project: SVS
   :members:
   :content-only:

Public Distance Utilities
-------------------------

.. doxygenenum:: svs::DistanceType
   :project: SVS

.. doxygenvariable:: svs::distance_type_v
   :project: SVS

.. doxygenclass:: svs::DistanceDispatcher
   :project: SVS
   :members:

