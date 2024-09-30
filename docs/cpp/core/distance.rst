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

