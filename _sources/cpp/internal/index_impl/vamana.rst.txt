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

