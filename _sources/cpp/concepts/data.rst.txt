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

.. _data_concept:

Abstract Datasets
=================

The concepts :cpp:concept:`svs::data::ImmutableMemoryDataset` and :cpp:concept:`svs::data::MemoryDataset` are used to encapsulate the expected behavior of classes implementing datasets.
These concepts are described below.
Concrete implementations of these concepts can be found :ref:`here <cpp_core_data>`.

Main Concepts
^^^^^^^^^^^^^

This sub-section highlights the main exported concepts that are expected to be used by this logical grouping of code.

.. doxygenconcept:: svs::data::ImmutableMemoryDataset
   :project: SVS

.. doxygenconcept:: svs::data::MemoryDataset
   :project: SVS


Public API
^^^^^^^^^^

.. doxygenconcept:: svs::data::HasValueType
   :project: SVS

.. doxygengroup:: data_concept_public
   :project: SVS
   :members:
   :content-only:
