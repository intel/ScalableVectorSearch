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
