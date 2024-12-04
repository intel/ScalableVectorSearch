.. _python_common:

Common Python API
=================

Memory Allocators
-----------------

.. autoclass:: svs.DRAM

Enums
-----

.. autoclass:: svs.DistanceType

.. autoclass:: svs.DataType

.. _python_common_helpers:

Helper Functions
----------------

.. autofunction:: svs.read_vecs

.. autofunction:: svs.write_vecs

.. warning::

    The user must specify the file extension corresponding to the desired file format in the ``filename`` argument of
    :py:func:`svs.write_vecs`.

.. autofunction:: svs.read_svs

.. autofunction:: svs.convert_fvecs_to_float16

.. autofunction:: svs.generate_test_dataset

.. autofunction:: svs.convert_vecs_to_svs
