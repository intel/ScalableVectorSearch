.. _python_common:

Common Python API
=================

Memory Allocators
-----------------

.. autoclass:: pysvs.DRAM

Enums
-----

.. autoclass:: pysvs.DistanceType

.. autoclass:: pysvs.DataType

.. _python_common_helpers:

Helper Functions
----------------

.. autofunction:: pysvs.read_vecs

.. autofunction:: pysvs.write_vecs

.. warning::

    The user must specify the file extension corresponding to the desired file format in the ``filename`` argument of
    :py:func:`pysvs.write_vecs`.

.. autofunction:: pysvs.read_svs

.. autofunction:: pysvs.convert_fvecs_to_float16

.. autofunction:: pysvs.generate_test_dataset

.. autofunction:: pysvs.convert_vecs_to_svs
