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
