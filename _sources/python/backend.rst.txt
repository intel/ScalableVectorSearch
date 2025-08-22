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

.. _python_backend:

Backend Selection
=================

.. warning::

    This is an advanced topic that may affect application performance or even lead to
    mysterious crashes. This section is meant for educational purposes and for those who
    wish to dig a little deeper.

The C++ library makes extensive use of Intel(R) AVX-512 instructions both in the form of direct intrinsics and through the `EVE <https://github.com/jfalcou/eve>`_ SIMD library.
Along with design decisions around generic programming and static dimensionality, this makes it quite difficult to surgically extract the portions of the library that are accelerated by these instructions.
In order to support multiple CPU micro-architectures including those that lack Intel(R) AVX-512 support, we instead ship multiple versions of the backing shared library, each targeting a different micro-architecture.
At run-time, the svs Python module tries to select the best backend for the current CPU.

There are several environment variables that can be set prior to loading the library that can influence this behavior.

* ``SVS_QUIET=YES``
    The svs loading logic may emit warnings when loading the backend.
    This can occur either when the loaded backend is targeting an old architecture and may have poor performance, or when the selection logic is explicitly bypassed and the resulting backend may not be compatible with the current CPU.

    These warning are suppressed when this environment variable is defined.

* ``SVS_OVERRIDE_BACKEND=<backend-name>``
    Explicitly bypass the backend selection logic.
    See the :py:func:`svs.loader.available_backends` for a list of valid strings for this variable.
    Misusing this feature can cause the application to not load or crash.

Documentation for functions regarding backend selection and the current backend are given below.

.. automodule:: svs.loader
   :members:
