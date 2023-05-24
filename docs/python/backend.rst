.. _python_backend:

Backend Selection
=================

.. warning::

    This is an advanced topic that may affect application performance or even lead to
    mysterious crashes. This section is meant for educational purposes and for those who
    wish to dig a little deeper.

The C++ library makes extensive use of AVX-512 instructions both in the form of direct intrinsics and through the `EVE <https://github.com/jfalcou/eve>`_ SIMD library.
Along with design decisions around generic programming and static dimensionality, this makes it quite difficult to surgically extract the portions of the library that are accelerated by these instructions.
In order to support multiple CPU micro-architectures including those that lack AVX-512 support, we instead ship multiple versions of the backing shared library, each targeting a different micro-architecture.
At run-time, the pysvs Python module tries to select the best backend for the current CPU.

There are several environment variables that can be set prior to loading the library that can influence this behavior.

* ``PYSVS_QUIET=YES``
    The pysvs loading logic may emit warnings when loading the backend.
    This can occur either when the loaded backend is targeting an old architecture and may have poor performance, or when the selection logic is explicitly bypassed and the resulting backend may not be compatible with the current CPU.

    These warning are suppressed when this environment variable is defined.

* ``PYSVS_OVERRIDE_BACKEND=<backend-name>``
    Explicitly bypass the backend selection logic.
    See the :py:func:`pysvs.loader.available_backends` for a list of valid strings for this variable.
    Misusing this feature can cause the application to not load or crash.

Documentation for functions regarding backend selection and the current backend are given below.

.. automodule:: pysvs.loader
   :members:
