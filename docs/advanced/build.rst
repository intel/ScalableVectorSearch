.. _build:

Library Building
****************
The SVS library comes in three main forms:

    * A header only C++ library.
    * A small set of command-line utilities for exploring similarity search.
    * A Python wheel for interactive use.

Extensive use of C++20 is used both through-out the SVS library and its dependencies (mainly the `EVE <https://github.com/jfalcou/eve>`_ SIMD library).
The minimum tested compilers are

    * GCC 11.0
    * Clang 13.0

.. contents::
   :local:
   :depth: 1


.. _building_python_library:

Building Python Library
=======================

Native Wheel
------------

We support two methods for building the binary wheel for the Python interface to the library: (1) as a native build and (2) as a more portable containerized build through cibuildwheel (see :ref:`cibuildwheel <build_cibuildwheel>`).

The following commands will perform the **native** build:

.. code-block:: sh

   cd bindings/python
   CC=gcc CXX=g++ python setup.py bdist_wheel -- [cmake arguments] -- -j$(nproc)
   pip install ./dist/pysvs*.whl

If the default build options are acceptable, the CMake arguments may be left empty.
Alternatively, pip can be used to drive the entire compilation process by (running from the root git directory)

.. code-block:: sh

   CC=gcc CXX=g++ pip install bindings/python -vvv

Building Without AVX512
^^^^^^^^^^^^^^^^^^^^^^^

The snippet below shows how to build the Python library without AVX512 instructions.
This is not recommended as AVX512 is critical for performance in many aspects of the library, but may be necessary if running on older hardware.

.. code-block:: sh

   cd bindings/python
   CC=gcc CXX=g++ python setup.py bdist_wheel -- -DSVS_NO_AVX512=YES -- -j$(nproc)
   pip install ./dist/pysvs*.whl

Debug Builds for Native Wheels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every once in a while, it is beneficial to use either a debug build of the Python wheel or at least build the wheel with debug information.
This can be difficult through the pip only approach as it likes to transitively build dependencies.
Instead, it is recommended to go through the two stage approach:

.. code-block:: sh

   cd bindings/python
   CC=gcc CXX=g++ python setup.py bdist_wheel --build-type=Debug -- -- -j$(nproc)

Building Notes
--------------

Occasionally, the Python build process will break mysteriously.
This usually occurs when messing around with different compilers, compile-time variables, and build strategies.
If this happens, try removing ``./bindings/python/_skbuild`` and ``./bindings/python/dist`` and going again.

C++ Build
=========

.. _cpp_cmake_support:

CMake Support
-------------

SVS provides a cmake target to enable source builds against the library:

    * ``svs::svs``: Links the code headers and shared library components (if applicable).
    * ``svs::compile_options``: Compiler flags helpful for building the libary.

Usage In CMake
^^^^^^^^^^^^^^

To include the C++ portion of the library in a CMake based project, follow the template below.

.. code-block:: cmake

    include(FetchContent)
    FetchContent_Declare(
        svs
        GIT_REPOSITORY https://github.com/IntelLabs/ScalableVectorSearch.git
        GIT_TAG dev
    )

    FetchContent_MakeAvailable(svs)

    # Link with the library
    target_link_libraries([my_target] PRIVATE|PUBLIC|INTERFACE svs::svs)

Installing Locally
^^^^^^^^^^^^^^^^^^

The C++ library can also be installed locally using CMake's installation logic.
**Note**: This approach is not recommended.

.. code-block:: sh

   mkdir build
   cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=my/directory
   cmake --build .
   cmake --install .


Building Utility Binaries
-------------------------

To build the utility binaries, unit tests, and examples, use the following:

.. code-block:: sh

    mkdir build
    cd build
    cmake .. -DSVS_BUILD_BINARIES=YES -DSVS_BUILD_TESTS=YES -DSVS_BUILD_EXAMPLES=YES
    cmake --build . -j$(nproc)

Build Options
=============

SVS supports the following build-time options.

.. table:: SVS CMake variables
    :width: 100

    +--------------------+--------------------+-----------------------------------------------+
    | CMake Option       | Supported Values   | Description                                   |
    |                    | (defaults in bold) |                                               |
    +====================+====================+===============================================+
    | SVS_BUILD_BINARIES | ON, **OFF**        | Build utility binaries.                       |
    +--------------------+--------------------+-----------------------------------------------+
    | SVS_BUILD_TESTS    | ON, **OFF**        | Build the unit test suite.                    |
    +--------------------+--------------------+-----------------------------------------------+
    | SVS_BUILD_DOCS     | ON, **OFF**        | Build the documentation.                      |
    +--------------------+--------------------+-----------------------------------------------+
    | SVS_BUILD_EXAMPLES | ON, **OFF**        | Build the documentation examples.             |
    +--------------------+--------------------+-----------------------------------------------+
    | SVS_NO_AVX512      | ON, **OFF**        | Disable AVX512 instructions from being used.  |
    |                    |                    | Helpful when running Valgrind as that tool    |
    |                    |                    | does not support AVX512.                      |
    |                    |                    |                                               |
    |                    |                    | This option is not required when compiling on |
    |                    |                    | non-AVX512 systems.                           |
    |                    |                    |                                               |
    |                    |                    | May be helpful on older systems that          |
    |                    |                    | experience down-clocking when using AVX512    |
    |                    |                    | instructions.                                 |
    +--------------------+--------------------+-----------------------------------------------+

The following variables can be found in CMake files but are intended for development and debug purposes.
As such, they are subject to change without notice.
Please avoid using them.

+---------------------------------+--------------------+-----------------------------------------------+
| SVS_EXPERIMENTAL_CHECK_BOUNDS   | ON, **OFF**        | Enable bounds checking on some data structure |
|                                 |                    | accesses. Can be helpful for debugging        |
|                                 |                    | out-of-bounds accesses.                       |
+---------------------------------+--------------------+-----------------------------------------------+
| SVS_EXPERIMENTAL_CLANG_TIDY     | ON, **OFF**        | Enable the clang-tidy static analyzer on the  |
|                                 |                    | utility binaries.                             |
|                                 |                    |                                               |
|                                 |                    | Requires ``SVS_BUILD_BINARIES=ON`` to be      |
|                                 |                    | effective.                                    |
+---------------------------------+--------------------+-----------------------------------------------+


.. _build_cibuildwheel:

(Advanced) CIBuildWheel and Microarchitecture Compatibility
===========================================================

This library uses C++ 20 and many AVX-512 hardware features to achieve performance.
However, we still want to support older CPUs and reasonably old Linux distributions that may have some GLIBC limitations.
To that end, we also support building the Python library using `cibuildwheel <https://cibuildwheel.readthedocs.io/en/stable/>`_ and enabling multiple microarchitecture backends.

To generate a wheel using your current version of Python you will need to cibuildwheel installed as well as `docker <https://www.docker.com/>`_.
Once those are installed, simply navigate to the root directory of the source and run

.. code-block:: sh

    cibuildwheel --only $(python tools/pybuild.py) bindings/python

The resulting Python wheel will be generated into the "wheelhouse" directly and can be installed from there.

If you wish to build wheels for all supported versions of Python, use the following:

.. code-block:: sh

    cibuildwheel bindings/python

Details on multi-arch support
-----------------------------

The cibuildwheel environment sets the ``PYSVS_MULTIARCH`` environment variable before triggering the build of the library.
The file ``bindings/python/setup.py`` file observes this variable and passes a list of micro-architectures to the CMake build system.
CMake will then compiler a version of the backend shared library for each given micro-architecture using that micro-architecture name as a suffix.
At run-time, the Python library will detect the CPU it is currently running on and attempt to load the most compatible shared libary.
See :ref:`this section <python_backend>` for details on backend inspection and selection.

(Advanced) Building the Documentation
=====================================

Library documentation is generated using `doxygen <https://www.doxygen.nl/>`_ to generate documentation for C++ code and `sphinx <https://www.sphinx-doc.org/en/master/>`_ to generate Python documentation and assemble the final website.

Prerequisites
-------------

The following prerequisites are required:

* Python documentation dependencies. These can be installed using

.. code-block:: sh

   pip install -U -r docs/requirements.txt

* Doxygen version 1.9.2 or higher (for C++ 20 support).
  Precompiled binaries are available `at this link <https://www.doxygen.nl/download.html>`_.

* The pysvs :ref:`Python module <building_python_library>` built and installed.

Building
--------

Run the following series of commands to set-up and build the documentation.

.. code-block:: sh

    mkdir build_doc && cd build_doc
    cmake .. -DSVS_BUILD_DOCS=YES -DDoxygen_ROOT="path/to/doxygen/bin"
    make

Alternatively, if pysvs has been installed in a non-standard directory, the final command will be

.. code-block:: sh

   PYTHONPATH="path/to/pysvs/dir" make
