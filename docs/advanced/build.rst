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

.. _build_cibuildwheel:

CIBuildWheel and Microarchitecture Compatibility
------------------------------------------------

We support building SVS as a Python wheel in a Docker container using `cibuildwheel <https://cibuildwheel.pypa.io>`_.

You need to install cibuildwheel and `docker <https://docs.docker.com/engine/install/>`_.
Once those are installed, navigate to the root directory of the source and run

.. code-block:: sh

    # clean any previous build artifacts
    rm -rf bindings/python/_skbuild
    # build container image
    cd docker/x86_64/manylinux2014
    ./build.sh
    # build Python wheel
    cd -
    cibuildwheel --only $(python3 tools/pybuild.py) bindings/python

If you wish to build wheels for all supported versions of Python, not just the version you are using, leave out ``--only``:

.. code-block:: sh

    cibuildwheel bindings/python

The resulting Python wheel will be generated in the ``wheelhouse`` directory and can be installed from there.

.. code-block:: sh

    pip install wheelhouse/svs*.whl


SVS uses C++20 and many AVX-512 hardware features to achieve performance.
However, we still want to support older CPUs and reasonably old Linux distributions that may have some GLIBC limitations.
The wheel built by cibuildwheel has maximum compatibility.


Customizing Wheels
------------------

The following commands will perform the **native** build, with optimizations specific to the CPU you are using:

.. code-block:: sh

   cd bindings/python
   python3 setup.py bdist_wheel -- [cmake arguments] -- -j$(nproc)
   pip install ./dist/svs*.whl

If the default build options are acceptable, the CMake arguments may be left empty. If you want optimizations for a different CPU than the one you are using for the build, you can specify a microarchitecture using ``SVS_MICROARCHS``:

.. code-block:: sh

   python3 setup.py bdist_wheel -- -DSVS_MICROARCHS=sapphirerapids -- -j$(nproc)


Debug Builds
^^^^^^^^^^^^

Every once in a while, it is beneficial to use either a debug build of the Python wheel or at least build the wheel with debug information.
This can be difficult through the pip only approach as it likes to transitively build dependencies.
Instead, it is recommended to go through the two stage approach:

.. code-block:: sh

   cd bindings/python
   python3 setup.py bdist_wheel --build-type=Debug -- -- -j$(nproc)

Building Notes
--------------

Occasionally, the Python build process will fail seemingly without cause.
This usually occurs when previously there has been a mix of different compilers, compile-time variables, and build strategies.
If this happens, try removing ``bindings/python/_skbuild`` and ``bindings/python/dist`` and going again.

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
        GIT_TAG main
    )

    FetchContent_MakeAvailable(svs)

    # Link with the library
    target_link_libraries([my_target] PRIVATE|PUBLIC|INTERFACE svs::svs)

Installing Locally
^^^^^^^^^^^^^^^^^^

The C++ library can also be installed locally using CMake's installation logic.

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

    +---------------------+--------------------+-----------------------------------------------+
    | CMake Option        | Supported Values   | Description                                   |
    |                     | (defaults in bold) |                                               |
    +=====================+====================+===============================================+
    | SVS_BUILD_BINARIES  | ON, **OFF**        | Build utility binaries.                       |
    +---------------------+--------------------+-----------------------------------------------+
    | SVS_BUILD_TESTS     | ON, **OFF**        | Build the unit test suite.                    |
    +---------------------+--------------------+-----------------------------------------------+
    | SVS_BUILD_BENCHMARK | ON, **OFF**        | Build the regression benchmarking suite.      |
    +---------------------+--------------------+-----------------------------------------------+
    | SVS_BUILD_DOCS      | ON, **OFF**        | Build the documentation.                      |
    +---------------------+--------------------+-----------------------------------------------+
    | SVS_BUILD_EXAMPLES  | ON, **OFF**        | Build the documentation examples.             |
    +---------------------+--------------------+-----------------------------------------------+
    | SVS_NO_AVX512       | ON, **OFF**        | Disable AVX512 instructions from being used.  |
    |                     |                    | Helpful when running Valgrind as that tool    |
    |                     |                    | does not support AVX512.                      |
    |                     |                    |                                               |
    |                     |                    | This option is not required when compiling on |
    |                     |                    | non-AVX512 systems.                           |
    +---------------------+--------------------+-----------------------------------------------+

Occasionally, more control over the compiled binaries and executables is desired (to aid binary size and compilation time).
These fine-grained variables are defined below.

.. table:: Advanced SVS CMake Variables
    :width: 100

    +-------------------------------------+--------------------+-----------------------------------------------+
    | CMake Option                        | Supported Values   | Description                                   |
    |                                     | (defaults in bold) |                                               |
    +=====================================+====================+===============================================+
    | SVS_INITIALIZE_LOGGER               | **ON**, OFF        | Enable the default SVS logger using the       |
    |                                     |                    | environment variable SVS_LOG_LEVEL and        |
    |                                     |                    | SVS_LOG_SINK (if they are defined).           |
    |                                     |                    |                                               |
    |                                     |                    | If disabled, the default SVS logger will be   |
    |                                     |                    | a null logger propagating no logging mesasges.|
    +-------------------------------------+--------------------+-----------------------------------------------+
    | SVS_FORCE_INTEGRATION_TESTS         | ON, **OFF**        | By default, integration tests will not be     |
    |                                     |                    | compiled when building in tests in debug mode |
    |                                     |                    | because debug builds of SVS are extremely     |
    |                                     |                    | slow.                                         |
    |                                     |                    |                                               |
    |                                     |                    | Setting this variable equal forces inclusion  |
    |                                     |                    | of integration tests in the test binary.      |
    |                                     |                    |                                               |
    |                                     |                    | This variable has no effect if                |
    |                                     |                    | ``SVS_BUILD_TESTS == OFF``.                   |
    +-------------------------------------+--------------------+-----------------------------------------------+
    | SVS_BUILD_BENCHMARK_TEST_GENERATORS | ON, **OFF**        | Build the routines that generate              |
    |                                     |                    | :ref:`reference <testing>` results for        |
    |                                     |                    | integration tests.                            |
    |                                     |                    |                                               |
    |                                     |                    | This is left off be default to reduce compile |
    |                                     |                    | times for the benchmark suite.                |
    +-------------------------------------+--------------------+-----------------------------------------------+
    | SVS_EXPERIMENTAL_BUILD_CUSTOM_MKL   | ON, **OFF**        | If the included modules have MKL has a        |
    |                                     |                    | dependency, this option will create a custom  |
    |                                     |                    | MKL shared-library using only the symbols     |
    |                                     |                    | needed by SVS.                                |
    |                                     |                    |                                               |
    |                                     |                    | This allows for compiled SVS executables to   |
    |                                     |                    | be portable.                                  |
    +-------------------------------------+--------------------+-----------------------------------------------+

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
| SVS_EXPERIMENTAL_LEANVEC        | ON, **OFF**        | Enable LeanVec for vector dimension reduction |
|                                 |                    | Requires MKL library to implement SVD/GEMM    |
+---------------------------------+--------------------+-----------------------------------------------+


Details on multi-arch support
-----------------------------

The cibuildwheel environment sets the ``SVS_MULTIARCH`` environment variable before triggering the build of the library.
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

* The svs :ref:`Python module <building_python_library>` built and installed.

Building
--------

Run the following series of commands to set-up and build the documentation.

.. code-block:: sh

    mkdir build_doc && cd build_doc
    cmake .. -DSVS_BUILD_DOCS=YES -DDoxygen_ROOT="path/to/doxygen/bin"
    make

Alternatively, if svs has been installed in a non-standard directory, the final command will be

.. code-block:: sh

   PYTHONPATH="path/to/svs/dir" make

(Advanced) MKL as a Dependency
==============================

Upcoming SVS features need to use functionality provided by MKL.
SVS can link with MKL in a number of ways.

First, if MKL is not needed, then compiled SVS artifacts should not try to link with MKL.
Second, a system MKL can be used with the combination:

.. code-block:: sh

    -DSVS_EXPERIMENTAL_LEANVEC=YES
    -DSVS_EXPERIMENTAL_BUILD_CUSTOM_MKL=NO

Note that if this option is used, you *may* need to include appropriate environment variable
for SVS to find MKL at run time.

Finally, SVS can also build and link with a custom MKL shared library using the
`custom shared object builder <https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-0/using-the-custom-shared-object-builder.html>`_ .
To use this feature, provide the following variables to Cmake at configuration time:

.. code-block:: sh

    -DSVS_EXPERIMENTAL_LEANVEC=YES
    -DSVS_EXPERIMENTAL_BUILD_CUSTOM_MKL=YES

