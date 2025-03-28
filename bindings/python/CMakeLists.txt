# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.21) # keep in-sync with `pyproject.toml`
project(svs)

# Install pybind11 manually for version control.
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.10.1
)
FetchContent_MakeAvailable(pybind11)

# Try to find the Python executable.
#
# If it's given as part of the Cmake arguments given by "scikit build", then use that.
# Otherwise, fall back to using plain old "python".
# If *THAT* doesn't work, give up.
if(DEFINED PYTHON_EXECUTABLE)
    set(SVS_PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
    set(SVS_PYTHON_EXECUTABLE "python")
endif()

# The micro architectures to compile for.
if(NOT DEFINED SVS_MICROARCHS)
    set(SVS_MICROARCHS native)
endif()

# Include the SVS library directly.
add_subdirectory("../.." "${CMAKE_CURRENT_BINARY_DIR}/svs")

# Run the python script to get optimization flags for the desired back-ends.
#
# FLAGS_SCRIPT - Path to the Python script that will take the compiler, compiler version,
#   and list of desired microarchitectures and generate optimization flags for each
#   microarchitecture.
#
# FLAGS_TEXT_FILE - List of optimization flags for each architecture.
#   Expected format:
#       -march=arch1,-mtune=arch1
#       -march=arch2,-mtune=arch2
#       ...
#       -march=archN,-mtune=archN
#
#   The number of lines should be equal to the number of microarchitectures.
#   NOTE: The entries within each line are separated by a comma on purpose to allow CMake
#   to read the whole file as a List and then use string replacement on the commas to turn
#   each line into a list in its own right.
#
# TEMP_JSON - JSON Manifest file describing the generated binaries. This is meant to be
#   included in the Python package to allow the Python code to reason about the packaged
#   libraries and select the correct one for loading.
#
set(FLAGS_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/microarch.py")
set(FLAGS_TEXT_FILE "${CMAKE_CURRENT_BINARY_DIR}/optimization_flags.txt")
set(FLAGS_MANIFEST_JSON "${CMAKE_CURRENT_BINARY_DIR}/flags_manifest.json")

execute_process(
    COMMAND
        ${SVS_PYTHON_EXECUTABLE}
        ${FLAGS_SCRIPT}
        ${FLAGS_TEXT_FILE}
        ${FLAGS_MANIFEST_JSON}
        --compiler ${CMAKE_CXX_COMPILER_ID}
        --compiler-version ${CMAKE_CXX_COMPILER_VERSION}
        --microarchitectures ${SVS_MICROARCHS}
    COMMAND_ERROR_IS_FATAL ANY
)

file(STRINGS "${FLAGS_TEXT_FILE}" OPTIMIZATION_FLAGS)
message("Flags: ${OPTIMIZATION_FLAGS}")
list(LENGTH OPTIMIZATION_FLAGS OPT_FLAGS_LENGTH)
message("Length of flags: ${OPT_FLAGS_LENGTH}")

# C++ files makind up the python bindings.
set(CPP_FILES
    src/allocator.cpp
    src/dynamic_vamana.cpp
    src/core.cpp
    src/flat.cpp
    src/python_bindings.cpp
    src/vamana.cpp
    src/vamana_common.cpp
    src/svs_mkl.cpp
)

# Generate a shared library for each target microarchitecture.
foreach(MICRO OPT_FLAGS IN ZIP_LISTS SVS_MICROARCHS OPTIMIZATION_FLAGS)
    set(LIB_NAME "_svs_${MICRO}")

    pybind11_add_module(${LIB_NAME} MODULE ${CPP_FILES})
    target_link_libraries(${LIB_NAME} PUBLIC svs::svs)
    # Dependency "fmt::fmt" obtained from "svs"
    target_link_libraries(${LIB_NAME} PRIVATE svs::compile_options fmt::fmt)

    string(REPLACE "," ";" OPT_FLAGS ${OPT_FLAGS})
    message("OPT Flags: ${OPT_FLAGS}")
    target_compile_options(${LIB_NAME} PRIVATE ${OPT_FLAGS})

    # Header files.
    target_include_directories(
        ${LIB_NAME}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    )

    # Comunicate to the C++ library the desired name of the library
    target_compile_options(${LIB_NAME} PRIVATE "-DSVS_MODULE_NAME=${LIB_NAME}")

    # If scikit build is running the compilation process,
    if(DEFINED SKBUILD)
        install(TARGETS ${LIB_NAME} DESTINATION .)

        # The extension module may need to load build or included libraries when loaded.

        # Placing build depedencies in the package and using relative RPATHs that
        # don't point outside of the package means that the built package is
        # relocatable. This allows for safe binary redistribution.
        set_target_properties(
            ${LIB_NAME}
            PROPERTIES
                INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}"
        )
    endif()
endforeach()

if(DEFINED SKBUILD)
    # Install the manifest JSON file.
    # This is kind of a hack to avoid the needing to explicitly move JSON file into the
    # source folder of the python library.
    install(FILES ${FLAGS_MANIFEST_JSON} DESTINATION .)

    # Install header files.
    install(
        DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include/svs"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.h"
    )

    # Installation for serialized objects upgrading.
    # Upgrading SVS objects from global serialization version ``v0.0.1`` to ``v0.0.2``
    # requires the application of a ``__schema__`` keyword to each canonically serialized
    # object.
    #
    # To facilitate this, an input schema file is provided which can be matched against
    # serialized SVS objects to automatically apply apply this field.
    #
    # Upgrading is done by ``svs.upgrader.upgrade``, which uses a default path to this
    # matching schema file.
    #
    # The logic here is what installs that file from the SVS repository into the Python
    # wheel.
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/../../data/serialization.toml" DESTINATION .)
endif()

