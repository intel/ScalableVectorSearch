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
    GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(pybind11)

# Include the SVS library directly.
add_subdirectory("../.." "${CMAKE_CURRENT_BINARY_DIR}/svs")

# C++ files makind up the python bindings.
set(CPP_FILES
    src/allocator.cpp
    src/dynamic_vamana.cpp
    src/dynamic_flat.cpp
    src/core.cpp
    src/flat.cpp
    src/python_bindings.cpp
    src/vamana.cpp
    src/vamana_common.cpp
    src/svs_mkl.cpp
)

# ivf
if (SVS_EXPERIMENTAL_ENABLE_IVF)
    list(APPEND CPP_FILES
        src/ivf.cpp
    )
endif()

set(LIB_NAME "_svs")
pybind11_add_module(${LIB_NAME} MODULE ${CPP_FILES})
target_link_libraries(${LIB_NAME} PRIVATE pybind11::module)
target_link_libraries(${LIB_NAME} PUBLIC svs::svs)
# Dependency "fmt::fmt" obtained from "svs"
target_link_libraries(${LIB_NAME} PRIVATE svs::compile_options fmt::fmt svs_x86_options_base)
target_include_directories(
    ${LIB_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)
if(DEFINED SKBUILD)
    install(TARGETS ${LIB_NAME} DESTINATION .)
    set_target_properties(
        ${LIB_NAME}
        PROPERTIES
        INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}"
    )
endif()

if(DEFINED SKBUILD)

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
