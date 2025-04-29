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

cmake_minimum_required(VERSION 3.21)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Check if we are on Linux, AArch64, and using GCC.
# This needs to happen *before* project() so flags/definitions are set early.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND
   (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_CXX_COMPILER_ID) # Check ID or if not yet set
  )
    message(STATUS "ARM Linux GCC detected: Adding -march=armv8-a and defining EVE_SUPPORTS_NEON for EVE compatibility.")

    # 1. Ensure compiler targets NEON-capable arch
    set(ARM_GCC_ARCH_FLAG "-march=armv8-a")
    # Check if already set by environment/toolchain to avoid duplication if needed
    if(NOT CMAKE_CXX_FLAGS MATCHES "-march=")
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARM_GCC_ARCH_FLAG}" CACHE STRING "CXX Flags" FORCE)
    endif()
     if(NOT CMAKE_C_FLAGS MATCHES "-march=")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARM_GCC_ARCH_FLAG}" CACHE STRING "C Flags" FORCE)
    endif()

    # 2. Force EVE to recognize NEON support by defining its internal macro
    #    This should influence EVE's headers during its own configuration via FetchContent.
    #    Using add_compile_definitions ensures it's passed via -D to the compiler.
    add_compile_definitions(EVE_SUPPORTS_NEON)

endif()

project(svs
    LANGUAGES CXX
    # The version is tested in the files:
    # - /tests/svs/lib/version.cpp
    # - /bindings/python/tests/test_common.py
    # Manually keep in-sync with:
    # - /bindings/python/setup.py
    VERSION 0.0.7
)

set(SVS_LIB svs_devel)

# Version variables.
set(SVS_VERSION ${PROJECT_VERSION})
set(SVS_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(SVS_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(SVS_VERSION_PATCH ${PROJECT_VERSION_PATCH})

add_library(${SVS_LIB} INTERFACE)
set_target_properties(${SVS_LIB} PROPERTIES EXPORT_NAME svs)
add_library(svs::svs ALIAS ${SVS_LIB})

# Runtime include directories are established in the installation logic.
target_include_directories(
    ${SVS_LIB}
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

# We use C++ 20 features in our exposed headers.
# Anyone using us as a depdency and including our headers will need to be C++20 compatible.
#
# Keep this variable in-sync with the `SVS_CXX_STANDARD` given below.
# We need to manually set the standard for `spdlog`
target_compile_features(${SVS_LIB} INTERFACE cxx_std_20)
set(SVS_CXX_STANDARD 20)

# Populate with the version number macro.
target_compile_options(
    ${SVS_LIB}
    INTERFACE
    "-DSVS_VERSION_MAJOR=${SVS_VERSION_MAJOR}"
    "-DSVS_VERSION_MINOR=${SVS_VERSION_MINOR}"
    "-DSVS_VERSION_PATCH=${SVS_VERSION_PATCH}"
)

#####
##### Options and extra build steps
#####

include("cmake/options.cmake")

include("cmake/clang-tidy.cmake")
include("cmake/eve.cmake")
include("cmake/pthread.cmake")
include("cmake/numa.cmake")
include("cmake/robin-map.cmake")
include("cmake/fmt.cmake")
include("cmake/spdlog.cmake")
include("cmake/toml.cmake")

#####
##### Build Objects
#####

if(SVS_BUILD_BINARIES)
    add_subdirectory(utils)
endif()

if(SVS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(SVS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# The benchmark directory contains a sub-component that is used by both the benchmarking
# framework and the unit-tests.
#
# If only the unit tests are enabled, then the benchmark will be built as a minimal
# component to avoid excessive compilation time.
if(SVS_BUILD_BENCHMARK OR SVS_BUILD_TESTS OR SVS_BUILD_BENCHMARK_TEST_GENERATORS)
    add_subdirectory(benchmark)
endif()

#####
##### Install Logic
#####

include(GNUInstallDirs)

# Location of auxiliary generated cmake files to help consumers of this package.
set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/svs")

# Install headers and target information.
install(
    TARGETS svs_devel svs_compile_options svs_native_options
    EXPORT svs-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/svs"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
)
install(
    EXPORT svs-targets
    NAMESPACE "svs::"
    DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

#####
##### Config File
#####

set(VERSION_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/svsConfigVersion.cmake")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/svsConfig.cmake"
    INSTALL_DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

# Don't make compatibility guarentees until we reach a compatibility milestone.
write_basic_package_version_file(
    ${VERSION_CONFIG}
    VERSION ${SVS_VERSION}
    COMPATIBILITY ExactVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/svsConfig.cmake"
    "${VERSION_CONFIG}"
    DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

# Copy any "Find*" files that may be needed.
set(CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(SVS_CMAKE_FIND_FILES
    ${CMAKE_DIR}/FindNuma.cmake
)

install(FILES
    ${SVS_CMAKE_FIND_FILES}
    DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

