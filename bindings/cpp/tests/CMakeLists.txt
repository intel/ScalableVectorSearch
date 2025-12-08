# Copyright 2025 Intel Corporation
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

include(FetchContent)

# Configure Catch2 for unit testing
# Do wide printing for the console logger for Catch2
# Reference: https://github.com/catchorg/Catch2/issues/1348
set(CATCH_CONFIG_CONSOLE_WIDTH "100" CACHE STRING "" FORCE)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CATCH_CONFIG_ENABLE_BENCHMARKING OFF CACHE BOOL "" FORCE)
set(CATCH_CONFIG_FAST_COMPILE OFF CACHE BOOL "" FORCE)

# If we don't configure the prefix version of the Catch2 macros, then we get compiler
# errors about repeated definitions of the macro `INFO`
set(CATCH_CONFIG_PREFIX_ALL ON CACHE BOOL "" FORCE)

# Download the Catch2 unit testing suite.
# Reference: https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
set(PRESET_CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD 20)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)

FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 11.2.0
)
FetchContent_MakeAvailable(fmt)

set(CMAKE_CXX_STANDARD ${PRESET_CMAKE_CXX_STANDARD})

# Add test executable
set(TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/runtime_test.cpp
)

add_executable(svs_runtime_test ${TEST_SOURCES})

# Link with the runtime library
target_link_libraries(svs_runtime_test PRIVATE
    svs_runtime
    Catch2::Catch2WithMain
    fmt::fmt
)

# Set C++ standard
target_compile_features(svs_runtime_test PRIVATE cxx_std_20)
set_target_properties(svs_runtime_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories for test utilities
target_include_directories(svs_runtime_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

# Enable testing with CTest
include(CTest)
enable_testing()

# Add the test to CTest
add_test(NAME svs_runtime_test COMMAND svs_runtime_test)

# Set test properties
set_tests_properties(svs_runtime_test PROPERTIES
    LABELS "runtime_bindings"
)

# Optional: Add Catch2's automatic test discovery
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(svs_runtime_test)
