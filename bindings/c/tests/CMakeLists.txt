# Copyright 2026 Intel Corporation
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

# Check if Catch2 is available
find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found, fetching from GitHub...")
    include(FetchContent)

    # Do wide printing for the console logger for Catch2
    set(CATCH_CONFIG_CONSOLE_WIDTH "100" CACHE STRING "" FORCE)
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(CATCH_CONFIG_ENABLE_BENCHMARKING OFF CACHE BOOL "" FORCE)
    set(CATCH_CONFIG_FAST_COMPILE OFF CACHE BOOL "" FORCE)
    set(CATCH_CONFIG_PREFIX_ALL ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.11.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Define test sources
set(C_API_TEST_SOURCES
    c_api_error.cpp
    c_api_algorithm.cpp
    c_api_storage.cpp
    c_api_search_params.cpp
    c_api_index_builder.cpp
    c_api_index.cpp
    c_api_dynamic_index.cpp
)

# Create test executable
add_executable(svs_c_api_tests ${C_API_TEST_SOURCES})

# Link with C API library and Catch2
target_link_libraries(svs_c_api_tests PRIVATE
    svs_c_api
    Catch2::Catch2WithMain
)

# Set C++ standard
target_compile_features(svs_c_api_tests PRIVATE cxx_std_17)

# Include directories
target_include_directories(svs_c_api_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Add test to CTest
include(CTest)

# Add Catch2 CMake module path
if(NOT Catch2_FOUND)
    # Catch2 was fetched, use its source directory
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
else()
    # Catch2 was found via find_package, use its module directory
    list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
endif()

include(Catch)
catch_discover_tests(svs_c_api_tests)

# Add a custom target to run tests
add_custom_target(run_c_api_tests
    COMMAND svs_c_api_tests
    DEPENDS svs_c_api_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running C API tests..."
)
