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

cmake_minimum_required(VERSION 3.21)
project(graph_metrics_ci LANGUAGES CXX)

# Use the same calculator and parameters with both library revisions, including
# a base revision that predates the calculator itself.
set(SVS_BUILD_BINARIES OFF CACHE BOOL "" FORCE)
set(SVS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SVS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SVS_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(SVS_BUILD_BENCHMARK_TEST_GENERATORS OFF CACHE BOOL "" FORCE)
add_subdirectory("${SVS_SOURCE_DIR}" svs)

get_filename_component(METRICS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# These dependencies are private to this standalone project. Normal SVS utility
# builds must not discover or download them.
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        graph_metrics_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(graph_metrics_json)
endif()

add_executable(
    graph_metrics "${METRICS_SOURCE_DIR}/utils/graph_metrics.cpp"
)
target_link_libraries(
    graph_metrics PRIVATE
    svs::svs svs_compile_options svs_x86_options_base
    fmt::fmt nlohmann_json::nlohmann_json OpenSSL::Crypto
)
