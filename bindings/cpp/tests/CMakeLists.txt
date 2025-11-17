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

set(CATCH_CONFIG_CONSOLE_WIDTH "100" CACHE STRING "" FORCE)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CATCH_CONFIG_ENABLE_BENCHMARKING OFF CACHE BOOL "" FORCE)
set(CATCH_CONFIG_PREFIX_ALL ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# Runtime tests
add_executable(svs_runtime_tests
    test_version_namespace.cpp
)

target_include_directories(svs_runtime_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(svs_runtime_tests PRIVATE
    svs_runtime
    Catch2::Catch2WithMain
)

target_compile_definitions(svs_runtime_tests PRIVATE
    FAISS_SVS_RUNTIME_VERSION=v0
)

# Register with CTest
include(CTest)
include(Catch)
catch_discover_tests(svs_runtime_tests)
