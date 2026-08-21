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

# These build two ways:
#
#   - as a subdirectory of the C API build, which bindings/c adds when
#     SVS_BUILD_EXAMPLES is on;
#   - as a standalone project against an *installed* C API package, which is how
#     CI runs them. The examples need nothing from the build tree, so testing
#     them against the package makes them consumers of the shipped headers and
#     exported target instead of only of freshly compiled objects.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.21)
    project(svs_c_api_examples LANGUAGES C)
    find_package(svs_c_api REQUIRED)
    set(EXAMPLE_LINK_TARGET svs::svs_c_api)
else()
    set(EXAMPLE_LINK_TARGET svs_c_api)
endif()

foreach(EXAMPLE_NAME simple save_load dynamic)
    set(EXAMPLE_TARGET c_api_${EXAMPLE_NAME})
    list(APPEND EXAMPLE_TARGETS ${EXAMPLE_TARGET})
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_NAME}.c)

    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${EXAMPLE_LINK_TARGET})
endforeach(EXAMPLE_NAME)

add_custom_target(c_api_examples ALL DEPENDS ${EXAMPLE_TARGETS})
