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

#####
##### The dispatch surface link probe.
#####
##### Not a Catch2 test: what is being checked is a property of the object file,
##### so the probe has to be its own translation unit, compiled at a known
##### instruction budget and inspected from outside.
#####

# The object library exists so that the probe's single object file can be named
# with $<TARGET_OBJECTS:>, which is what the linkage check reads.
add_library(dispatch_surface_probe_objects OBJECT x86/link_probe.cpp)
target_link_libraries(
    dispatch_surface_probe_objects
    PRIVATE svs::svs svs::compile_options svs::x86_options_base
)

# svs::x86_options_base is -march=x86-64 -mtune=generic: the probe must know no
# more about the host than an arbitrary consumer of the headers does.
add_executable(dispatch_surface_probe)
target_link_libraries(
    dispatch_surface_probe
    PRIVATE
    dispatch_surface_probe_objects
    svs::svs
    svs::compile_options
    svs::x86_options_base
)

# Calls every kernel whose ISA level this host satisfies. Which kernels those are
# depends on the host, so this covers the whole surface only across the CI matrix.
add_test(NAME dispatch_surface_probe COMMAND dispatch_surface_probe)

if(DEFINED CMAKE_NM AND CMAKE_NM)
    set(svs_nm "${CMAKE_NM}")
else()
    find_program(svs_nm NAMES nm llvm-nm)
endif()

if(svs_nm)
    # Host-independent, unlike the run above: it reads the symbol table rather
    # than executing anything, so it sees every kernel in the surface everywhere.
    add_test(
        NAME dispatch_surface_linkage
        COMMAND
        "${CMAKE_COMMAND}"
        "-DSVS_PROBE_OBJECT=$<TARGET_OBJECTS:dispatch_surface_probe_objects>"
        "-DSVS_ARCHIVE=$<TARGET_FILE:svs_x86_objects>"
        "-DSVS_NM=${svs_nm}"
        # Not PROJECT_SOURCE_DIR: the downstream repository adds this directory
        # to its own project, where that points somewhere else entirely.
        -P "${CMAKE_CURRENT_LIST_DIR}/../../cmake/check-dispatch-linkage.cmake"
    )
else()
    message(STATUS "nm not found; skipping the dispatch surface linkage test")
endif()
