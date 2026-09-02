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

#####
##### Tests of the x86 dispatch surface.
#####
##### Not Catch2 tests: what is being checked is a property of an object file or of
##### a process, so each probe has to be its own translation unit, compiled at a
##### known instruction budget and inspected from outside.
#####
##### Not PROJECT_SOURCE_DIR anywhere below: the downstream repository adds this
##### directory to its own project, where that points somewhere else entirely.
#####

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/dispatch-levels.cmake")

set(svs_dispatch_cmake_dir "${CMAKE_CURRENT_LIST_DIR}/dispatch-checks")

# The object libraries exist so that each probe's single object file can be named
# with $<TARGET_OBJECTS:>, which is what the symbol-table checks read.
#
# svs::x86_options_base is -march=x86-64 -mtune=generic: a probe must know no more
# about the host than an arbitrary consumer of the headers does.
function(svs_add_dispatch_probe name source)
    add_library(${name}_objects OBJECT ${source})
    target_link_libraries(
        ${name}_objects PRIVATE svs::svs svs::compile_options svs::x86_options_base
    )
    add_executable(${name})
    target_link_libraries(
        ${name}
        PRIVATE ${name}_objects svs::svs svs::compile_options svs::x86_options_base
    )
endfunction()

# Names every kernel the surface declares and nothing else, so a declared-but-
# uninstantiated kernel is an undefined symbol here.
svs_add_dispatch_probe(dispatch_surface_probe x86/link_probe.cpp)

# Reaches the kernels through the entry points instead, so it also sees whether the
# entry points route to the whole surface.
svs_add_dispatch_probe(dispatch_entry_probe x86/entry_probe.cpp)

# Calls every kernel whose ISA level this host satisfies. Which kernels those are
# depends on the host, so this covers the whole surface only across the CI matrix.
add_test(NAME dispatch_surface_probe COMMAND dispatch_surface_probe)
add_test(NAME dispatch_entry_probe COMMAND dispatch_entry_probe)

if(DEFINED CMAKE_NM AND CMAKE_NM)
    set(svs_nm "${CMAKE_NM}")
else()
    find_program(svs_nm NAMES nm llvm-nm)
endif()
if(DEFINED CMAKE_OBJDUMP AND CMAKE_OBJDUMP)
    set(svs_objdump "${CMAKE_OBJDUMP}")
else()
    find_program(svs_objdump NAMES objdump llvm-objdump)
endif()
find_program(svs_gdb NAMES gdb)

if(svs_nm)
    # Host-independent, unlike the runs above: it reads the symbol table rather
    # than executing anything, so it sees every kernel in the surface everywhere.
    add_test(
        NAME dispatch_surface_linkage
        COMMAND
        "${CMAKE_COMMAND}"
        "-DSVS_PROBE_OBJECT=$<TARGET_OBJECTS:dispatch_surface_probe_objects>"
        "-DSVS_ARCHIVE=$<TARGET_FILE:svs_x86_objects>"
        "-DSVS_NM=${svs_nm}"
        -P "${svs_dispatch_cmake_dir}/check-dispatch-linkage.cmake"
    )

    # The same symbols, counted from the declaration instead of from a probe built
    # out of the same generated header -- the one check a generator bug cannot pass
    # by being wrong consistently.
    add_test(
        NAME dispatch_surface_declaration
        COMMAND
        "${CMAKE_COMMAND}"
        "-DSVS_MANIFEST=${CMAKE_BINARY_DIR}/dispatch_surface.manifest.cmake"
        "-DSVS_TYPE_PAIR_HEADER=${CMAKE_CURRENT_LIST_DIR}/../../include/svs/multi-arch/x86/preprocessor.h"
        "-DSVS_ENUM_HEADER=${CMAKE_CURRENT_LIST_DIR}/../../include/svs/core/distance/distance_core.h"
        "-DSVS_ARCHIVE=$<TARGET_FILE:svs_x86_objects>"
        "-DSVS_CONSUMER_OBJECT=$<TARGET_OBJECTS:dispatch_entry_probe_objects>"
        "-DSVS_NM=${svs_nm}"
        -P "${svs_dispatch_cmake_dir}/check-dispatch-declaration.cmake"
    )
else()
    message(STATUS "nm not found; skipping the dispatch surface symbol-table tests")
endif()

if(svs_objdump)
    # One test per ISA level: a level whose object file carries instructions its
    # runtime predicate does not guarantee faults on a host the dispatcher routes
    # to it, which no symbol-table check can see.
    foreach(tu_spec IN LISTS SVS_DISPATCH_TU_SPECS)
        svs_parse_tu_spec("${tu_spec}" svs_src svs_level svs_arch svs_infix)
        add_test(
            NAME dispatch_instructions_${svs_infix}
            COMMAND
            "${CMAKE_COMMAND}"
            "-DSVS_OBJECT=$<TARGET_OBJECTS:${svs_infix}_obj>"
            "-DSVS_LEVEL=${svs_level}"
            "-DSVS_ARCH=${svs_arch}"
            "-DSVS_OBJDUMP=${svs_objdump}"
            -P "${svs_dispatch_cmake_dir}/check-dispatch-instructions.cmake"
        )
    endforeach()
else()
    message(STATUS "objdump not found; skipping the dispatch surface instruction tests")
endif()

if(svs_nm AND svs_gdb)
    # The only check that observes a kernel run rather than exist: a specialization
    # lost behind an `#if` still links and still counts.
    add_test(
        NAME dispatch_surface_execution
        COMMAND
        "${CMAKE_COMMAND}"
        "-DSVS_PROBE=$<TARGET_FILE:dispatch_entry_probe>"
        "-DSVS_NM=${svs_nm}"
        "-DSVS_GDB=${svs_gdb}"
        -P "${svs_dispatch_cmake_dir}/check-dispatch-execution.cmake"
    )
else()
    message(STATUS "gdb not found; skipping the dispatch surface execution test")
endif()
