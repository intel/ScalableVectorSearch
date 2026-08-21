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
##### Derives the dispatch surface declared in cmake/dispatch-surface.cmake.
#####
##### Produces:
#####   include/svs/core/distance/dispatch_surface.h  (source tree, committed)
#####   SVS_DISPATCH_TU_SPECS -- "<src>|<level>|<arch>|<infix>", one per ISA level
#####

include_guard(GLOBAL)

set(SVS_DEFAULT_DISPATCH_SURFACE_FILE "${CMAKE_CURRENT_LIST_DIR}/dispatch-surface.cmake")
set(SVS_DISPATCH_SURFACE_FILE "${SVS_DEFAULT_DISPATCH_SURFACE_FILE}"
    CACHE FILEPATH
    "Declaration of the ahead-of-time distance-kernel dispatch surface"
)

# Reads the declaration and rejects it if it is malformed. Also runnable on its
# own -- see .github/scripts/check_dispatch_surface.sh.
set(SVS_X86_SRC_DIR "${PROJECT_SOURCE_DIR}/include/svs/multi-arch/x86")
include("${CMAKE_CURRENT_LIST_DIR}/validate-dispatch-surface.cmake")

file(REAL_PATH "${SVS_DISPATCH_SURFACE_FILE}" svs_surface_real)
file(REAL_PATH "${SVS_DEFAULT_DISPATCH_SURFACE_FILE}" svs_default_surface_real)
if(svs_surface_real STREQUAL svs_default_surface_real)
    set(svs_surface_is_default TRUE)
else()
    set(svs_surface_is_default FALSE)
    message(STATUS
        "Dispatch surface overridden by ${SVS_DISPATCH_SURFACE_FILE}; the "
        "committed header will not be refreshed"
    )
endif()

# Re-run configure when the declaration changes, so the generated header and the
# translation units cannot go stale.
set_property(
    DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${SVS_DISPATCH_SURFACE_FILE}"
)

#####
##### Build the macro bodies
#####

# Line continuations are emitted with a trailing backslash; the generated macros
# are one logical line each.
set(SVS_GEN_DIM_COUNT ${SVS_DIM_COUNT})

set(SVS_GEN_DIM_LOOP "\\\n")
foreach(dim IN LISTS SVS_DIM_LIST)
    string(APPEND SVS_GEN_DIM_LOOP "    M(${dim}) \\\n")
endforeach()
string(APPEND SVS_GEN_DIM_LOOP "    /* end */")

set(SVS_GEN_TARGET_LOOP "\\\n")
set(SVS_DISPATCH_TU_SPECS)
foreach(level_spec IN LISTS SVS_ISA_LEVELS)
    string(REPLACE "|" ";" level_fields "${level_spec}")
    list(GET level_fields 0 level)
    list(GET level_fields 1 arch)
    list(GET level_fields 2 infix)

    foreach(dim IN LISTS SVS_DIM_LIST)
        string(APPEND SVS_GEN_TARGET_LOOP "    M(${dim}, ${level}) \\\n")
    endforeach()

    # One translation unit per level, named after the level's infix. The file
    # itself is short -- it loops over the generated extent list -- but it is
    # committed rather than generated, because the downstream repository compiles
    # these sources by path. Its existence was checked during validation.
    list(APPEND SVS_DISPATCH_TU_SPECS
        "${SVS_X86_SRC_DIR}/${infix}.cpp|${level}|${arch}|${infix}"
    )
endforeach()
string(APPEND SVS_GEN_TARGET_LOOP "    /* end */")

#####
##### Emit the header
#####

# The build always compiles against the build-tree copy, and it is placed ahead
# of the source include directory so that it wins.
set(SVS_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
set(SVS_GENERATED_DISPATCH_HEADER
    "${SVS_GENERATED_INCLUDE_DIR}/svs/core/distance/dispatch_surface.h"
)
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/templates/dispatch_surface.h.in"
    "${SVS_GENERATED_DISPATCH_HEADER}"
    @ONLY
)
target_include_directories(
    ${SVS_LIB} BEFORE INTERFACE $<BUILD_INTERFACE:${SVS_GENERATED_INCLUDE_DIR}>
)

# Refresh the committed copy too, but only for the default declaration: the
# committed header exists so that a bare `-I include` compile works without
# CMake, and a one-off build with an overridden surface must not rewrite it.
# configure_file only touches the file when the content changes, so this neither
# dirties the tree nor forces rebuilds.
if(svs_surface_is_default)
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/templates/dispatch_surface.h.in"
        "${PROJECT_SOURCE_DIR}/include/svs/core/distance/dispatch_surface.h"
        @ONLY
    )
endif()

list(LENGTH SVS_ISA_LEVELS svs_level_count)
message(STATUS
    "Dispatch surface: ${SVS_DIM_COUNT} extents (${SVS_FIXED_DIM_COUNT} fixed + "
    "svs::Dynamic) x ${svs_level_count} ISA levels"
)
