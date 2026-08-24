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
if(NOT EXISTS "${SVS_DISPATCH_SURFACE_FILE}")
    message(FATAL_ERROR
        "SVS_DISPATCH_SURFACE_FILE does not exist: ${SVS_DISPATCH_SURFACE_FILE}"
    )
endif()
include("${SVS_DISPATCH_SURFACE_FILE}")

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
##### Validate the extent list
#####

if(NOT SVS_SUPPORTED_DIMS)
    message(FATAL_ERROR
        "SVS_SUPPORTED_DIMS is empty in ${SVS_DISPATCH_SURFACE_FILE}. At least "
        "one fixed extent is required."
    )
endif()

foreach(dim IN LISTS SVS_SUPPORTED_DIMS)
    if(NOT dim MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR
            "SVS_SUPPORTED_DIMS contains '${dim}', which is not a positive "
            "integer. svs::Dynamic is required and is appended automatically, "
            "so it must not be listed."
        )
    endif()
endforeach()

set(svs_dims_sorted ${SVS_SUPPORTED_DIMS})
list(REMOVE_DUPLICATES svs_dims_sorted)
list(LENGTH SVS_SUPPORTED_DIMS svs_dims_given)
list(LENGTH svs_dims_sorted svs_dims_unique)
if(NOT svs_dims_given EQUAL svs_dims_unique)
    message(FATAL_ERROR
        "SVS_SUPPORTED_DIMS contains duplicate extents. Every extent must "
        "appear exactly once."
    )
endif()

# svs::Dynamic is mandatory: it is what serves every dimensionality without a
# fixed-extent kernel, and the library is incorrect without it.
set(svs_dim_list ${SVS_SUPPORTED_DIMS} "svs::Dynamic")
list(LENGTH svs_dim_list SVS_GEN_DIM_COUNT)

#####
##### Validate the ISA levels
#####

if(NOT SVS_ISA_LEVELS)
    message(FATAL_ERROR "SVS_ISA_LEVELS is empty in ${SVS_DISPATCH_SURFACE_FILE}.")
endif()

set(svs_seen_levels)
set(svs_seen_infixes)
foreach(level_spec IN LISTS SVS_ISA_LEVELS)
    string(REPLACE "|" ";" level_fields "${level_spec}")
    list(LENGTH level_fields nfields)
    if(NOT nfields EQUAL 3)
        message(FATAL_ERROR
            "Malformed SVS_ISA_LEVELS entry '${level_spec}': expected exactly "
            "three '|'-separated fields <enumerator>|<instruction budget>|<TU infix>."
        )
    endif()
    list(GET level_fields 0 level)
    list(GET level_fields 1 arch)
    list(GET level_fields 2 infix)
    foreach(field level arch infix)
        if(NOT ${field})
            message(FATAL_ERROR
                "Malformed SVS_ISA_LEVELS entry '${level_spec}': ${field} is empty."
            )
        endif()
    endforeach()
    if(level IN_LIST svs_seen_levels)
        message(FATAL_ERROR "Duplicate ISA level '${level}' in SVS_ISA_LEVELS.")
    endif()
    if(infix IN_LIST svs_seen_infixes)
        message(FATAL_ERROR
            "Duplicate TU infix '${infix}' in SVS_ISA_LEVELS; infixes name "
            "generated files and must be unique."
        )
    endif()
    list(APPEND svs_seen_levels ${level})
    list(APPEND svs_seen_infixes ${infix})
endforeach()

#####
##### Generate the header
#####

# Line continuations are emitted with a trailing backslash; the generated macros
# are one logical line each.
set(SVS_GEN_DIM_LOOP "\\\n")
foreach(dim IN LISTS svs_dim_list)
    string(APPEND SVS_GEN_DIM_LOOP "    M(${dim}) \\\n")
endforeach()
string(APPEND SVS_GEN_DIM_LOOP "    /* end */")

set(SVS_GEN_TARGET_LOOP "\\\n")
set(SVS_GEN_LEVEL_LOOP "\\\n")
set(SVS_DISPATCH_TU_SPECS)
set(svs_x86_src_dir "${PROJECT_SOURCE_DIR}/include/svs/multi-arch/x86")
foreach(level_spec IN LISTS SVS_ISA_LEVELS)
    string(REPLACE "|" ";" level_fields "${level_spec}")
    list(GET level_fields 0 level)
    list(GET level_fields 1 arch)
    list(GET level_fields 2 infix)

    string(APPEND SVS_GEN_LEVEL_LOOP "    M(${level}) \\\n")
    foreach(dim IN LISTS svs_dim_list)
        string(APPEND SVS_GEN_TARGET_LOOP "    M(${dim}, ${level}) \\\n")
    endforeach()

    # One translation unit per level, named after the level's infix. The file
    # itself is short -- it loops over the generated extent list -- but it is
    # committed rather than generated, because the private repository compiles
    # these sources by path.
    set(tu_src "${svs_x86_src_dir}/${infix}.cpp")
    if(NOT EXISTS "${tu_src}")
        message(FATAL_ERROR
            "ISA level '${level}' has no translation unit: expected ${tu_src}. "
            "Adding a level to SVS_ISA_LEVELS requires creating that file."
        )
    endif()
    list(APPEND SVS_DISPATCH_TU_SPECS "${tu_src}|${level}|${arch}|${infix}")
    list(APPEND svs_level_report
        "AVX_AVAILABILITY::${level} -march=${arch} ${infix}.cpp"
    )
endforeach()
string(APPEND SVS_GEN_TARGET_LOOP "    /* end */")
string(APPEND SVS_GEN_LEVEL_LOOP "    /* end */")

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

#####
##### Report the surface
#####

list(LENGTH SVS_ISA_LEVELS svs_level_count)
string(REPLACE ";" " " svs_dims_display "${SVS_SUPPORTED_DIMS}")
message(STATUS
    "Dispatch surface: ${SVS_GEN_DIM_COUNT} extents x ${svs_level_count} ISA levels"
)
message(STATUS "  extents: ${svs_dims_display} svs::Dynamic")
foreach(entry IN LISTS svs_level_report)
    message(STATUS "  level:   ${entry}")
endforeach()

# Every enumerator without a translation unit is still reachable -- the entry
# points fall back to it -- so its kernels are built by each consumer instead.
set(svs_enum_header "${PROJECT_SOURCE_DIR}/include/svs/core/distance/distance_core.h")
if(EXISTS "${svs_enum_header}")
    file(READ "${svs_enum_header}" svs_enum_text)
    if(svs_enum_text MATCHES "enum class AVX_AVAILABILITY[ \t\r\n]*{([^}]*)}")
        string(REPLACE "," ";" svs_enumerators "${CMAKE_MATCH_1}")
        set(svs_undeclared)
        foreach(enumerator IN LISTS svs_enumerators)
            string(STRIP "${enumerator}" enumerator)
            if(enumerator AND NOT enumerator IN_LIST svs_seen_levels)
                list(APPEND svs_undeclared "${enumerator}")
            endif()
        endforeach()
        if(svs_undeclared)
            string(REPLACE ";" ", " svs_undeclared_display "${svs_undeclared}")
            message(STATUS "  not in the surface: ${svs_undeclared_display}")
            message(STATUS
                "           dispatched to, but compiled by no translation unit, so "
                "every consumer instantiates those kernels itself, at its own -march"
            )
        endif()
    endif()
endif()
