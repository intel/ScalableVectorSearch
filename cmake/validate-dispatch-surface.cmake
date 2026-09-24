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
##### Reads and checks a dispatch-surface declaration.
#####
##### Inputs:
#####   SVS_DISPATCH_SURFACE_FILE -- the declaration to read
#####   SVS_X86_SRC_DIR           -- where per-level translation units live
#####
##### Outputs:
#####   SVS_SUPPORTED_DIMS, SVS_ISA_LEVELS -- verbatim from the declaration
#####   SVS_DIM_LIST                       -- extents, with svs::Dynamic appended
#####   SVS_DIM_COUNT                      -- length of SVS_DIM_LIST
#####   SVS_FIXED_DIM_COUNT                -- length of SVS_SUPPORTED_DIMS
#####

# Scoped this file via PUSH/POP so it does not affect the includer
cmake_policy(PUSH)
cmake_policy(SET CMP0007 NEW)
cmake_policy(SET CMP0057 NEW)

include("${CMAKE_CURRENT_LIST_DIR}/dispatch-levels.cmake")

if(NOT SVS_DISPATCH_SURFACE_FILE)
    message(FATAL_ERROR "SVS_DISPATCH_SURFACE_FILE is not set.")
endif()
if(NOT EXISTS "${SVS_DISPATCH_SURFACE_FILE}")
    message(FATAL_ERROR
        "SVS_DISPATCH_SURFACE_FILE does not exist: ${SVS_DISPATCH_SURFACE_FILE}"
    )
endif()
if(NOT SVS_X86_SRC_DIR)
    message(FATAL_ERROR "SVS_X86_SRC_DIR is not set.")
endif()

# Set the variables and include the surface file to populate them
set(SVS_SUPPORTED_DIMS)
set(SVS_ISA_LEVELS)
include("${SVS_DISPATCH_SURFACE_FILE}")


# Sanity checks DIMS
if(NOT SVS_SUPPORTED_DIMS)
    message(FATAL_ERROR
        "SVS_SUPPORTED_DIMS is empty in ${SVS_DISPATCH_SURFACE_FILE}. At least "
        "one fixed extent is required."
    )
endif()

# Only numerical values
foreach(dim IN LISTS SVS_SUPPORTED_DIMS)
    if(NOT dim MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR
            "SVS_SUPPORTED_DIMS contains '${dim}', which is not a positive "
            "integer. svs::Dynamic is required and is appended automatically, "
            "so it must not be listed."
        )
    endif()
endforeach()

# Remove duplicates
set(svs_dims_deduped ${SVS_SUPPORTED_DIMS})
list(REMOVE_DUPLICATES svs_dims_deduped)
list(LENGTH SVS_SUPPORTED_DIMS SVS_FIXED_DIM_COUNT)
list(LENGTH svs_dims_deduped svs_dims_unique)
if(NOT SVS_FIXED_DIM_COUNT EQUAL svs_dims_unique)
    message(FATAL_ERROR
        "SVS_SUPPORTED_DIMS contains duplicate extents. Every extent must "
        "appear exactly once."
    )
endif()

# Append svs::Dynamic
set(SVS_DIM_LIST ${SVS_SUPPORTED_DIMS} "svs::Dynamic")
list(LENGTH SVS_DIM_LIST SVS_DIM_COUNT)


# Sanity checks ISA
if(NOT SVS_ISA_LEVELS)
    message(FATAL_ERROR "SVS_ISA_LEVELS is empty in ${SVS_DISPATCH_SURFACE_FILE}.")
endif()

set(svs_distance_core_header
    "${CMAKE_CURRENT_LIST_DIR}/../include/svs/core/distance/distance_core.h"
)
if(NOT EXISTS "${svs_distance_core_header}")
    message(FATAL_ERROR
        "Cannot find ${svs_distance_core_header} to read AVX_AVAILABILITY from."
    )
endif()
file(READ "${svs_distance_core_header}" svs_distance_core_text)
if(NOT svs_distance_core_text MATCHES "enum class AVX_AVAILABILITY[ \t\r\n]*{([^}]*)}")
    message(FATAL_ERROR
        "Cannot find 'enum class AVX_AVAILABILITY { ... }' in "
        "${svs_distance_core_header}."
    )
endif()
string(REPLACE "," ";" svs_legal_levels "${CMAKE_MATCH_1}")
list(TRANSFORM svs_legal_levels STRIP)
# NONE means "no level is present" and is only there as fallback
list(REMOVE_ITEM svs_legal_levels "NONE")

set(svs_seen_levels)
set(svs_seen_infixes)
set(svs_seen_archs)
foreach(level_spec IN LISTS SVS_ISA_LEVELS)
    svs_parse_isa_level("${level_spec}" level arch infix)
    foreach(field level arch infix)
        if(NOT ${field})
            message(FATAL_ERROR
                "Malformed SVS_ISA_LEVELS entry '${level_spec}': ${field} is empty."
            )
        endif()
    endforeach()
    if(level STREQUAL "NONE")
        message(FATAL_ERROR
            "ISA level 'NONE' in SVS_ISA_LEVELS is not declarable: it means no "
            "level is present, so it has no translation unit and no object "
            "library for a row to name."
        )
    endif()
    if(NOT level IN_LIST svs_legal_levels)
        string(REPLACE ";" ", " svs_legal_levels_display "${svs_legal_levels}")
        message(FATAL_ERROR
            "Unknown ISA level '${level}' in SVS_ISA_LEVELS: not an enumerator "
            "of svs::distance::AVX_AVAILABILITY in ${svs_distance_core_header} "
            "(legal levels: ${svs_legal_levels_display})."
        )
    endif()
    if(level IN_LIST svs_seen_levels)
        message(FATAL_ERROR "Duplicate ISA level '${level}' in SVS_ISA_LEVELS.")
    endif()
    if(infix IN_LIST svs_seen_infixes)
        message(FATAL_ERROR
            "Duplicate TU infix '${infix}' in SVS_ISA_LEVELS; infixes name "
            "generated files and must be unique."
        )
    endif()
    if(arch IN_LIST svs_seen_archs)
        message(FATAL_ERROR
            "Duplicate -march '${arch}' in SVS_ISA_LEVELS; each level's -march "
            "is its instruction budget, so two levels sharing one budget compile "
            "the weaker level with instructions its runtime predicate does not "
            "guarantee, and hosts routed to it fault."
        )
    endif()
    if(NOT EXISTS "${SVS_X86_SRC_DIR}/${infix}.cpp")
        message(FATAL_ERROR
            "ISA level '${level}' has no translation unit: expected "
            "${SVS_X86_SRC_DIR}/${infix}.cpp. Adding a level to SVS_ISA_LEVELS "
            "requires creating that file."
        )
    endif()
    list(APPEND svs_seen_levels ${level})
    list(APPEND svs_seen_infixes ${infix})
    list(APPEND svs_seen_archs ${arch})
endforeach()

# distance_core.h #errors on x86_64 unless both are present, so a surface
# omitting either is not a smaller build: it cannot compile.
foreach(svs_mandatory_level AVX2 AVX512)
    if(NOT svs_mandatory_level IN_LIST svs_seen_levels)
        message(FATAL_ERROR
            "SVS_ISA_LEVELS omits mandatory level '${svs_mandatory_level}'; "
            "distance_core.h requires both AVX2 and AVX512 to be present."
        )
    endif()
endforeach()

cmake_policy(POP)
