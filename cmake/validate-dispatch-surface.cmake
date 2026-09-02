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
##### This is deliberately free of build-system state so that it runs in script
##### mode as well as during configure:
#####
#####   cmake -DSVS_DISPATCH_SURFACE_FILE=<file> \
#####         -DSVS_X86_SRC_DIR=<dir> \
#####         -P cmake/validate-dispatch-surface.cmake
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

## In script mode there is no cmake_minimum_required, so policies default to OLD.
## Both of these are load-bearing here: CMP0007 keeps an empty `|`-field from
## vanishing when the entry is split, and CMP0057 enables `IN_LIST`.
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

# The declaration is plain CMake: it sets SVS_SUPPORTED_DIMS and SVS_ISA_LEVELS
# and does nothing else. Clear them first so that a declaration which forgets one
# is reported as empty rather than inheriting a value from the caller.
set(SVS_SUPPORTED_DIMS)
set(SVS_ISA_LEVELS)
include("${SVS_DISPATCH_SURFACE_FILE}")

#####
##### The extent list
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

# svs::Dynamic is mandatory: it is what serves every dimensionality without a
# fixed-extent kernel, and the library is incorrect without it.
set(SVS_DIM_LIST ${SVS_SUPPORTED_DIMS} "svs::Dynamic")
list(LENGTH SVS_DIM_LIST SVS_DIM_COUNT)

#####
##### The ISA levels
#####

if(NOT SVS_ISA_LEVELS)
    message(FATAL_ERROR "SVS_ISA_LEVELS is empty in ${SVS_DISPATCH_SURFACE_FILE}.")
endif()

set(svs_seen_levels)
set(svs_seen_infixes)
foreach(level_spec IN LISTS SVS_ISA_LEVELS)
    svs_parse_isa_level("${level_spec}" level arch infix)
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
    if(NOT EXISTS "${SVS_X86_SRC_DIR}/${infix}.cpp")
        message(FATAL_ERROR
            "ISA level '${level}' has no translation unit: expected "
            "${SVS_X86_SRC_DIR}/${infix}.cpp. Adding a level to SVS_ISA_LEVELS "
            "requires creating that file."
        )
    endif()
    list(APPEND svs_seen_levels ${level})
    list(APPEND svs_seen_infixes ${infix})
endforeach()

cmake_policy(POP)
