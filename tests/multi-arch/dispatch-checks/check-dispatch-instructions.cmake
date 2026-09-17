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
##### Checks that an ISA level's object file stays inside its instruction budget.
#####
##### Run in script mode:
#####
#####   cmake -DSVS_OBJECT=<avx2.cpp.o> \
#####         -DSVS_LEVEL=AVX2 -DSVS_ARCH=haswell \
#####         -DSVS_OBJDUMP=<objdump> \
#####         [-DSVS_NO_AVX512=<bool>] \
#####         -P tests/multi-arch/dispatch-checks/check-dispatch-instructions.cmake
#####
##### A level promises the host satisfies its runtime predicate and nothing more,
##### so an instruction the predicate does not guarantee is an illegal-instruction
##### fault on a host the dispatcher considers supported.
#####
##### SVS_NO_AVX512 mirrors the build's own option of that name (cmake/options.cmake):
##### -mno-avx512f beats a level's later -march= on GCC, so an AVX-512-bearing level
##### then gets the generic scalar fallback instead of its wide kernels, and the
##### budget below has to flip from requiring AVX-512 to forbidding it.
#####

# Without it, CMP0007 is unset in script mode and the empty "forbids nothing"
# field of a budget row below is silently dropped rather than read as empty.
cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_LIST_DIR}/lib.cmake")

svs_require(SVS_OBJECT SVS_LEVEL SVS_ARCH SVS_OBJDUMP)
svs_require_files(SVS_OBJECT)

# Optional: absent callers (and any caller predating this option) get the
# default build's budget, matching cmake/options.cmake's own OFF default.
if(NOT DEFINED SVS_NO_AVX512)
    set(SVS_NO_AVX512 OFF)
endif()

# What each instruction class looks like in AT&T disassembly. Register classes are
# matched with their `%` sigil so that a mangled name can never look like one.
set(svs_class_ymm "%ymm")
set(svs_class_zmm "%zmm")
set(svs_class_mask "%k[1-7]")
set(svs_class_vnni "vpdpwssd|vpdpbusd|vpdpwssds|vpdpbusds")

# "<-march>|<required classes>|<forbidden classes>", one row per instruction budget
# any ISA level declares. A budget with no row here is a hard error: an
# unrecognized -march silently permitting everything is how a level ends up
# emitting instructions its runtime predicate does not guarantee.
set(svs_budget_table
    "x86-64||ymm zmm mask vnni"
    "haswell|ymm|zmm mask vnni"
    "skylake-avx512|zmm|vnni"
    "cascadelake|zmm vnni|"
)

# Only the arches whose default budget above requires AVX-512: -mno-avx512f
# does not touch haswell, so AVX2 keeps its normal row and has no entry here.
set(svs_noavx512_budget_table
    "skylake-avx512||zmm mask vnni"
    "cascadelake||zmm mask vnni"
)

set(svs_budget_found FALSE)
set(svs_noavx512_active FALSE)
if(SVS_NO_AVX512)
    foreach(row IN LISTS svs_noavx512_budget_table)
        string(REPLACE "|" ";" fields "${row}")
        list(GET fields 0 arch)
        if(arch STREQUAL SVS_ARCH)
            list(GET fields 1 svs_required)
            list(GET fields 2 svs_forbidden)
            set(svs_budget_found TRUE)
            set(svs_noavx512_active TRUE)
            break()
        endif()
    endforeach()
endif()
if(NOT svs_budget_found)
    foreach(row IN LISTS svs_budget_table)
        string(REPLACE "|" ";" fields "${row}")
        list(GET fields 0 arch)
        if(arch STREQUAL SVS_ARCH)
            list(GET fields 1 svs_required)
            list(GET fields 2 svs_forbidden)
            set(svs_budget_found TRUE)
            break()
        endif()
    endforeach()
endif()
if(NOT svs_budget_found)
    message(FATAL_ERROR
        "No instruction budget is known for -march=${SVS_ARCH} (ISA level "
        "${SVS_LEVEL}). Add a row to svs_budget_table in this file stating what "
        "that budget requires and forbids."
    )
endif()
string(REPLACE " " ";" svs_required "${svs_required}")
string(REPLACE " " ";" svs_forbidden "${svs_forbidden}")

execute_process(
    COMMAND "${SVS_OBJDUMP}" -d --no-show-raw-insn "${SVS_OBJECT}"
    OUTPUT_VARIABLE svs_disassembly
    ERROR_VARIABLE err
    RESULT_VARIABLE status
)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "${SVS_OBJDUMP} failed on ${SVS_OBJECT}: ${err}")
endif()

# Drop the `<symbol>:` heading lines: they are the only place a mangled name
# appears, and a mangled name must not be mistaken for an instruction.
string(REGEX REPLACE "\n[0-9a-f]+ <[^\n]*>:" "\n" svs_disassembly "${svs_disassembly}")

function(svs_count_class out_var class)
    if(NOT DEFINED svs_class_${class})
        message(FATAL_ERROR "Unknown instruction class '${class}' in the budget table.")
    endif()
    string(REGEX MATCHALL "${svs_class_${class}}" hits "${svs_disassembly}")
    list(LENGTH hits count)
    set(${out_var} ${count} PARENT_SCOPE)
endfunction()

set(errors 0)
set(summary)

foreach(class IN LISTS svs_required)
    svs_count_class(count "${class}")
    if(count EQUAL 0)
        message("${SVS_OBJECT} contains no ${class} instructions.")
        message("ISA level ${SVS_LEVEL} is compiled at -march=${SVS_ARCH}, which is")
        message("chosen precisely so the kernels use them. An object without any is")
        message("a level built at the wrong budget, or one whose kernels fell back")
        message("to the generic scalar template.")
        math(EXPR errors "${errors} + 1")
    else()
        list(APPEND summary "${count} ${class}")
    endif()
endforeach()

foreach(class IN LISTS svs_forbidden)
    svs_count_class(count "${class}")
    if(NOT count EQUAL 0)
        message("${SVS_OBJECT} contains ${count} ${class} instructions.")
        if(svs_noavx512_active)
            message("SVS_NO_AVX512 is set, so -mno-avx512f is supposed to keep AVX-512")
            message("out of every level regardless of -march=${SVS_ARCH}. A hit here means")
            message("that flag lost to something -- most likely flag order or a per-TU")
            message("override -- and this object still faults on a non-AVX-512 host.")
        else()
            message("ISA level ${SVS_LEVEL} guarantees only what its runtime predicate")
            message("tests, so a host the dispatcher routes here faults on them. Either")
            message("lower -march=${SVS_ARCH} in cmake/dispatch-surface.cmake, or give")
            message("these kernels their own level with a predicate that covers them.")
        endif()
        math(EXPR errors "${errors} + 1")
    endif()
endforeach()

if(NOT errors EQUAL 0)
    message(FATAL_ERROR "dispatch instruction check failed for level ${SVS_LEVEL}")
endif()

if(svs_required STREQUAL "")
    set(summary_display "nothing required")
else()
    string(REPLACE ";" ", " summary_display "${summary}")
endif()
if(svs_forbidden STREQUAL "")
    set(forbidden_display "nothing forbidden")
else()
    string(REPLACE ";" ", " forbidden_display "${svs_forbidden}")
    set(forbidden_display "no ${forbidden_display}")
endif()
if(svs_noavx512_active)
    message(
        "dispatch instructions: level ${SVS_LEVEL} at -march=${SVS_ARCH} with "
        "SVS_NO_AVX512 set requires no wide instructions, and confirms ${forbidden_display}"
    )
else()
    message(
        "dispatch instructions: level ${SVS_LEVEL} at -march=${SVS_ARCH} has "
        "${summary_display}, and ${forbidden_display}"
    )
endif()
