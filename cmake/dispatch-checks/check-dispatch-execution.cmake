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
##### Checks which ISA level a call through the entry points actually enters.
#####
##### Run in script mode:
#####
#####   cmake -DSVS_PROBE=<entry_probe> -DSVS_NM=<nm> -DSVS_GDB=<gdb> \
#####         -P cmake/dispatch-checks/check-dispatch-execution.cmake
#####
##### Everything else about the surface is a property of the symbol table, which a
##### specialization can satisfy while never running: one that disappears behind an
##### `#if` still links and still counts. This breaks on every level's kernel for one
##### extent and type pair, and reports the one the host's run reaches.
#####
##### Only the level this host satisfies is checked. The weaker levels are checked by
##### hosts that satisfy only those, which is what the CI matrix is for.
#####

foreach(required SVS_PROBE SVS_NM SVS_GDB)
    if(NOT ${required})
        message(FATAL_ERROR "${required} is not set.")
    endif()
endforeach()
if(NOT EXISTS "${SVS_PROBE}")
    message(FATAL_ERROR "SVS_PROBE does not exist: ${SVS_PROBE}")
endif()

# The probe owns the runtime predicates, so it -- not this script -- decides which
# level the entry points are obliged to choose here.
execute_process(
    COMMAND "${SVS_PROBE}" --report
    OUTPUT_VARIABLE report
    ERROR_VARIABLE err
    RESULT_VARIABLE status
)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "${SVS_PROBE} --report failed: ${err}")
endif()
if(NOT report MATCHES "expect-level ([0-9]+)")
    message(FATAL_ERROR "${SVS_PROBE} --report printed no expected level:\n${report}")
endif()
set(expected_digit "${CMAKE_MATCH_1}")
if(NOT report MATCHES "probe-extent ([0-9]+)")
    message(FATAL_ERROR "${SVS_PROBE} --report printed no probe extent:\n${report}")
endif()
set(extent "${CMAKE_MATCH_1}")

if(expected_digit EQUAL 0)
    message(
        "dispatch execution: this host satisfies no ISA level in the surface, so the "
        "entry points can only reach AVX_AVAILABILITY::NONE; nothing to check"
    )
    return()
endif()

#####
##### The candidate symbols
#####

# The type pair the probe calls, Itanium-mangled. It must be one with a kernel at
# every level in the surface, or the entry points legitimately route lower.
set(svs_probe_pair "aa")

# L2 at that pair for the extent the probe reports: one symbol per level, including
# any AVX_AVAILABILITY::NONE fallback the consumer instantiated itself.
execute_process(
    COMMAND "${SVS_NM}" --defined-only "${SVS_PROBE}"
    OUTPUT_VARIABLE raw
    ERROR_VARIABLE err
    RESULT_VARIABLE status
)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "${SVS_NM} failed on ${SVS_PROBE}: ${err}")
endif()

set(candidates)
string(REPLACE "\n" ";" lines "${raw}")
foreach(line IN LISTS lines)
    # The mangled name is the last whitespace-separated field.
    string(REGEX MATCH "[^ \t]+$" symbol "${line}")
    if(NOT symbol MATCHES "^_ZN3svs8distance6L2ImplILm${extent}E${svs_probe_pair}.*7computeE")
        continue()
    endif()
    # Skip GCC's `.isra` clones: gdb reads a dot in a linespec as a file name.
    if(symbol MATCHES "\\.")
        continue()
    endif()
    list(APPEND candidates "${symbol}")
endforeach()
list(REMOVE_DUPLICATES candidates)

list(LENGTH candidates n_candidates)
if(n_candidates EQUAL 0)
    message(FATAL_ERROR
        "No L2 kernel symbol for extent ${extent} at the mangled type pair "
        "'${svs_probe_pair}' is defined in ${SVS_PROBE}. Either the entry points "
        "inlined every kernel, in which case the `extern template` declarations are "
        "not in effect, or the probe no longer calls that extent and pair."
    )
endif()

#####
##### Run under the debugger
#####

set(gdb_args -q -batch)
foreach(symbol IN LISTS candidates)
    list(APPEND gdb_args -ex "break ${symbol}")
endforeach()
list(APPEND gdb_args -ex "run --report" -ex "info symbol $pc")

execute_process(
    COMMAND "${SVS_GDB}" ${gdb_args} "${SVS_PROBE}"
    OUTPUT_VARIABLE gdb_out
    ERROR_VARIABLE gdb_err
    RESULT_VARIABLE status
)
if(NOT status EQUAL 0)
    message("${gdb_out}")
    message(FATAL_ERROR "${SVS_GDB} failed on ${SVS_PROBE}: ${gdb_err}")
endif()

# gdb demangles for `info symbol`, so the level appears as the enum's underlying
# value: `(svs::distance::AVX_AVAILABILITY)2`.
if(NOT gdb_out MATCHES "AVX_AVAILABILITY\\)([0-9]+)")
    message("${gdb_out}")
    message(FATAL_ERROR
        "The run never entered any of the ${n_candidates} kernels breakpointed for "
        "extent ${extent}. A call through the entry points reached none of them, so "
        "the dispatch is not routing to the surface."
    )
endif()
set(actual_digit "${CMAKE_MATCH_1}")

if(NOT actual_digit EQUAL expected_digit)
    message("The entry points routed a call to AVX_AVAILABILITY level ${actual_digit},")
    message("but this host satisfies level ${expected_digit}. Either the runtime")
    message("predicates in the entry points disagree with the ones the probe uses, or")
    message("the enumerator order in the surface no longer matches the dispatch order.")
    message(FATAL_ERROR "dispatch execution check failed")
endif()

message(
    "dispatch execution: a call to L2 at extent ${extent} entered "
    "AVX_AVAILABILITY level ${actual_digit}, the highest this host satisfies"
)
