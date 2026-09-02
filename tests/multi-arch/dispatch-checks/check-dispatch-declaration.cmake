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
##### Checks the built kernels against the declaration, not against the generated
##### header.
#####
##### Run in script mode:
#####
#####   cmake -DSVS_MANIFEST=build/dispatch_surface.manifest.cmake \
#####         -DSVS_TYPE_PAIR_HEADER=include/svs/multi-arch/x86/preprocessor.h \
#####         -DSVS_ENUM_HEADER=include/svs/core/distance/distance_core.h \
#####         -DSVS_ARCHIVE=<libsvs_x86_objects.a> \
#####         -DSVS_CONSUMER_OBJECT=<entry_probe.cpp.o> \
#####         -DSVS_NM=<nm> \
#####         -P tests/multi-arch/dispatch-checks/check-dispatch-declaration.cmake
#####
##### tests/multi-arch/dispatch-checks/check-dispatch-linkage.cmake compares the archive
##### against a probe, and both are generated from the same header: a generator
##### that dropped an extent would drop it from both and still agree. The
##### expectation here is derived from the three hand-written sources instead --
##### the extent list and levels from the manifest, the type pairs, and the enumerator
##### order -- so the generated header is not consulted at all and a generator bug
##### has nowhere to hide.
#####

# Without it, CMP0057 is unset in script mode and the `IN_LIST` tests below are
# rejected as unknown arguments.
cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_LIST_DIR}/lib.cmake")

svs_require(SVS_MANIFEST SVS_TYPE_PAIR_HEADER SVS_ENUM_HEADER SVS_ARCHIVE SVS_CONSUMER_OBJECT SVS_NM)
svs_require_files(SVS_MANIFEST SVS_TYPE_PAIR_HEADER SVS_ENUM_HEADER SVS_ARCHIVE SVS_CONSUMER_OBJECT)

#####
##### Ground truth 1: the extents and the ISA levels
#####

include("${SVS_MANIFEST}")

if(NOT SVS_MANIFEST_FIXED_EXTENTS OR NOT SVS_MANIFEST_LEVELS)
    message(FATAL_ERROR
        "${SVS_MANIFEST} is not a dispatch-surface manifest; it does not declare "
        "extents or ISA levels."
    )
endif()

# svs::Dynamic mangles as its numeric value: the extent is a size_t template
# argument, and Dynamic is SIZE_MAX.
set(svs_expected_extents ${SVS_MANIFEST_FIXED_EXTENTS} "18446744073709551615")
set(svs_expected_levels ${SVS_MANIFEST_LEVELS})

#####
##### Ground truth 2: the enumerator order, which fixes the mangled digit
#####

file(READ "${SVS_ENUM_HEADER}" svs_enum_text)
if(NOT svs_enum_text MATCHES "enum class AVX_AVAILABILITY[ \t\r\n]*{([^}]*)}")
    message(FATAL_ERROR
        "No `enum class AVX_AVAILABILITY` found in ${SVS_ENUM_HEADER}. Its "
        "enumerator order is what maps an ISA level onto the digit in a mangled "
        "name, so it cannot be inferred."
    )
endif()
string(REPLACE "," ";" svs_enumerators "${CMAKE_MATCH_1}")
set(svs_enum_index 0)
foreach(enumerator IN LISTS svs_enumerators)
    string(STRIP "${enumerator}" enumerator)
    if(enumerator)
        set(svs_digit_of_${enumerator} ${svs_enum_index})
        math(EXPR svs_enum_index "${svs_enum_index} + 1")
    endif()
endforeach()

set(svs_expected_digits)
foreach(level IN LISTS svs_expected_levels)
    if(NOT DEFINED svs_digit_of_${level})
        message(FATAL_ERROR
            "ISA level '${level}' is declared in ${SVS_SURFACE_FILE} but is not an "
            "AVX_AVAILABILITY enumerator."
        )
    endif()
    list(APPEND svs_expected_digits ${svs_digit_of_${level}})
endforeach()

#####
##### Ground truth 3: how many type pairs each level has kernels for
#####

file(READ "${SVS_TYPE_PAIR_HEADER}" svs_pairs_text)
# Fold line continuations away so each #define is one line to match against.
string(REGEX REPLACE "\\\\[ \t]*\r?\n" " " svs_pairs_text "${svs_pairs_text}")

# Returns the number of M(...) invocations in a type-pair list macro, following one
# level of aliasing -- `#define SVS_TYPE_PAIRS_AVX2 SVS_FOR_EACH_TYPE_PAIR`.
function(svs_type_pair_count out_var macro_name)
    if(NOT svs_pairs_text MATCHES "#define ${macro_name}(\\([^)]*\\))?[ \t]+([^\n]*)")
        message(FATAL_ERROR
            "No `#define ${macro_name}` in ${SVS_TYPE_PAIR_HEADER}. Every ISA level "
            "needs one; without it the level instantiates nothing."
        )
    endif()
    set(body "${CMAKE_MATCH_2}")
    string(STRIP "${body}" body)
    if(body MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        svs_type_pair_count(count "${body}")
        set(${out_var} ${count} PARENT_SCOPE)
        return()
    endif()
    string(REGEX MATCHALL "M\\(" hits "${body}")
    list(LENGTH hits count)
    if(count EQUAL 0)
        message(FATAL_ERROR "${macro_name} in ${SVS_TYPE_PAIR_HEADER} invokes M zero times.")
    endif()
    set(${out_var} ${count} PARENT_SCOPE)
endfunction()

#####
##### Ground truth 4: the distances
#####

# Adding a distance means adding its Impl class here; otherwise its kernels are
# counted by nothing and a level could ship without them.
set(svs_distance_classes "L2Impl" "IPImpl" "CosineSimilarityImpl")

set(svs_mangled_classes)
foreach(class IN LISTS svs_distance_classes)
    string(LENGTH "${class}" len)
    list(APPEND svs_mangled_classes "${len}${class}")
endforeach()

#####
##### The expectation
#####

# Keyed <mangled class>_<extent>_<digit>; the count is the number of type pairs,
# one symbol each.
set(svs_expected_keys)
set(svs_expected_total 0)
foreach(level IN LISTS svs_expected_levels)
    svs_type_pair_count(pairs "SVS_TYPE_PAIRS_${level}")
    set(svs_pairs_of_${level} ${pairs})
    foreach(class IN LISTS svs_mangled_classes)
        foreach(extent IN LISTS svs_expected_extents)
            set(key "${class}_${extent}_${svs_digit_of_${level}}")
            list(APPEND svs_expected_keys "${key}")
            set(svs_want_${key} ${pairs})
            math(EXPR svs_expected_total "${svs_expected_total} + ${pairs}")
        endforeach()
    endforeach()
endforeach()

#####
##### What was built
#####

# Mangled names throughout: demangled ones carry `[clone .isra.0]` suffixes that
# differ between a local instantiation and an explicit one.
set(svs_kernel_regex "^_ZN3svs8distance([0-9]+[A-Za-z0-9_]+)ILm([0-9]+)E.*AVX_AVAILABILITYE([0-9]+)E")
set(svs_distance_regex "_ZN3svs8distance.*AVX_AVAILABILITY")

# Sets <prefix>_KEYS, <prefix>_FOREIGN and <prefix>_HAVE_<key> in the caller.
function(svs_tally prefix symbols)
    set(keys)
    set(foreign)
    foreach(symbol IN LISTS symbols)
        if(NOT symbol MATCHES "${svs_kernel_regex}")
            continue()
        endif()
        set(key "${CMAKE_MATCH_1}_${CMAKE_MATCH_2}_${CMAKE_MATCH_3}")
        if(NOT key IN_LIST svs_expected_keys)
            list(APPEND foreign "${symbol}")
            continue()
        endif()
        if(NOT key IN_LIST keys)
            list(APPEND keys "${key}")
            set(${prefix}_HAVE_${key} 0)
        endif()
        math(EXPR next "${${prefix}_HAVE_${key}} + 1")
        set(${prefix}_HAVE_${key} ${next})
        set(${prefix}_HAVE_${key} ${next} PARENT_SCOPE)
    endforeach()
    set(${prefix}_KEYS "${keys}" PARENT_SCOPE)
    set(${prefix}_FOREIGN "${foreign}" PARENT_SCOPE)
endfunction()

svs_nm_symbols(archive_defines "${SVS_ARCHIVE}" "${svs_distance_regex}" NM_ARGS --defined-only)
svs_nm_symbols(consumer_defines "${SVS_CONSUMER_OBJECT}" "${svs_distance_regex}" NM_ARGS --defined-only)
svs_nm_symbols(consumer_references "${SVS_CONSUMER_OBJECT}" "${svs_distance_regex}" NM_ARGS --undefined-only)

svs_tally(ARCHIVE "${archive_defines}")
svs_tally(CONSUMER "${consumer_references}")

#####
##### Compare
#####

set(errors 0)

# Sets <prefix>_ERRORS in the caller. `what` names the thing being compared, for
# the failure message.
function(svs_compare prefix what)
    set(wrong)
    foreach(key IN LISTS svs_expected_keys)
        set(have 0)
        if(DEFINED ${prefix}_HAVE_${key})
            set(have ${${prefix}_HAVE_${key}})
        endif()
        if(NOT have EQUAL ${svs_want_${key}})
            list(APPEND wrong "${key}: expected ${svs_want_${key}}, got ${have}")
        endif()
    endforeach()
    list(LENGTH wrong n_wrong)
    set(${prefix}_ERRORS 0 PARENT_SCOPE)
    if(NOT n_wrong EQUAL 0)
        message("${n_wrong} of ${what} disagree with the manifest.")
        message("Each entry is <distance>_<extent>_<ISA level>. A count of 0 means")
        message("the declaration asks for kernels that were never built; a count")
        message("below the number of type pairs means the level is missing some.")
        message("    Mismatched:")
        svs_report_first(wrong 10)
        set(${prefix}_ERRORS 1 PARENT_SCOPE)
    endif()
endfunction()

svs_compare(ARCHIVE "the archive's kernel groups")
svs_compare(CONSUMER "the kernel groups a consumer reaches through the entry points")
math(EXPR errors "${ARCHIVE_ERRORS} + ${CONSUMER_ERRORS}")

if(ARCHIVE_FOREIGN)
    list(LENGTH ARCHIVE_FOREIGN n)
    message("${n} kernels in the archive are outside the declared surface.")
    message("Nothing in the manifest asks for them, so no consumer reaches")
    message("them and they only add to the size of the library.")
    message("    Outside the surface:")
    svs_report_first(ARCHIVE_FOREIGN 10)
    math(EXPR errors "${errors} + 1")
endif()

# The whole point of the extern declarations. A kernel the consumer defines at a
# declared level came from the generic primary template at the consumer's own
# -march instead of from the library.
set(consumer_leaks)
foreach(symbol IN LISTS consumer_defines)
    if(symbol MATCHES "${svs_kernel_regex}")
        if(CMAKE_MATCH_3 IN_LIST svs_expected_digits)
            list(APPEND consumer_leaks "${symbol}")
        endif()
    endif()
endforeach()
list(LENGTH consumer_leaks n_leaks)
if(NOT n_leaks EQUAL 0)
    message("${n_leaks} kernels at a declared ISA level are instantiated by the consumer.")
    message("Each is missing its `extern template` declaration, so every consumer of")
    message("the headers builds it locally, from the generic scalar template, at")
    message("whatever -march that consumer uses.")
    message("    Instantiated locally:")
    svs_report_first(consumer_leaks 10)
    math(EXPR errors "${errors} + 1")
endif()

if(NOT errors EQUAL 0)
    message(FATAL_ERROR "dispatch declaration check failed")
endif()

list(LENGTH svs_expected_levels n_levels)
list(LENGTH svs_expected_extents n_extents)
list(LENGTH svs_distance_classes n_distances)
set(pair_display)
foreach(level IN LISTS svs_expected_levels)
    list(APPEND pair_display "${level}: ${svs_pairs_of_${level}}")
endforeach()
string(REPLACE ";" ", " pair_display "${pair_display}")
message(
    "dispatch declaration: ${svs_expected_total} kernels required by "
    "the manifest (${n_levels} levels x ${n_extents} extents x "
    "${n_distances} distances x type pairs per level {${pair_display}}); the "
    "archive defines exactly those and the entry points reach all of them"
)
