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
##### Checks the dispatch surface against the symbols that were actually built.
#####
##### Run in script mode:
#####
#####   cmake -DSVS_PROBE_OBJECT=<link_probe.cpp.o> \
#####         -DSVS_ARCHIVE=<libsvs_x86_objects.a> \
#####         -DSVS_NM=<nm> \
#####         -P tests/multi-arch/dispatch-checks/check-dispatch-linkage.cmake
#####
##### The probe object names every kernel the surface declares and nothing else
##### (see tests/multi-arch/x86/link_probe.cpp), so the kernels it *references*
##### are exactly the kernels the archive must *define* -- and it must define no
##### others. Both directions are checked, plus that the probe defines none of
##### its own.
#####

include("${CMAKE_CURRENT_LIST_DIR}/lib.cmake")

svs_require(SVS_PROBE_OBJECT SVS_ARCHIVE SVS_NM)
svs_require_files(SVS_PROBE_OBJECT SVS_ARCHIVE)

# Distance kernels, and nothing else in the archive. Mangled names are used
# throughout: demangled ones carry `[clone .isra.0]` suffixes that differ between
# a local instantiation and an explicit one.
set(svs_kernel_regex "_ZN3svs8distance.*Impl")

svs_nm_symbols(probe_defines "${SVS_PROBE_OBJECT}" "${svs_kernel_regex}" NM_ARGS --defined-only)
svs_nm_symbols(probe_references "${SVS_PROBE_OBJECT}" "${svs_kernel_regex}" NM_ARGS --undefined-only)
svs_nm_symbols(archive_defines "${SVS_ARCHIVE}" "${svs_kernel_regex}" NM_ARGS --defined-only)

list(LENGTH probe_defines n_probe_defines)
list(LENGTH probe_references n_probe_references)
list(LENGTH archive_defines n_archive_defines)

set(errors 0)

# A probe that names nothing is not a passing probe. This is what an LTO build
# looks like here, since the object holds IR rather than symbols.
if(n_probe_references EQUAL 0 AND n_probe_defines EQUAL 0)
    message("${SVS_PROBE_OBJECT} names no distance kernels at all.")
    message("Nothing can be concluded from it. If this is a link-time-optimized")
    message("build, nm cannot see the symbols and this check does not apply.")
    message(FATAL_ERROR "dispatch linkage check found no symbols to check")
endif()

# The probe compiles at -march=x86-64 and guarantees nothing about the host, so a
# kernel it defines itself is a kernel some other consumer would also define
# itself -- at whatever -march that consumer happens to use.
if(NOT n_probe_defines EQUAL 0)
    message("${n_probe_defines} kernels are instantiated by the probe itself.")
    message("Each is missing its `extern template` declaration, so every consumer")
    message("of the headers instantiates it locally, from the generic primary")
    message("template, at the consumer's own -march. Declare them: the extern")
    message("blocks in the distance headers must cover the whole surface.")
    message("    Instantiated locally:")
    svs_report_first(probe_defines 10)
    math(EXPR errors "${errors} + 1")
endif()

# Declared but never instantiated. The link should already have failed, so this
# only fires when the object is inspected without being linked.
set(missing ${probe_references})
if(archive_defines)
    list(REMOVE_ITEM missing ${archive_defines})
endif()
list(LENGTH missing n_missing)
if(NOT n_missing EQUAL 0)
    message("${n_missing} kernels are declared but never instantiated.")
    message("They are declared `extern template` in the distance headers but no")
    message("translation unit defines them, so linking against the library fails.")
    message("    Undefined:")
    svs_report_first(missing 10)
    math(EXPR errors "${errors} + 1")
endif()

# Instantiated but unreachable through the surface: dead weight in the archive.
set(unreachable ${archive_defines})
if(probe_references)
    list(REMOVE_ITEM unreachable ${probe_references})
endif()
list(LENGTH unreachable n_unreachable)
if(NOT n_unreachable EQUAL 0)
    message("${n_unreachable} kernels are instantiated but are not part of the")
    message("dispatch surface. Nothing declares them, so no consumer reaches")
    message("them; they only add to the size of the library.")
    message("    Unreachable:")
    svs_report_first(unreachable 10)
    math(EXPR errors "${errors} + 1")
endif()

if(NOT errors EQUAL 0)
    message(FATAL_ERROR "dispatch linkage check failed")
endif()

message(
    "dispatch linkage: ${n_archive_defines} kernels declared, instantiated and "
    "reachable; none instantiated by the consumer"
)
