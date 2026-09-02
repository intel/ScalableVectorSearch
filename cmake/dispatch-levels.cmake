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

include_guard(GLOBAL)

# Parses an ISA level spec from SVS_ISA_LEVELS and returns the three fields.
# The spec format is <level>|<arch>|<infix>.
function(svs_parse_isa_level spec out_level out_arch out_infix)
  string(REPLACE "|" ";" fields "${spec}")
  list(LENGTH fields n)
  if(NOT n EQUAL 3)
    message(FATAL_ERROR "Malformed ISA level '${spec}': expected <level>|<arch>|<infix>.")
  endif()
  list(GET fields 0 v0)
  list(GET fields 1 v1)
  list(GET fields 2 v2)
  set(${out_level} "${v0}" PARENT_SCOPE)
  set(${out_arch}  "${v1}" PARENT_SCOPE)
  set(${out_infix} "${v2}" PARENT_SCOPE)
endfunction()

# Parses a translation unit spec from SVS_DISPATCH_TU_SPECS and returns the four fields.
# The spec format is <src>|<level>|<arch>|<infix>.
function(svs_parse_tu_spec spec out_src out_level out_arch out_infix)
  string(REPLACE "|" ";" fields "${spec}")
  list(LENGTH fields n)
  if(NOT n EQUAL 4)
    message(FATAL_ERROR "Malformed TU spec '${spec}': expected <src>|<level>|<arch>|<infix>.")
  endif()
  list(GET fields 0 v0)
  list(GET fields 1 v1)
  list(GET fields 2 v2)
  list(GET fields 3 v3)
  set(${out_src}    "${v0}" PARENT_SCOPE)
  set(${out_level}  "${v1}" PARENT_SCOPE)
  set(${out_arch}   "${v2}" PARENT_SCOPE)
  set(${out_infix}  "${v3}" PARENT_SCOPE)
endfunction()
