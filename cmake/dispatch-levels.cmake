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

##
## This is a cmake helper to split a spec provided as `a|b|c`
## into three fields `[a, b, c]`
##

include_guard(GLOBAL)

# Scoped this file via PUSH/POP so it does not affect the includer
cmake_policy(PUSH)
cmake_policy(SET CMP0007 NEW)

# Splits `spec` on '|' into `n` fields
function(svs_split_fields spec n)
  string(REPLACE "|" ";" fields "${spec}")
  list(LENGTH fields n_fields)
  if(NOT n_fields EQUAL n)
    message(FATAL_ERROR "Malformed spec '${spec}': expected ${n} '|'-separated fields, got ${n_fields}.")
  endif()
  math(EXPR last_index "${n} - 1")
  foreach(i RANGE 0 ${last_index})
    list(GET fields ${i} value)
    list(GET ARGN ${i} out_var)
    set(${out_var} "${value}" PARENT_SCOPE)
  endforeach()
endfunction()

# e.g. svs_parse_isa_level("AVX2|haswell|avx2" level arch infix) sets
# level=AVX2, arch=haswell, infix=avx2
function(svs_parse_isa_level spec out_level out_arch out_infix)
  svs_split_fields("${spec}" 3 level arch infix)
  set(${out_level} "${level}" PARENT_SCOPE)
  set(${out_arch} "${arch}" PARENT_SCOPE)
  set(${out_infix} "${infix}" PARENT_SCOPE)
endfunction()

# e.g. svs_parse_tu_spec("avx2.cpp|AVX2|haswell|avx2" src level arch infix) sets
# src=avx2.cpp, level=AVX2, arch=haswell, infix=avx2
function(svs_parse_tu_spec spec out_src out_level out_arch out_infix)
  svs_split_fields("${spec}" 4 src level arch infix)
  set(${out_src} "${src}" PARENT_SCOPE)
  set(${out_level} "${level}" PARENT_SCOPE)
  set(${out_arch} "${arch}" PARENT_SCOPE)
  set(${out_infix} "${infix}" PARENT_SCOPE)
endfunction()

cmake_policy(POP)
