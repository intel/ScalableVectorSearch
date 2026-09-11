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

function(svs_require)          # svs_require(VAR_A VAR_B ...)
  foreach(var IN LISTS ARGN)
    if(NOT ${var})
      message(FATAL_ERROR "${var} is not set.")
    endif()
  endforeach()
endfunction()

function(svs_require_files)    # svs_require_files(VAR_A VAR_B ...)
  foreach(var IN LISTS ARGN)
    if(NOT EXISTS "${${var}}")
      message(FATAL_ERROR "${var} does not exist: ${${var}}")
    endif()
  endforeach()
endfunction()

function(svs_nm_symbols out_var file regex)
  cmake_parse_arguments(arg "" "" "NM_ARGS" ${ARGN})
  execute_process(
    COMMAND "${SVS_NM}" ${arg_NM_ARGS} "${file}"
    OUTPUT_VARIABLE raw
    ERROR_VARIABLE err
    RESULT_VARIABLE status
  )
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${SVS_NM} failed on ${file}: ${err}")
  endif()

  set(symbols)
  string(REPLACE "\n" ";" lines "${raw}")
  foreach(line IN LISTS lines)
    if(line MATCHES "${regex}")
      # The mangled name is the last whitespace-separated field.
      string(REGEX MATCH "[^ \t]+$" symbol "${line}")
      list(APPEND symbols "${symbol}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES symbols)
  set(${out_var} "${symbols}" PARENT_SCOPE)
endfunction()

function(svs_report_first list_var limit)
  set(shown ${${list_var}})
  list(LENGTH shown count)
  if(count GREATER limit)
    list(SUBLIST shown 0 ${limit} shown)
  endif()
  foreach(entry IN LISTS shown)
    message("      ${entry}")
  endforeach()
  if(count GREATER limit)
    math(EXPR rest "${count} - ${limit}")
    message("      ... and ${rest} more")
  endif()
endfunction()
