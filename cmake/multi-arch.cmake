# Copyright 2025 Intel Corporation
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

add_library(svs_x86_options_base INTERFACE)
add_library(svs::x86_options_base ALIAS svs_x86_options_base)
target_compile_options(svs_x86_options_base INTERFACE -march=nehalem -mtune=nehalem)

set(SVS_X86_SRC_DIR "${PROJECT_SOURCE_DIR}/include/svs/multi-arch/x86")
set(SVS_X86_SRC_FILES
  "${SVS_X86_SRC_DIR}/avx2.cpp"
  "${SVS_X86_SRC_DIR}/avx512.cpp"
)

set(SVS_X86_OBJ_NAMES
  "avx2"
  "avx512"
)
add_library(svs_x86_avx2 INTERFACE)
add_library(svs_x86_avx512 INTERFACE)
target_compile_options(svs_x86_avx2 INTERFACE -march=haswell -mtune=haswell)
target_compile_options(svs_x86_avx512 INTERFACE -march=cascadelake -mtune=cascadelake)
set(svs_x86_options
  svs_x86_avx2
  svs_x86_avx512
)

add_library(svs_x86_objects STATIC)
foreach(SRC OBJ_NAME x86_option IN ZIP_LISTS SVS_X86_SRC_FILES SVS_X86_OBJ_NAMES svs_x86_options)
    if(NOT EXISTS "${SRC}")
       message(FATAL_ERROR "Missing source file for: ${SRC}")
    endif()
    add_library(${OBJ_NAME} OBJECT ${SRC})

    target_link_libraries(${OBJ_NAME} PRIVATE ${SVS_LIB} svs::compile_options fmt::fmt ${x86_option})

    target_sources(svs_x86_objects PRIVATE $<TARGET_OBJECTS:${OBJ_NAME}>)
endforeach()
target_link_libraries(svs_export INTERFACE svs_x86_objects)
