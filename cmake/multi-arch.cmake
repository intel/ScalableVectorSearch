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

# function to create a set of object files with microarch instantiations

add_library(svs_x86_options_base INTERFACE)
set(X86_OBJECT_FILES "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
    function(create_x86_instantiations)
    endfunction()
    return()
endif()
target_compile_options(svs_x86_options_base INTERFACE -march=nehalem -mtune=nehalem)

set(SVS_X86_DIR "${PROJECT_SOURCE_DIR}/include/svs/multi-arch/x86")


# TODO: refactor
set(svs_x86_avx_list "avx2" "avx512")

add_library(svs_x86_avx2 INTERFACE)
add_library(svs_x86_avx512 INTERFACE)
target_compile_options(svs_x86_avx2 INTERFACE -march=skylake -mtune=skylake)
target_compile_options(svs_x86_avx512 INTERFACE -march=cascadelake -mtune=cascadelake)

set(svs_x86_options "")
list(APPEND svs_x86_options svs_x86_avx2)
list(APPEND svs_x86_options svs_x86_avx512)

function(create_x86_instantiations)
    set(X86_OBJECT_FILES "")
    foreach(avx x86_option IN ZIP_LISTS svs_x86_avx_list svs_x86_options)
        set(OBJ_NAME "x86_${avx}")
        set(SRC_FILE "${SVS_X86_DIR}/${avx}.cpp")
        if(NOT EXISTS "${SRC_FILE}")
           message(FATAL_ERROR "Missing source file for x86: ${SRC_FILE}")
        endif()
        add_library(${OBJ_NAME} OBJECT ${SRC_FILE})

        target_link_libraries(${OBJ_NAME} PRIVATE ${SVS_LIB} svs::compile_options fmt::fmt ${x86_option})

        list(APPEND X86_OBJECT_FILES $<TARGET_OBJECTS:${OBJ_NAME}>)
    endforeach()

    set(X86_OBJECT_FILES "${X86_OBJECT_FILES}" CACHE INTERNAL "X86-specific object files")
endfunction()
