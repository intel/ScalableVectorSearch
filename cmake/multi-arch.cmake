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

set(SVS_X86_SRC_DIR "${PROJECT_SOURCE_DIR}/include/svs/multi-arch/x86")
set(SVS_X86
  "${SVS_X86_SRC_DIR}/avx2.cpp,avx2,haswell"
  "${SVS_X86_SRC_DIR}/avx512.cpp,avx512,cascadelake"
)

# AMX (tile load/store/config plus the bf16 dot-product tile instruction) is optional:
# -march=sapphirerapids is the flag amx.cpp is built with below, same one-codename-per-ISA
# pattern as haswell/cascadelake above, and the first compiler/-march combination that
# ships it; a compiler that rejects it (too old, or a non-x86 cross build) must still
# produce a working library with the level compiled out, so amx.cpp is added to SVS_X86
# only when this check passes, and SVS_AMX_TILE_BF16_SUPPORTED (read by
# include/svs/lib/preprocessor.h) is defined only then.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=sapphirerapids" SVS_COMPILER_SUPPORTS_AMX)
if(SVS_COMPILER_SUPPORTS_AMX)
  list(APPEND SVS_X86 "${SVS_X86_SRC_DIR}/amx.cpp,amx,sapphirerapids")
endif()

set(SVS_X86_OBJECT_FILES)
foreach(x86_info IN LISTS SVS_X86)
    string(REPLACE "," ";" x86_info "${x86_info}")
    list(GET x86_info 0 src)
    list(GET x86_info 1 avx)
    list(GET x86_info 2 arch)
    set(lib_name "svs_x86_${avx}")
    add_library(${lib_name} INTERFACE)
    target_compile_options(${lib_name} INTERFACE -march=${arch} -mtune=${arch})
    set(obj_name ${arch}_obj)

    add_library(${obj_name} OBJECT ${src})
    target_link_libraries(${obj_name} PRIVATE ${SVS_LIB} svs::compile_options fmt::fmt ${lib_name})
    list(APPEND SVS_X86_OBJECT_FILES $<TARGET_OBJECTS:${obj_name}>)
endforeach()

add_library(svs_x86_objects STATIC ${SVS_X86_OBJECT_FILES})
target_link_libraries(svs_export INTERFACE svs_x86_objects)

if(SVS_COMPILER_SUPPORTS_AMX)
  # Uniform across every consumer of svs_export, unlike the per-file -march flags above:
  # PackedOperand (inner_product_matrix.h) decides whether to fill the AMX tile layout
  # from this, and it is included by TUs that never pass -march=sapphirerapids
  # themselves (gem.h, maxsim.h, tests).
  target_compile_definitions(svs_export INTERFACE SVS_AMX_TILE_BF16_SUPPORTED)
endif()

install(
    TARGETS svs_x86_objects
    EXPORT svs-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
