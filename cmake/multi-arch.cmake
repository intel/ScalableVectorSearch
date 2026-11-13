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
# Tuple format: source,name,arch[,extra_option,extra_option,...]. Fields beyond the
# first three are optional and additive: an existing tuple (no 4th+ field) still gets
# exactly -march=${arch} -mtune=${arch}, unchanged; a tuple that adds fields gets those
# two flags plus every extra option, appended verbatim. amx.cpp uses this to compile at
# exactly the AVX-512 level's instruction set (arch=cascadelake, matching the avx512
# tuple below) plus the two AMX extensions, rather than at -march=sapphirerapids: that
# also enables AVX512-FP16/BF16/AVX-VNNI/etc, which the runtime probe
# (amx_tile_bf16_available()) never tests and a hypervisor could expose AMX without, so
# the compiler must not be free to use them when auto-vectorizing ordinary code in that
# file.
set(SVS_X86
  "${SVS_X86_SRC_DIR}/avx2.cpp,avx2,haswell"
  "${SVS_X86_SRC_DIR}/avx512.cpp,avx512,cascadelake"
)

# AMX (tile load/store/config plus the bf16 dot-product tile instruction) is optional: a
# compiler that rejects -mamx-tile/-mamx-bf16 (too old, or a non-x86 cross build) must
# still produce a working library with the level compiled out, so amx.cpp is added to
# SVS_X86 only when this check passes, and SVS_AMX_TILE_BF16_SUPPORTED (read by
# include/svs/lib/preprocessor.h) is defined only then.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mamx-tile -mamx-bf16" SVS_COMPILER_SUPPORTS_AMX)
if(SVS_COMPILER_SUPPORTS_AMX)
  list(APPEND SVS_X86 "${SVS_X86_SRC_DIR}/amx.cpp,amx,cascadelake,-mamx-tile,-mamx-bf16")
endif()

set(SVS_X86_OBJECT_FILES)
foreach(x86_info IN LISTS SVS_X86)
    string(REPLACE "," ";" x86_info "${x86_info}")
    list(LENGTH x86_info x86_info_len)
    list(GET x86_info 0 src)
    list(GET x86_info 1 avx)
    list(GET x86_info 2 arch)
    set(lib_name "svs_x86_${avx}")
    add_library(${lib_name} INTERFACE)
    target_compile_options(${lib_name} INTERFACE -march=${arch} -mtune=${arch})
    # Object target names key off `arch` by default, but the amx tuple above
    # intentionally reuses cascadelake's `arch` (same instruction set as the avx512
    # tuple, see the comment above SVS_X86): when extra options are present, the object
    # target is named from `avx` instead, so it doesn't collide with that other tuple's
    # target of the same name.
    if(x86_info_len GREATER 3)
        list(SUBLIST x86_info 3 -1 extra_options)
        target_compile_options(${lib_name} INTERFACE ${extra_options})
        set(obj_name ${avx}_obj)
    else()
        set(obj_name ${arch}_obj)
    endif()

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
