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

# Writes the generated dispatch-surface header and populates
# SVS_DISPATCH_TU_SPECS -- "<src>|<level>|<arch>|<infix>", one entry per ISA
# level. The extent list and the levels themselves are declared in
# cmake/dispatch-surface.cmake.
include("${CMAKE_CURRENT_LIST_DIR}/generate-dispatch-surface.cmake")

set(SVS_X86_OBJECT_FILES)
foreach(tu_spec IN LISTS SVS_DISPATCH_TU_SPECS)
    string(REPLACE "|" ";" tu_fields "${tu_spec}")
    list(GET tu_fields 0 src)
    list(GET tu_fields 2 arch)
    list(GET tu_fields 3 infix)

    # Carries the instruction budget for this level, and nothing else.
    set(lib_name "svs_x86_${infix}")
    add_library(${lib_name} INTERFACE)
    target_compile_options(${lib_name} INTERFACE -march=${arch} -mtune=${arch})

    set(obj_name ${arch}_obj)
    add_library(${obj_name} OBJECT ${src})
    target_link_libraries(
        ${obj_name} PRIVATE ${SVS_LIB} svs::compile_options fmt::fmt ${lib_name}
    )
    list(APPEND SVS_X86_OBJECT_FILES $<TARGET_OBJECTS:${obj_name}>)
endforeach()

add_library(svs_x86_objects STATIC ${SVS_X86_OBJECT_FILES})
target_link_libraries(svs_export INTERFACE svs_x86_objects)

install(
    TARGETS svs_x86_objects
    EXPORT svs-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
