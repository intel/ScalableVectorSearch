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

function(create_x86_instantiations src_files obj_names x86_options)
    set(X86_OBJECT_FILES "")
    foreach(src obj_name x86_option IN ZIP_LISTS src_files obj_names x86_options)
        if(NOT EXISTS "${src}")
           message(FATAL_ERROR "Missing source file for x86: ${src}")
        endif()
        add_library(${obj_name} OBJECT ${src})

        target_link_libraries(${obj_name} PRIVATE ${SVS_LIB} svs::compile_options fmt::fmt ${x86_option})

        list(APPEND X86_OBJECT_FILES $<TARGET_OBJECTS:${obj_name}>)
    endforeach()

    set(X86_OBJECT_FILES "${X86_OBJECT_FILES}" CACHE INTERNAL "X86-specific object files")
endfunction()
