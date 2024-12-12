# Copyright 2023 Intel Corporation
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
##### Try to find the clang-tidy executable and set it up.
#####

if(SVS_EXPERIMENTAL_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy-14 clang-tidy-13 clang-tidy-12 clang-tidy)

    if(NOT CLANG_TIDY_EXE)
        message(WARNING "SVS_EXPERIMENTAL_CLANG_TIDY is ON but clang-tidy is not found!")
        set(CLANG_TIDY_COMMAND "" CACHE STRING "" FORCE)
    else()
        set(CLANG_TIDY_COMMAND
            "${CLANG_TIDY_EXE}"
            "--format-style=file"
            "--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy"
            "--header-filter=${CMAKE_SOURCE_DIR}/include/svs/*"
        )
        message("Clang tidy command: ${CLANG_TIDY_COMMAND}")
    endif()
endif()
