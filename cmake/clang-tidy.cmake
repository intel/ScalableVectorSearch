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
    find_program(CLANG_TIDY_EXE
        NAMES
            clang-tidy-17 clang-tidy-18 clang-tidy
    )

    if(NOT CLANG_TIDY_EXE)
        message(WARNING "SVS_EXPERIMENTAL_CLANG_TIDY is ON but clang-tidy is not found!")
        set(CLANG_TIDY_COMMAND "" CACHE STRING "" FORCE)
    else()
        # Walk up to find .clang-tidy when built standalone from bindings/cpp.
        if(EXISTS "${CMAKE_SOURCE_DIR}/.clang-tidy")
            set(SVS_CLANG_TIDY_CONFIG "${CMAKE_SOURCE_DIR}/.clang-tidy")
            set(SVS_CLANG_TIDY_HEADER_FILTER "${CMAKE_SOURCE_DIR}/include/svs/.*")
        elseif(EXISTS "${CMAKE_SOURCE_DIR}/../../.clang-tidy")
            set(SVS_CLANG_TIDY_CONFIG "${CMAKE_SOURCE_DIR}/../../.clang-tidy")
            set(SVS_CLANG_TIDY_HEADER_FILTER "${CMAKE_SOURCE_DIR}/include/svs/.*")
        else()
            message(FATAL_ERROR "SVS_EXPERIMENTAL_CLANG_TIDY is ON but no .clang-tidy was found")
        endif()

        set(CLANG_TIDY_COMMAND
            "${CLANG_TIDY_EXE}"
            "--format-style=file"
            "--config-file=${SVS_CLANG_TIDY_CONFIG}"
            "--header-filter=${SVS_CLANG_TIDY_HEADER_FILTER}"
            "--warnings-as-errors=clang-diagnostic-*,bugprone-use-after-move"
            # Suppress noise from args (e.g. --gcc-toolchain) that affect link-time
            # paths but are unused during clang-tidy's parse-only invocation.
            "--extra-arg=-Wno-unused-command-line-argument"
            # Resolve <omp.h> via the gcc toolchain set below.
            "--extra-arg=-fopenmp"
        )

        # Point clang-tidy at gcc's toolchain so it can find libstdc++ headers.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            get_filename_component(_svs_gcc_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
            get_filename_component(_svs_gcc_toolchain "${_svs_gcc_bin_dir}" DIRECTORY)
            list(APPEND CLANG_TIDY_COMMAND
                "--extra-arg=--gcc-toolchain=${_svs_gcc_toolchain}"
            )
        endif()

        message(STATUS "Clang tidy command: ${CLANG_TIDY_COMMAND}")
    endif()
endif()
