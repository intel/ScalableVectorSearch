# Copyright (C) 2023 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly stated
# in the License.

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
