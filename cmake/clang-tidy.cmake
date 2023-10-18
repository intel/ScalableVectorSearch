#####
##### Try to find the clang-tidy executable and set it up.
#####

if(SVS_EXPERIMENTAL_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy-13 clang-tidy-12 clang-tidy)

    if(NOT CLANG_TIDY_EXE)
        message(WARNING "SVS_EXPERIMENTAL_CLANG_TIDY is ON but clang-tidy is not found!")
        set(CLANG_TIDY_COMMAND "" CACHE STRING "" FORCE)
    else()
        set(CLANG_TIDY_COMMAND
            "${CLANG_TIDY_EXE}"
            "--format-style=file"
            "--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy"
            "-header-filter=${CMAKE_SOURCE_DIR}/svs/*"
        )
        message("Clang tidy command: ${CLANG_TIDY_COMMAND}")
    endif()
endif()
