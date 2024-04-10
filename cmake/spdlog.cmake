# Fetch `spdlog`
# Configure it to use our version of `fmtlib`.
# Since we consume spdlog

# TODO: We need a better way of doing dependency management and potentially linking
# against system libraries instead of always downloading.
#
# As far as I can tell, libraries compiled with C++11 *should* be ABI compatible with
# those compiled with C++20 (modulo compiler bugs).
#
# However, if we can head off some of those issues, might as well.
set(PRESET_CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD ${SVS_CXX_STANDARD})

set(SPDLOG_INSTALL YES CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL YES CACHE BOOL "" FORCE)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)
target_link_libraries(${SVS_LIB} INTERFACE spdlog::spdlog)

set(CMAKE_CXX_STANDARD ${PRESET_CMAKE_CXX_STANDARD})
