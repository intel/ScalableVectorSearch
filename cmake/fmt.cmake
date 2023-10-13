set(CMAKE_POSITION_INDEPENDENT_CODE YES)
set(FMT_INSTALL YES)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 10.1.1
)
FetchContent_MakeAvailable(fmt)
target_link_libraries(${SVS_LIB} INTERFACE fmt::fmt)
