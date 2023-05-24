Include(FetchContent)
FetchContent_Declare(
    eve
    GIT_REPOSITORY https://github.com/jfalcou/eve
    GIT_TAG v2022.09.1
)

set(EVE_BUILD_TEST OFF)
FetchContent_MakeAvailable(eve)
target_link_libraries(${SVS_LIB} INTERFACE eve::eve)
