Include(FetchContent)
FetchContent_Declare(
    RobinMap
    GIT_REPOSITORY https://github.com/Tessil/robin-map
    GIT_TAG v1.0.1
)

FetchContent_MakeAvailable(RobinMap)
target_link_libraries(${SVS_LIB} INTERFACE tsl::robin_map)
