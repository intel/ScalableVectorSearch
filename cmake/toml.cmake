include(FetchContent)

# By default, tomlplusplus is not configured to enable installation.
# The patch we carry adds an override.
set(TOML_PATCH "tomlplusplus_v330.patch")
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.3.0
    PATCH_COMMAND
    git apply --ignore-whitespace "${CMAKE_CURRENT_LIST_DIR}/patches/${TOML_PATCH}"
)

# Set the override variable to enable toml++ installation.
set(TOMLPLUSPLUS_INSTALL ON)
FetchContent_MakeAvailable(tomlplusplus)
target_link_libraries(${SVS_LIB} INTERFACE tomlplusplus::tomlplusplus)
