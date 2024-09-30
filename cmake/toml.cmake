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

include(FetchContent)

# By default, tomlplusplus is not configured to enable installation.
# The patch we carry adds an override.

# We use an external script that tries to apply the patch only once.
# This hopefully allows us to modify our CMake infrastructure after an
# initial configuration without the patch being applied again and failing.
set(TOML_PATCH "${CMAKE_CURRENT_LIST_DIR}/patches/tomlplusplus_v330.patch")
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.3.0
    PATCH_COMMAND
    ${CMAKE_CURRENT_LIST_DIR}/patches/apply_patch_toml.sh ${TOML_PATCH}
)

# Set the override variable to enable toml++ installation.
set(TOMLPLUSPLUS_INSTALL ON)
FetchContent_MakeAvailable(tomlplusplus)
target_link_libraries(${SVS_LIB} INTERFACE tomlplusplus::tomlplusplus)
