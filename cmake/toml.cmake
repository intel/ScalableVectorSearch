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
