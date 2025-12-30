#!/bin/bash
# Copyright 2025 Intel Corporation
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

set -e

# Source gcc-toolset-11 to ensure we use the system's GCC 11
# This is required because we are not using Conda compilers
if [ -f "/opt/rh/gcc-toolset-11/enable" ]; then
    source /opt/rh/gcc-toolset-11/enable
elif command -v scl_source >/dev/null 2>&1; then
    source scl_source enable gcc-toolset-11 || true
else
    echo "WARNING: gcc-toolset-11 not found, proceeding without sourcing it"
fi

# build runtime tests flag?
CMAKE_ARGS=(
    "-DCMAKE_INSTALL_PREFIX=${PREFIX}"
    "-DSVS_RUNTIME_ENABLE_LVQ_LEANVEC=${ENABLE_LVQ_LEANVEC:-ON}"
)

# Add SVS_URL if specified (for fetching static library)
if [ -n "$SVS_URL" ]; then
    CMAKE_ARGS+=("-DSVS_URL=$SVS_URL")
fi

cmake -B build "${CMAKE_ARGS[@]}" -S .

cmake --build build -j
cmake --install build

# Create lib64 symlink if needed (for compatibility)
cd "${PREFIX}"
if [ ! -e lib64 ] && [ -d lib ]; then
    ln -s lib lib64
fi
