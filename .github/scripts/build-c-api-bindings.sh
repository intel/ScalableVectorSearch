#!/bin/bash
# Copyright 2026 Intel Corporation
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

# Configure, build, install and package the C API bindings.
#
# Inputs (all optional, with defaults suitable for a local run):
#   ENABLE_LVQ_LEANVEC  ON to statically link the LVQ/LeanVec backend
#   REQUIRE_LTO_ARCHIVE ON to fail (not warn) if the compiler can't consume the
#                       LTO archive; set in CI, left off for local builds
#   SUFFIX              artifact name suffix (e.g. -public-only)
#   WORKSPACE           repository root; defaults to this script's repo so it
#                       also runs outside the container

set -e

# In the manylinux/rockylinux containers the pinned gcc-toolset lives behind an
# scl profile script; harmless no-op on a plain runner.
source /etc/bashrc 2>/dev/null || true

# Repo root, derived from this script's location so no git metadata is needed.
WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD_DIR="${WORKSPACE}/build_c_api"
INSTALL_DIR="${WORKSPACE}/install_c_api"
ENABLE_LVQ_LEANVEC="${ENABLE_LVQ_LEANVEC:-OFF}"
REQUIRE_LTO_ARCHIVE="${REQUIRE_LTO_ARCHIVE:-OFF}"

echo "compiler: $(${CXX:-c++} --version | head -1)"

rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -B"${BUILD_DIR}" -S"${WORKSPACE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DSVS_BUILD_C_API=ON \
    -DSVS_BUILD_TESTS=ON \
    -DSVS_BUILD_EXAMPLES=ON \
    -DSVS_RUNTIME_ENABLE_LVQ_LEANVEC="${ENABLE_LVQ_LEANVEC}" \
    -DSVS_REQUIRE_LTO_ARCHIVE="${REQUIRE_LTO_ARCHIVE}"

cmake --build "${BUILD_DIR}" -j"$(nproc)"

# Install only the C API component: the dependency headers that a full install
# would also emit are not part of the shipped interface.
cmake --install "${BUILD_DIR}" --component C_API

tar -czf "${WORKSPACE}/svs-c-api${SUFFIX}.tar.gz" -C "${INSTALL_DIR}" .
echo "Packaged ${WORKSPACE}/svs-c-api${SUFFIX}.tar.gz"
