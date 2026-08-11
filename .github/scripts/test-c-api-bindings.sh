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

# Integration test for the packaged C API: verifies the tarball is a usable
# package rather than just a successful compile. Runs against the artifact only,
# with no access to the build tree.
#
# Inputs:
#   SUFFIX      artifact name suffix (e.g. -public-only)
#   WORKSPACE   repository root; defaults to this script's repo so it also runs
#               outside the container

set -e

# Match build-c-api-bindings.sh: pick up the container's pinned gcc-toolset.
source /etc/bashrc 2>/dev/null || true

# Repo root, derived from this script's location so no git metadata is needed.
WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAGE_DIR="${WORKSPACE}/c_api_integration"

# Prefer the artifact downloaded by the workflow, else a tarball built locally.
TARBALL="${WORKSPACE}/c_api_artifact/svs-c-api${SUFFIX}.tar.gz"
if [ ! -e "${TARBALL}" ]; then
    TARBALL="${WORKSPACE}/svs-c-api${SUFFIX}.tar.gz"
fi

INSTALL_DIR="${STAGE_DIR}/install"
CONSUMER_BUILD="${STAGE_DIR}/consumer-build"

rm -rf "${STAGE_DIR}"
mkdir -p "${INSTALL_DIR}"
tar -xzf "${TARBALL}" -C "${INSTALL_DIR}"

echo "::group::Package contents"
find "${INSTALL_DIR}" -type f -o -type l | sort
echo "::endgroup::"

LIBDIR="${INSTALL_DIR}/lib"
LIB="${LIBDIR}/libsvs_c_api.so"
if [ ! -e "${LIB}" ]; then
    echo "ERROR: ${LIB} missing from the package"
    exit 1
fi

echo "::group::Strong exported symbols"
nm -D --defined-only "${LIB}" | awk '$2=="T"{print $3}' | sort
echo "::endgroup::"

# Only the documented svs_* C ABI may be exported with strong linkage. This also
# guards the statically linked LVQ/LeanVec backend against leaking symbols.
#
# std:: template instantiations (_ZNSt/_ZSt) are excluded: GCC emits some of these
# with strong linkage from the LTO archive, and they are standard-library code
# rather than SVS implementation detail. The check still catches any leak of an
# actual svs/proprietary internal.
LEAKED=$(nm -D --defined-only "${LIB}" | awk '$2=="T"{print $3}' \
    | grep -v '^svs_' | grep -vE '^_Z+(N?)St' || true)
if [ -n "${LEAKED}" ]; then
    echo "ERROR: non-svs_ symbols exported from the C API:"
    echo "${LEAKED}"
    exit 1
fi

# Build a standalone C project against the installed CMake package, the way a
# downstream integration would. Catches exported-target defects (a missing
# find_dependency, or a C++ requirement leaking onto a C consumer) that a
# build-tree-only test cannot see.
cmake -B"${CONSUMER_BUILD}" -S"${WORKSPACE}/bindings/c/tests/consumer" \
    -DCMAKE_PREFIX_PATH="${INSTALL_DIR}"
cmake --build "${CONSUMER_BUILD}"

LD_LIBRARY_PATH="${LIBDIR}" "${CONSUMER_BUILD}/c_api_consumer"
