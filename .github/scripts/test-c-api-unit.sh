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

# Run the C API unit tests out of an existing build tree.
#
# Inputs:
#   WORKSPACE   repository root; defaults to this script's repo so it also runs
#               outside the container

set -e

# Match build-c-api-bindings.sh: pick up the container's pinned gcc-toolset.
source /etc/bashrc 2>/dev/null || true

# Repo root, derived from this script's location so no git metadata is needed.
WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BUILD_DIR="${WORKSPACE}/build_c_api"

# LVQ/LeanVec need a specific ISA. The tests already accept
# SVS_ERROR_UNSUPPORTED_HW (but never SVS_ERROR_NOT_IMPLEMENTED), so this is
# reported for triage rather than used to skip anything.
echo "vendor: $(grep -m1 vendor_id /proc/cpuinfo || echo unknown)"
echo "model:  $(grep -m1 'model name' /proc/cpuinfo || echo unknown)"
echo "avx512: $(grep -o 'avx512[a-z_0-9]*' /proc/cpuinfo | sort -u | tr '\n' ' ')"

ctest --test-dir "${BUILD_DIR}" --output-on-failure --no-tests=error
