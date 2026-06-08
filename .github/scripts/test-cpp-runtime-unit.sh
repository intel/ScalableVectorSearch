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

set -e

# Source environment setup (for compiler and MKL)
source /etc/bashrc || true
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh 2>/dev/null || true
fi

CTEST_DIR="bindings/cpp/build_cpp_bindings/tests"

# Check if running on Intel hardware or if LVQ/LeanVec is not enabled
if grep -q "GenuineIntel" /proc/cpuinfo || [ "${ENABLE_LVQ_LEANVEC}" != "ON" ]; then
    ctest --test-dir $CTEST_DIR --output-on-failure --no-tests=error --verbose
else
    echo "Non-Intel CPU detected - running tests with LVQ/LeanVec XFAIL"
    # Run non-LVQ/LeanVec tests normally
    ctest --test-dir $CTEST_DIR --output-on-failure --no-tests=error --verbose -E "(LVQ|LeanVec)"

    # Run LVQ/LeanVec tests expecting failure on non-Intel hardware
    set +e
    ctest --test-dir $CTEST_DIR --output-on-failure --verbose -R "(LVQ|LeanVec)"
    exit_code=$?
    set -e

    if [ $exit_code -ne 0 ]; then
        echo "XFAIL: LVQ/LeanVec ctest tests failed as expected on non-Intel hardware"
    else
        echo "UNEXPECTED: LVQ/LeanVec ctest tests passed on non-Intel hardware"
        exit 1
    fi
fi
