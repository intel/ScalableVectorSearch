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

# Runs cmake/validate-dispatch-surface.cmake over the fixtures in
# tests/cmake/dispatch-surface and checks each verdict.
#
#   valid-*.cmake    must be accepted.
#   invalid-*.cmake  must be rejected, with a message containing the substring
#                    given by that fixture's `# EXPECT-ERROR:` line.
#
# Needs nothing but cmake -- no compiler, no dependencies, no build directory.

set -uo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fixtures="${root}/tests/cmake/dispatch-surface"
validator="${root}/cmake/validate-dispatch-surface.cmake"
x86_src_dir="${root}/include/svs/multi-arch/x86"

if [[ ! -d ${fixtures} ]]; then
    echo "no fixture directory: ${fixtures}" >&2
    exit 1
fi

# The default declaration must itself be valid -- checked as its own case so that
# a broken default is reported here rather than only at configure time.
run_validator() {
    cmake "-DSVS_DISPATCH_SURFACE_FILE=$1" "-DSVS_X86_SRC_DIR=${x86_src_dir}" \
        -P "${validator}" 2>&1
}

# CMake indents and line-wraps error text, so compare against a whitespace-
# collapsed copy of the output.
flatten() { tr '\n' ' ' | tr -s '[:space:]' ' '; }

failures=0
checked=0

check_accepted() {
    local fixture=$1 name=$2 output status
    output=$(run_validator "${fixture}")
    status=$?
    if ((status != 0)); then
        echo "FAIL ${name}: expected to be accepted, but validation failed:" >&2
        echo "${output}" | sed 's/^/    /' >&2
        ((failures++))
    else
        echo "ok   ${name}: accepted"
    fi
    ((checked++))
}

check_rejected() {
    local fixture=$1 name=$2 expected output status
    expected=$(sed -n 's/^# EXPECT-ERROR: *//p' "${fixture}")
    if [[ -z ${expected} ]]; then
        echo "FAIL ${name}: fixture has no '# EXPECT-ERROR:' line" >&2
        ((failures++))
        ((checked++))
        return
    fi

    output=$(run_validator "${fixture}")
    status=$?
    if ((status == 0)); then
        echo "FAIL ${name}: expected rejection, but validation succeeded" >&2
        ((failures++))
    elif [[ $(printf '%s' "${output}" | flatten) != *"${expected}"* ]]; then
        echo "FAIL ${name}: rejected, but not for the stated reason." >&2
        echo "    expected: ${expected}" >&2
        echo "${output}" | sed 's/^/    actual:   /' >&2
        ((failures++))
    else
        echo "ok   ${name}: rejected (${expected})"
    fi
    ((checked++))
}

check_accepted "${root}/cmake/dispatch-surface.cmake" "dispatch-surface.cmake (default)"

for fixture in "${fixtures}"/*.cmake; do
    name=$(basename "${fixture}")
    case ${name} in
    valid-*) check_accepted "${fixture}" "${name}" ;;
    invalid-*) check_rejected "${fixture}" "${name}" ;;
    *)
        echo "FAIL ${name}: fixture name must start with valid- or invalid-" >&2
        ((failures++))
        ((checked++))
        ;;
    esac
done

echo
if ((failures != 0)); then
    echo "${failures} of ${checked} dispatch-surface checks failed" >&2
    exit 1
fi
echo "all ${checked} dispatch-surface checks passed"
