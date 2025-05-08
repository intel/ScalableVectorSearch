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

# Exit immediately if error
set -e

##### CPU Name #####
if [[ -n "$SDE_FLAG" ]]; then
    echo "CPU: SDE Emulation ($SDE_FLAG)"
else
    if [[ "$(uname -s)" == "Linux" ]]; then
        echo "CPU: $(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d':' -f2 | xargs)"
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
    else
        echo "CPU: unsupported OS"
    fi
fi

##### AVX Support #####
ARCH=$(uname -m)
# Only check AVX support on x86 architectures
if [[ "$ARCH" != "x86_64" && "$ARCH" != "i386" && "$ARCH" != "i686" ]]; then
    echo "AVX Support: Not applicable"
    exit 0
fi

# Check AVX support with compiler intrinsics
cat > check_avx.c <<'EOF'
#include <stdio.h>
#include <immintrin.h>

int main() {
    if (__builtin_cpu_supports("avx512f")) {
        printf("AVX512\n");
    } else if (__builtin_cpu_supports("avx2")) {
        printf("AVX2\n");
    } else if (__builtin_cpu_supports("avx")) {
        printf("AVX\n");
    } else {
        printf("No AVX\n");
    }
    return 0;
}
EOF

# Compile and run the AVX detection program
echo -n "AVX Support: "
if gcc -O2 check_avx.c -o check_avx 2>/dev/null; then
    if [[ -n "$SDE_FLAG" ]]; then
        sde64 -$SDE_FLAG -- ./check_avx
    else
        ./check_avx
    fi
else
    echo "Detection failed with compiler error"
fi

# Clean up
rm -f check_avx check_avx.c