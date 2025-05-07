#!/bin/bash

# Check AVX support using compiler intrinsics
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

##### CPU Name #####
if [[ -n "$SDE_FLAG" ]]; then
    echo "CPU: SDE Emulated $SDE_FLAG"
else
    if [[ "$(uname -s)" == "Linux" ]]; then
        echo "CPU: $(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d':' -f2 | xargs)"
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
    else
        echo "CPU: Unknown (unsupported OS)"
    fi
fi

###### AVX Support #####
echo -n "AVX Support: "
gcc -O2 check_avx.c -o check_avx

if [[ -n "$SDE_FLAG" ]]; then
    sde64 -$SDE_FLAG -- ./check_avx
else
    ./check_avx
fi

rm check_avx check_avx.c