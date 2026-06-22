/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <dlfcn.h>

namespace svs::detail {

#ifdef __x86_64__
struct AVXRuntimeFlags {
    AVXRuntimeFlags() {
        unsigned int eax, ebx, ecx, edx;

        __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));

        unsigned int max_leaf = eax;
        bool extended_features = max_leaf >= 7;

        __asm__ __volatile__("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(7), "c"(0));

        avx2 = extended_features && ((ebx & (1 << 5)) != 0);
        avx512f = extended_features && ((ebx & (1 << 16)) != 0);
        avx512vnni = extended_features && ((ecx & (1 << 11)) != 0);

        // AVX10 is enumerated by leaf 7 sub-leaf 0 EDX bit 19, with the version and
        // supported vector widths reported by leaf 0x24 sub-leaf 0.
        bool avx10_present = extended_features && ((edx & (1 << 19)) != 0);
        if (avx10_present && max_leaf >= 0x24) {
            __asm__ __volatile__("cpuid"
                                 : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                                 : "a"(0x24), "c"(0));
            avx10_version = ebx & 0xFF;             // EBX[7:0]
            avx10_512 = (ebx & (1 << 18)) != 0;     // EBX[18]: 512-bit vector support
        }
    }

    bool is_avx2_supported() const noexcept { return avx2; }
    bool is_avx512f_supported() const noexcept { return avx512f; }
    bool is_avx512vnni_supported() const noexcept { return avx512vnni; }

    // True when the CPU implements AVX10 at 512-bit width. Such parts run the existing
    // AVX-512 kernels even if they no longer enumerate the legacy AVX512F bit.
    bool is_avx10_supported() const noexcept { return avx10_version > 0 && avx10_512; }
    unsigned int avx10_version_supported() const noexcept { return avx10_version; }

    // The AVX-512 distance kernels are reachable on either legacy AVX-512 or AVX10/512.
    bool is_avx512_path_supported() const noexcept {
        return avx512f || is_avx10_supported();
    }

    bool avx2;
    bool avx512f;
    bool avx512vnni;
    unsigned int avx10_version = 0;
    bool avx10_512 = false;
};
#else
struct AVXRuntimeFlags {
    bool is_avx2_supported() const noexcept { return false; }
    bool is_avx512f_supported() const noexcept { return false; }
    bool is_avx512vnni_supported() const noexcept { return false; }
    bool is_avx10_supported() const noexcept { return false; }
    unsigned int avx10_version_supported() const noexcept { return 0; }
    bool is_avx512_path_supported() const noexcept { return false; }
};
#endif

inline const AVXRuntimeFlags avx_runtime_flags = {};

} // namespace svs::detail
