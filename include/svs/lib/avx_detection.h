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

#ifdef __x86_64__
#include "eve/detection.hpp"
#endif

#include <dlfcn.h>

namespace svs::detail {

inline bool is_avx2_supported() {
#ifdef __x86_64__
    return eve::is_supported(eve::avx2);
#else
    return false;
#endif
}

inline bool is_avx512_supported() {
#ifdef __x86_64__
    return eve::is_supported(eve::avx512);
#else
    return false;
#endif
}

inline bool is_avx512vnni_supported() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;

    // Check if CPUID supports extended features (leaf 7)
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));

    // If the highest supported leaf is less than 7, extended features are not supported
    if (eax < 7) {
        return false;
    }

    // Get extended features (leaf 7, sub-leaf 0)
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(7), "c"(0));

    // Check for AVX512_VNNI support (bit 11 of ECX)
    return (ecx & (1 << 11)) != 0;
#else
    return false;
#endif
}

} // namespace svs::detail
