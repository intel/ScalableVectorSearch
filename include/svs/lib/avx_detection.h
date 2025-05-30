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

        bool extended_features = eax >= 7;

        __asm__ __volatile__("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(7), "c"(0));

        avx2 = extended_features && ((ebx & (1 << 5)) != 0);
        avx512f = extended_features && ((ebx & (1 << 16)) != 0);
        avx512vnni = extended_features && ((ecx & (1 << 11)) != 0);
    }

    bool is_avx2_supported() const noexcept { return avx2; }
    bool is_avx512f_supported() const noexcept { return avx512f; }
    bool is_avx512vnni_supported() const noexcept { return avx512vnni; }

    bool avx2;
    bool avx512f;
    bool avx512vnni;
};
#else
struct AVXRuntimeFlags {
    bool is_avx2_supported() const noexcept { return false; }
    bool is_avx512f_supported() const noexcept { return false; }
    bool is_avx512vnni_supported() const noexcept { return false; }
};
#endif

inline const AVXRuntimeFlags avx_runtime_flags = {};

} // namespace svs::detail
