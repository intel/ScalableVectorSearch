/*
 * Copyright 2026 Intel Corporation
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

#include "svs/core/distance/distance_core.h"
#include "svs/lib/avx_detection.h"

#include <algorithm>

namespace svs_test {

using svs::distance::AVX_AVAILABILITY;

// The runtime predicate behind each ISA level, and the only place the dispatch
// tests state it: a level added to the surface without a specialization here is
// an undefined symbol, deliberately -- the predicate cannot be guessed.
template <AVX_AVAILABILITY Level> bool host_satisfies();

template <> inline bool host_satisfies<AVX_AVAILABILITY::AVX2>() {
    return svs::detail::avx_runtime_flags.is_avx2_supported();
}

template <> inline bool host_satisfies<AVX_AVAILABILITY::AVX512>() {
    return svs::detail::avx_runtime_flags.is_avx512f_supported();
}

template <> inline bool host_satisfies<AVX_AVAILABILITY::AVX512_VNNI>() {
    return svs::detail::avx_runtime_flags.is_avx512vnni_supported();
}

/// The level the distance entry points must choose on this host.
///
/// The entry points test the strongest level first, so their choice is the
/// highest-numbered satisfied level; AVX_AVAILABILITY::NONE is the fallback.
inline int expected_level() {
    int highest = static_cast<int>(AVX_AVAILABILITY::NONE);
#define SVS_HOST_LEVEL_ONE(LEVEL)                                               \
    if (host_satisfies<AVX_AVAILABILITY::LEVEL>()) {                            \
        highest = std::max(highest, static_cast<int>(AVX_AVAILABILITY::LEVEL)); \
    }
    SVS_FOR_EACH_ISA_LEVEL(SVS_HOST_LEVEL_ONE)
#undef SVS_HOST_LEVEL_ONE
    return highest;
}

} // namespace svs_test
