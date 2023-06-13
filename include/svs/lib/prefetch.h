/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

#include "svs/lib/misc.h"

#include <cstdint>
#include <iterator>
#include <span>
#ifdef __SSE__
#include <x86intrin.h>
#endif

namespace svs::lib {

#ifdef __SSE__
template <typename T> void prefetch_l0(const T* ptr) {
    _mm_prefetch(static_cast<const void*>(ptr), _MM_HINT_T0);
}
#else
// Do nothing if prefetch is not-available.
template <typename T> void prefetch_l0(const T* ptr) {}
#endif

const size_t CACHELINE_BYTES = 64;
const size_t MAX_EXTRA_PREFETCH_CTRL = 3;

template <typename T, size_t Extent> void prefetch_l0(std::span<T, Extent> span) {
    auto byte_span = std::as_bytes(span);
    const std::byte* base = byte_span.data();
    const size_t bytes = byte_span.size();
    size_t num_prefetches = div_round_up(bytes, CACHELINE_BYTES);

    if (Extent != Dynamic) { // Dimension provided at compile time
        // It sends an extra prefetch when size is not multiple of 64 and is less than
        // certain number of cachelines. Logic behind the last constraint is to save
        // bandwidth as stream prefetch kicks in after few consecutive accesses.
        // NOTE: these heuristics are based on the empirical results of our workloads
        num_prefetches += static_cast<size_t>(
            (bytes % CACHELINE_BYTES != 0) &&
            (bytes < MAX_EXTRA_PREFETCH_CTRL * CACHELINE_BYTES)
        );
    }

    for (size_t i = 0; i < num_prefetches; ++i) {
        prefetch_l0(base + CACHELINE_BYTES * i);
    }
}

// Default prefetching to L0
template <typename... Args> void prefetch(Args&&... args) {
    prefetch_l0(std::forward<Args>(args)...);
}
} // namespace svs::lib
