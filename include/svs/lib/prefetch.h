/*
 * Copyright 2023 Intel Corporation
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

/////
///// Loop Prefetcher.
/////

struct PrefetchParameters {
    size_t lookahead = 1;
    size_t step = 1;
};

namespace detail {
inline size_t select_imax(PrefetchParameters p, size_t imax) {
    return ((p.lookahead == 0) || (p.step == 0)) ? 0 : imax;
}
} // namespace detail

template <typename Op, typename Pred = ReturnsTrueType> class Prefetcher {
  public:
    // Static assertions.
    static_assert(
        std::invocable<Op, size_t>,
        "Prefetch operation must be callable with an integer argument!"
    );
    static_assert(
        std::invocable<Pred, size_t>,
        "Prefetch predicate must be callable with an integer argument!"
    );

    // Constructor.
    Prefetcher(PrefetchParameters parameters, size_t imax, Op&& op, Pred&& pred = Pred{})
        : step_{parameters.step}
        , i_{0}
        , imax_{detail::select_imax(parameters, imax)}
        , slack_{std::min(imax_, parameters.lookahead)}
        , op_{std::move(op)}
        , pred_{std::move(pred)} {}

    Prefetcher(
        PrefetchParameters parameters, size_t imax, const Op& op, const Pred& pred = Pred{}
    )
        : step_{parameters.step}
        , i_{0}
        , imax_{detail::select_imax(parameters, imax)}
        , slack_{std::min(imax_, parameters.lookahead)}
        , op_{op}
        , pred_{pred} {}

    // Advance the prefetcher
    void operator()() {
        assert(i_ <= imax_);

        // Done prefetching.
        if (done()) {
            return;
        }

        // In steady state, prefetch one valid item.
        if (slack_ == 0) {
            // Increment to a non-skipped value to prefetch.
            while (!pred_(i_)) {
                i_++;
                if (done()) {
                    return;
                }
            }
            assert(!done());
            op_(i_++);
            return;
        }

        // Setting step = 0 should or lookahead == 0 should disable prefetching entirely.
        assert(step_ > 0);
        const size_t items_to_prefetch =
            (step_ == 1) ? (slack_ + 1) : std::min(slack_ + 1, step_);
        size_t prefetched = 0;
        do {
            // Walk forward until the predicate is satisfied.
            while (!pred_(i_)) {
                i_++;
                if (done()) {
                    return;
                }
            }
            // Prefetch this item.
            assert(!done());
            op_(i_++);
            ++prefetched;
        } while (!done() && prefetched != items_to_prefetch);
        slack_ -= (prefetched - 1);
    }

    bool done() const { return i_ == imax_; }

  private:
    size_t step_;
    size_t i_;
    size_t imax_;
    size_t slack_;
    [[no_unique_address]] Op op_;
    [[no_unique_address]] Pred pred_;
};

template <typename Op>
Prefetcher<std::remove_cvref_t<Op>>
make_prefetcher(PrefetchParameters parameters, size_t imax, Op&& op) {
    return Prefetcher<std::remove_cvref_t<Op>>{parameters, imax, SVS_FWD(op)};
}

template <typename Op, typename Pred>
Prefetcher<std::remove_cvref_t<Op>, std::remove_cvref_t<Pred>>
make_prefetcher(PrefetchParameters parameters, size_t imax, Op&& op, Pred&& pred) {
    return Prefetcher<std::remove_cvref_t<Op>, std::remove_cvref_t<Pred>>{
        parameters, imax, SVS_FWD(op), SVS_FWD(pred)};
}

} // namespace svs::lib
