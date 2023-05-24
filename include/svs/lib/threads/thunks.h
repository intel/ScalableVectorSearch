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

// stdlib
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// local
#include "svs/lib/misc.h"
#include "svs/lib/threads/types.h"

namespace svs {
namespace threads {

// The function type used by the lowest level threads.
using FunctionType = std::function<void(uint64_t)>;
struct ThreadFunctionRef {
    FunctionType* fn;
    size_t thread_id;
    void operator()() const { (*fn)(thread_id); }
};

/////
///// Thunks
/////

namespace thunks {
namespace concepts {

template <typename F>
concept ConvertibleToFunctionType = std::convertible_to<F, FunctionType>;

} // namespace concepts

template <typename F, typename... Args> struct Thunk {};

template <typename F>
    requires concepts::ConvertibleToFunctionType<F>
struct Thunk<F> {
    static FunctionType wrap(ThreadCount /*unused*/, F& f) {
        return FunctionType{std::forward<F>(f)};
    }
};

// Static partition
template <typename F, typename I> struct Thunk<F, StaticPartition<I>> {
    static FunctionType wrap(ThreadCount nthreads, F& f, StaticPartition<I> space) {
        // Captures:
        // - `f` by reference: Lives outside function scope.
        // - `space` by value: Cheap to copy.
        // - `nthreads` by value: Cheap to copy.
        return [&f, space, nthreads](uint64_t tid) {
            auto nthr = static_cast<size_t>(nthreads);
            auto r = balance(space.size(), nthr, tid);

            // No work for this thread.
            if (r.empty()) {
                return;
            }

            auto this_range =
                IteratorPair{space.begin() + r.start(), space.begin() + r.stop()};
            f(this_range, tid);
        };
    }
};

// Dynamic partition
template <typename F, typename I> struct Thunk<F, DynamicPartition<I>> {
    static FunctionType
    wrap(ThreadCount SVS_UNUSED(nthreads), F& f, DynamicPartition<I> space) {
        auto count = std::make_shared<std::atomic<uint64_t>>(0);
        return [&f, space, count](uint64_t tid) {
            size_t grainsize = space.grainsize;
            size_t iterator_size = space.size();
            for (;;) {
                uint64_t i = count->fetch_add(1, std::memory_order_relaxed);
                auto start = grainsize * i;
                if (start >= iterator_size) {
                    return;
                }

                auto stop = std::min(grainsize * (i + 1), iterator_size);
                auto this_range =
                    IteratorPair{std::begin(space) + start, std::begin(space) + stop};
                f(this_range, tid);
            }
        };
    }
};

// Thunk entry point.
template <typename F, typename... Args>
FunctionType wrap(ThreadCount nthreads, F& f, Args&&... args) {
    return Thunk<F, std::decay_t<Args>...>::wrap(nthreads, f, std::forward<Args>(args)...);
}

} // namespace thunks
} // namespace threads
} // namespace svs
