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

// stdlib
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

// local
#include "svs/lib/misc.h"
#include "svs/lib/threads/types.h"

namespace svs {
namespace threads {

// Move on Copy
template <typename T> struct MoC {
    MoC(T&& rhs)
        : obj(std::move(rhs)) {}
    MoC(const MoC& other)
        : obj(std::move(other.obj)) {}
    T& get() { return obj; }
    mutable T obj;
};

struct ThreadFunctionRef {
  public:
    const std::function<void(size_t)>* fn{nullptr};
    size_t thread_id = 0;

  public:
    ThreadFunctionRef() = default;
    ThreadFunctionRef(const std::function<void(size_t)>* fn_, size_t thread_id_)
        : fn{fn_}
        , thread_id{thread_id_} {}

    void operator()() const { (*fn)(thread_id); }
};

/////
///// Thunks
/////

namespace thunks {

template <typename F, typename... Args> struct Thunk {};

// No change to the underlying lambda.
template <std::invocable<size_t> F> struct Thunk<F> {
    static auto wrap(ThreadCount /*unused*/, F& f) -> F& { return f; }
};

// Static partition
template <typename F, typename I> struct Thunk<F, StaticPartition<I>> {
    static auto wrap(ThreadCount nthreads, F& f, StaticPartition<I> space) {
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
    static auto wrap(ThreadCount SVS_UNUSED(nthreads), F& f, DynamicPartition<I> space) {
        auto count_ = std::make_unique<std::atomic<uint64_t>>(0
        ); // workaround for atomic being not copyable and movable
        return [&f, space, count = MoC(std::move(count_))](uint64_t tid) mutable {
            size_t grainsize = space.grainsize;
            size_t iterator_size = space.size();
            for (;;) {
                uint64_t i = count.get()->fetch_add(1, std::memory_order_relaxed);
                auto start = grainsize * i;
                if (start >= iterator_size) {
                    return;
                }

                auto stop =
                    std::min(grainsize * (i + 1), static_cast<uint64_t>(iterator_size));
                auto this_range =
                    IteratorPair{std::begin(space) + start, std::begin(space) + stop};
                f(this_range, tid);
            }
        };
    }
};

// Thunk entry point.
template <typename F, typename... Args>
auto wrap(ThreadCount nthreads, F& f, Args&&... args) {
    return Thunk<F, std::decay_t<Args>...>::wrap(nthreads, f, std::forward<Args>(args)...);
}

} // namespace thunks
} // namespace threads
} // namespace svs
