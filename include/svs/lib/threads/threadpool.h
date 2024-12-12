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
#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <vector>

#include "svs/lib/numa.h"
#include "svs/lib/threads/thread.h"
#include "svs/lib/threads/thunks.h"
#include "svs/lib/threads/types.h"
#include "svs/lib/tuples.h"
#include "svs/third-party/fmt.h"

namespace svs {
namespace threads {

// clang-format off
template <typename Pool>
concept ThreadPool = requires(Pool& pool, const Pool& const_pool, FunctionRef f) {
    ///
    /// Return the number of threads in the thread pool.
    ///
    { const_pool.size() } -> std::convertible_to<size_t>;

    ///
    /// Run the fundamental function on each thread.
    ///
    pool.run(f);
};

template <typename Pool>
concept ResizeableThreadPool = requires(Pool& pool, size_t sz) {
    requires(ThreadPool<Pool>);

    ///
    /// Change the number of threads in the pool.
    ///
    pool.resize(sz);
};
// clang-format on

template <ThreadPool Pool, typename F> void run(Pool& pool, F&& f) {
    auto f_wrapped = thunks::wrap(ThreadCount{pool.size()}, f);
    pool.run(FunctionRef(f_wrapped));
}

template <ThreadPool Pool, typename T, typename F> void run(Pool& pool, T&& arg, F&& f) {
    if (!arg.empty()) {
        auto f_wrapped = thunks::wrap(ThreadCount{pool.size()}, f, std::forward<T>(arg));
        pool.run(FunctionRef(f_wrapped));
    }
}

/////
///// Implementations
/////

///
/// Low-overhead threadpool that runs assignments on a single thread.
///
class SequentialThreadPool {
  public:
    SequentialThreadPool() = default;
    static constexpr size_t size() { return 1; }
    static void run(FunctionRef f) { f(0); }
};
static_assert(ThreadPool<SequentialThreadPool>);

///
/// Construct a Thread with a specified spin time.
///
class DefaultBuilder {
  private:
    uint64_t spin_time_;

  public:
    explicit DefaultBuilder(uint64_t spin_time = default_spintime())
        : spin_time_{spin_time} {}
    Thread build(uint64_t /*tid*/) const { return Thread{spin_time_}; }
};

SVS_VALIDATE_BOOL_ENV(SVS_ENABLE_NUMA);
#if SVS_ENABLE_NUMA
///
/// Construct main threads for each socket of a multi-socket system.
/// This should be used to partition work across sockets with `IntraNUMABuilder`s used
/// to distribute work for within each socket.
///
class InterNUMABuilder {
  private:
    uint64_t spin_time_;
    uint64_t num_nodes_;

  public:
    explicit InterNUMABuilder(
        uint64_t spin_time = default_spintime(), uint64_t num_nodes = numa::num_nodes()
    )
        : spin_time_{spin_time}
        , num_nodes_{num_nodes} {}

    Thread build(uint64_t node) const {
        return Thread{spin_time_, [node]() { return numa::NodeBind{node}; }};
    }
};
#endif

template <typename Builder> class NativeThreadPoolBase {
  private:
    Builder builder_;
    std::vector<Thread> threads_{};
    // Make a unique-ptr to allow the thread pool to be moved.
    // Mutexes cannot be moved or copied.
    std::unique_ptr<std::mutex> use_mutex_{std::make_unique<std::mutex>()};

  public:
    // Allocate `num_threads - 1` threads since the main thread participates in the work
    // as well.
    template <typename... Args>
    explicit NativeThreadPoolBase(uint64_t num_threads = 1, Args&&... args)
        : builder_{std::forward<Args>(args)...} {
        for (size_t tid = 0; tid < num_threads - 1; ++tid) {
            threads_.push_back(builder_.build(tid + 1));
        }
    }

    size_t size() const { return threads_.size() + 1; }

    void resize(size_t new_size) {
        new_size = std::max(new_size, size_t{1});
        std::lock_guard lock{*use_mutex_};
        // If we're shrinking, it's okay to let existing threads' destructors run.
        if (new_size < size()) {
            threads_.resize(new_size - 1);
        } else {
            while (new_size > size()) {
                threads_.push_back(builder_.build(size()));
            }
        }
    }

    void run(FunctionRef f) {
        std::lock_guard lock{*use_mutex_};
        for (size_t i = 0; i < threads_.size(); ++i) {
            threads_[i].assign({f, i + 1});
        }
        // Run on the main function.
        try {
            f(0);
        } catch (const std::exception& error) { manage_exception_during_run(error.what()); }

        // Wait until all threads are done.
        // If any thread fails, then we're throwing.
        for (size_t i = 0; i < threads_.size(); ++i) {
            auto& thread = threads_[i];
            thread.wait();
            if (!thread.is_okay()) {
                manage_exception_during_run();
            }
        }
    }

  private:
    void manage_exception_during_run(const std::string& thread_0_message = {}) {
        auto message = std::string{};
        auto inserter = std::back_inserter(message);
        if (!thread_0_message.empty()) {
            fmt::format_to(inserter, "Thread 0: {}\n", thread_0_message);
        }

        // Manage all other exceptions thrown, restarting crashed threads.
        for (size_t i = 0; i < threads_.size(); ++i) {
            auto& thread = threads_[i];
            thread.wait();
            if (!thread.is_okay()) {
                try {
                    thread.unsafe_get_exception();
                } catch (const std::exception& error) {
                    fmt::format_to(inserter, "Thread {}: {}\n", i + 1, error.what());
                }
                // Restart the thread.
                threads_[i].shutdown();
                threads_[i] = builder_.build(i + 1);
            }
        }
        throw ThreadingException{std::move(message)};
    }
};

using NativeThreadPool = NativeThreadPoolBase<DefaultBuilder>;
// Ensure that we satisfy the requirements for a threadpool.
static_assert(ResizeableThreadPool<NativeThreadPool>);

SVS_VALIDATE_BOOL_ENV(SVS_ENABLE_NUMA);
#if SVS_ENABLE_NUMA
/////
///// Numa Stuff
/////

using InterNUMAThreadPool = NativeThreadPoolBase<InterNUMABuilder>;

inline InterNUMAThreadPool
internuma_threadpool(size_t num_nodes = numa::num_nodes(), size_t spintime = 10) {
    return InterNUMAThreadPool{num_nodes, spintime};
}

template <typename F>
auto create_on_nodes(InterNUMAThreadPool& threadpool, F&& f)
    -> numa::NumaLocal<std::result_of_t<F(size_t)>> {
    using RetType = std::result_of_t<F(size_t)>;
    return numa::NumaLocal<RetType>(threadpool.size(), [&](auto& slots) {
        assert(slots.size() == threadpool.size());
        threads::run(threadpool, [&](uint64_t tid) { slots[tid] = f(tid); });
    });
}
#endif

} // namespace threads
} // namespace svs
