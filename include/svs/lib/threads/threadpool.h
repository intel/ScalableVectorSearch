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
#include <queue>
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

///
/// @class threadpool_requirements
///
/// ThreadPool
/// ===========
/// An acceptable thread pool should implement two methods:
/// * ``size_t size()``. This method should return the number of threads used in the thread
/// pool.
/// * ``void parallel_for(std::function<void(size_t)> f, size_t n)``. This method should
/// execute ``f``. Here, ``f(i)`` represents a task on the ``i^th`` partition, and ``n``
/// represents the number of partitions that need to be executed.
///

// clang-format off
template <typename Pool>
concept ThreadPool = requires(Pool& pool, const Pool& const_pool, std::function<void(size_t)> f, size_t n) {
    ///
    /// Return the number of threads in the thread pool.
    ///
    { const_pool.size() } -> std::convertible_to<size_t>;

    ///
    /// Run the fundamental function on each thread.
    ///
    pool.parallel_for(std::move(f), n);

};
// clang-format on

template <ThreadPool Pool, typename F> void parallel_for(Pool& pool, F&& f) {
    pool.parallel_for(thunks::wrap(ThreadCount{pool.size()}, f), pool.size());
}

// Current partitioning methods create n partitions where n equals the
// number of threads. Adjust the number of partitions to match the problem size
// if the problem size is smaller than the number of threads.
template <ThreadPool Pool, typename T, typename F>
void parallel_for(Pool& pool, T&& arg, F&& f) {
    if (!arg.empty()) {
        size_t n = std::min(arg.size(), pool.size());
        pool.parallel_for(thunks::wrap(ThreadCount{n}, f, std::forward<T>(arg)), n);
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

    static void parallel_for(std::function<void(size_t)> f, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            f(i);
        }
    }
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

    // Support resize
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

    // TODO: This assignment underutilizes the main thread.
    //       It's okay for now as current partitioning methods set n equals to the number of
    //       threads
    void parallel_for(std::function<void(size_t)> f, size_t n) {
        std::lock_guard lock{*use_mutex_};
        for (size_t i = 0; i < n - 1; ++i) {
            threads_[i % (threads_.size())].assign({&f, i + 1});
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
static_assert(ThreadPool<NativeThreadPool>);

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
        threads::parallel_for(threadpool, [&](uint64_t tid) { slots[tid] = f(tid); });
    });
}
#endif

/////
///// A thread pool that dynamically switches between single-threaded and multi-threaded
/// execution.
///// - If `n == 1`, the task will be executed on the main thread without any locking
/// mechanism.
///// - For `n > 1`, the tasks will be delegated to the internal `NativeThreadPool` for
/// parallel execution.
/////
class SwitchNativeThreadPool {
  public:
    SwitchNativeThreadPool(size_t num_threads)
        : threadpool_{num_threads} {}

    size_t size() const { return threadpool_.size(); }

    void parallel_for(std::function<void(size_t)> f, size_t n) {
        if (n == 1) {
            try {
                f(0);
            } catch (const std::exception& error) {
                threadpool_.manage_exception_during_run(error.what());
            }
        } else {
            threadpool_.parallel_for(std::move(f), n);
        }
    }

  private:
    NativeThreadPool threadpool_;
};

/////
///// A handy reference wrapper for situations where we only want to share a thread pool
/////
template <ThreadPool Pool> class ThreadPoolReferenceWrapper {
  public:
    ThreadPoolReferenceWrapper(Pool& threadpool)
        : threadpool_{threadpool} {}

    size_t size() const { return threadpool_.size(); }

    void parallel_for(std::function<void(size_t)> f, size_t n) {
        threadpool_.parallel_for(std::move(f), n);
    }

  private:
    Pool& threadpool_;
};

/////
///// A thread pool implementation using std::async
/////
class CppAsyncThreadPool {
  public:
    explicit CppAsyncThreadPool(size_t max_async_tasks)
        : _max_async_tasks{max_async_tasks} {}

    void parallel_for(std::function<void(size_t)> f, size_t n) {
        std::vector<std::future<void>> futures;
        futures.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            futures.emplace_back(std::async(
                std::launch::async, [&f](size_t i) { f(i); }, i
            ));
            if (futures.size() == _max_async_tasks) {
                // wait until all async tasks are finished
                std::for_each(futures.begin(), futures.end(), [](std::future<void>& fu) {
                    fu.get();
                });
                futures.clear();
            }
        }

        // wait until all async tasks are finished
        std::for_each(futures.begin(), futures.end(), [](std::future<void>& fu) {
            fu.get();
        });
    }

    size_t size() const { return _max_async_tasks; }

    // support resize
    void resize(size_t max_async_tasks) { _max_async_tasks = max_async_tasks; }

  private:
    size_t _max_async_tasks{1};
};
static_assert(ThreadPool<CppAsyncThreadPool>);

/////
///// A thread pool implementation using a centrialized task queue
/////
class QueueThreadPool {
  public:
    explicit QueueThreadPool(size_t num_threads) {
        threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this]() {
                while (!stop_) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mtx_);
                        while (queue_.empty() && !stop_) {
                            cv_.wait(lock);
                        }
                        if (!queue_.empty()) {
                            task = queue_.front();
                            queue_.pop();
                        }
                    }

                    if (task) {
                        task();
                    }
                }
            });
        }
    }

    QueueThreadPool(QueueThreadPool&&) = delete;
    QueueThreadPool(const QueueThreadPool&) = delete;
    QueueThreadPool& operator=(QueueThreadPool&&) = delete;
    QueueThreadPool& operator=(const QueueThreadPool&) = delete;

    ~QueueThreadPool() {
        shutdown();
        for (auto& t : threads_) {
            t.join();
        }
    }

    template <typename C> std::future<void> insert(C&& task) {
        std::promise<void> prom;
        std::future<void> fu = prom.get_future();
        {
            std::scoped_lock lock(mtx_);
            queue_.push([moc = MoC{std::move(prom)},
                         task = std::forward<C>(task)]() mutable {
                task();
                moc.obj.set_value();
            });
        }
        cv_.notify_one();
        return fu;
    }

    size_t size() const { return threads_.size(); }

    void shutdown() {
        std::scoped_lock lock(mtx_);
        stop_ = true;
        cv_.notify_all();
    }

  private:
    std::vector<std::thread> threads_;
    std::mutex mtx_;
    std::condition_variable cv_;

    bool stop_{false};
    std::queue<std::function<void()>> queue_;
};

/////
///// The wrapper for QueueThreadPool to work on SVS
/////
class QueueThreadPoolWrapper {
  public:
    QueueThreadPoolWrapper(size_t num_threads)
        : threadpool_{std::make_unique<QueueThreadPool>(num_threads)} {}

    void parallel_for(std::function<void(size_t)> f, size_t n) {
        std::vector<std::future<void>> futures;
        futures.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            futures.emplace_back(threadpool_->insert([&f, i]() { f(i); }));
        }

        // wait until all tasks are finished
        for (auto& fu : futures) {
            fu.get();
        }
    }

    size_t size() const { return threadpool_->size(); }

  private:
    std::unique_ptr<QueueThreadPool> threadpool_;
};
static_assert(ThreadPool<QueueThreadPoolWrapper>);

/////
///// Type erasure thread pool implementation
/////

class ThreadPoolInterface {
  public:
    virtual ~ThreadPoolInterface() = default;
    virtual size_t size() const = 0;
    virtual void parallel_for(std::function<void(size_t)>, size_t) = 0;
};

template <ThreadPool Impl> class ThreadPoolImpl : public ThreadPoolInterface {
  public:
    explicit ThreadPoolImpl(Impl&& impl)
        : ThreadPoolInterface{}
        , impl_{std::move(impl)} {}

    size_t size() const override { return impl_.size(); }

    void parallel_for(std::function<void(size_t)> f, size_t n) override {
        impl_.parallel_for(std::move(f), n);
    }

    Impl& get() { return impl_; }

  private:
    Impl impl_;
};

class ThreadPoolHandle {
  public:
    template <ThreadPool Impl>
    explicit ThreadPoolHandle(Impl&& impl)
        requires(!std::is_same_v<Impl, ThreadPoolHandle>) &&
                std::is_rvalue_reference_v<Impl&&>
        : impl_{std::make_unique<ThreadPoolImpl<Impl>>(std::move(impl))} {}

    size_t size() const { return impl_->size(); }

    void parallel_for(std::function<void(size_t)> f, size_t n) {
        impl_->parallel_for(std::move(f), n);
    }

    template <ThreadPool Impl> auto& get() {
        ThreadPoolImpl<Impl>* result = dynamic_cast<ThreadPoolImpl<Impl>*>(impl_.get());
        if (result != nullptr) {
            return result->get();
        } else {
            throw ANNEXCEPTION("Failed to cast to the provided threadpool type");
        }
    }

  private:
    std::unique_ptr<ThreadPoolInterface> impl_;
};

// SVS default threadpool
using DefaultThreadPool = NativeThreadPool;

} // namespace threads
} // namespace svs
