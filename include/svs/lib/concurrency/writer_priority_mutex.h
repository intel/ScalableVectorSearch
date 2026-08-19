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

#include "svs/lib/exception.h"

#include <condition_variable>
#include <cstddef>
#include <mutex>

#if defined(__GLIBC__)
#define SVS_WRITER_PRIORITY_MUTEX_PTHREAD 1
#include <pthread.h>
#else
#define SVS_WRITER_PRIORITY_MUTEX_PTHREAD 0
#endif

namespace svs {

///
/// @brief A shared mutex that will not starve its writer.
///
/// Drop-in for ``std::shared_mutex`` (same member names, so ``std::shared_lock`` and
/// ``std::unique_lock`` work unchanged), but writer-preferring: once a thread is waiting in
/// ``lock()``, newly-arriving ``lock_shared()`` callers queue behind it instead of jumping
/// ahead.
///
/// For ``svs::concurrent::MutableVamanaIndex`` this is a correctness requirement, not a
/// micro-optimization. That index holds this lock shared for the whole duration of every
/// search and exclusively for the brief windows where the dataset's capacity grows. Under
/// continuous query load from N threads there is essentially always at least one reader
/// inside the lock, so with libstdc++'s default reader-preferring ``std::shared_mutex``
/// (``PTHREAD_RWLOCK_PREFER_READER_NP``) the writer never acquires it and insertion hangs
/// forever. That was observed, not theorized: 8 query threads spinning at 100% CPU with the
/// inserting thread parked on a futex indefinitely.
///
/// It is worth being explicit about what this buys and what it costs. It buys liveness for
/// the writer. It costs a full query stall on every capacity growth, because the writer
/// must wait for all in-flight searches to drain. That stall is the price of leaving
/// ``svs::data::SimpleData<..., Blocked<...>>`` untouched -- see the discussion in
/// ``svs/concurrent/mutable_vamana_index.h``.
///
/// Two implementations, selected on whether glibc's non-portable rwlock attribute is
/// available:
///
/// * **glibc:** set ``PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP`` on the rwlock, so
///   ``lock_shared`` keeps the same atomic fast path ``std::shared_mutex`` has.
/// * **Everything else** (notably macOS, which has no ``pthread_rwlockattr_setkind_np``):
///   a condition-variable implementation. Correct, but it takes an uncontended
///   ``std::mutex`` on every ``lock_shared``, which is measurably more expensive on the
///   search path.
///
class WriterPriorityMutex {
  public:
#if SVS_WRITER_PRIORITY_MUTEX_PTHREAD
    WriterPriorityMutex() {
        pthread_rwlockattr_t attr;
        if (pthread_rwlockattr_init(&attr) != 0) {
            throw ANNEXCEPTION("Failed to initialize rwlock attributes!");
        }
        // The whole point of this class. Note glibc spells the writer-preferring policy
        // "nonrecursive": a thread holding the lock shared must not try to upgrade.
        pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
        const int rc = pthread_rwlock_init(&lock_, &attr);
        pthread_rwlockattr_destroy(&attr);
        if (rc != 0) {
            throw ANNEXCEPTION("Failed to initialize rwlock!");
        }
    }

    ~WriterPriorityMutex() { pthread_rwlock_destroy(&lock_); }
#else
    WriterPriorityMutex() = default;
    ~WriterPriorityMutex() = default;
#endif

    // Neither copyable nor movable, matching ``std::shared_mutex``.
    WriterPriorityMutex(const WriterPriorityMutex&) = delete;
    WriterPriorityMutex& operator=(const WriterPriorityMutex&) = delete;
    WriterPriorityMutex(WriterPriorityMutex&&) = delete;
    WriterPriorityMutex& operator=(WriterPriorityMutex&&) = delete;

#if SVS_WRITER_PRIORITY_MUTEX_PTHREAD

    void lock() { pthread_rwlock_wrlock(&lock_); }
    bool try_lock() { return pthread_rwlock_trywrlock(&lock_) == 0; }
    void unlock() { pthread_rwlock_unlock(&lock_); }

    void lock_shared() { pthread_rwlock_rdlock(&lock_); }
    bool try_lock_shared() { return pthread_rwlock_tryrdlock(&lock_) == 0; }
    void unlock_shared() { pthread_rwlock_unlock(&lock_); }

  private:
    pthread_rwlock_t lock_;

#else

    void lock() {
        std::unique_lock guard{mutex_};
        // Register before waiting: this is what makes arriving readers queue behind us.
        ++waiting_writers_;
        condition_.wait(guard, [this] { return !writer_active_ && readers_ == 0; });
        --waiting_writers_;
        writer_active_ = true;
    }

    bool try_lock() {
        std::lock_guard guard{mutex_};
        if (writer_active_ || readers_ != 0) {
            return false;
        }
        writer_active_ = true;
        return true;
    }

    void unlock() {
        {
            std::lock_guard guard{mutex_};
            writer_active_ = false;
        }
        condition_.notify_all();
    }

    void lock_shared() {
        std::unique_lock guard{mutex_};
        condition_.wait(guard, [this] { return !writer_active_ && waiting_writers_ == 0; });
        ++readers_;
    }

    bool try_lock_shared() {
        std::lock_guard guard{mutex_};
        if (writer_active_ || waiting_writers_ != 0) {
            return false;
        }
        ++readers_;
        return true;
    }

    void unlock_shared() {
        bool last_reader = false;
        {
            std::lock_guard guard{mutex_};
            last_reader = (--readers_ == 0);
        }
        if (last_reader) {
            condition_.notify_all();
        }
    }

  private:
    std::mutex mutex_{};
    std::condition_variable condition_{};
    std::size_t readers_{0};
    std::size_t waiting_writers_{0};
    bool writer_active_{false};

#endif
};

} // namespace svs
