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

#include <atomic>

namespace svs {

namespace detail {
inline void pause() { __builtin_ia32_pause(); }
} // namespace detail

///
/// Lightweight spin lock suitable for low-contention locking.
///
class SpinLock {
  public:
    SpinLock() = default;

    ///
    /// Implement C++ named requirements "Lockable"
    ///
    /// Attempts to aqcquire the lock for the current execution agent without blocking.
    /// Return `true` if the lock was acquired, `false` otherwise.
    ///
    bool try_lock() noexcept {
        bool expected = false;
        return value_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }

    ///
    /// Part of the implementation of the C++ named requirement "BasicLockable".
    ///
    /// Blocks until a lock can be acquired for the current execution agent.
    ///
    void lock() {
        while (!try_lock()) {
            detail::pause();
        }
    }

    ///
    /// Part of the implementation of the C++ named requirement "BasicLockable".
    ///
    /// Releases the non-shared lock held by the execution agent.
    /// Throws no exceptions.
    ///
    void unlock() noexcept { value_.store(false, std::memory_order_release); }

    ///
    /// Return `true` if the lock is held by some (not necessarily the current) execution
    /// agent.
    ///
    bool islocked() const noexcept { return value_.load(std::memory_order_acquire); }

  private:
    std::atomic<bool> value_{false};
};

} // namespace svs
