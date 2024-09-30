/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
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
