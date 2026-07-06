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

#include <climits>
#include <condition_variable>
#include <mutex>

namespace svs::lib {

///
/// @brief A mutex supporting shared, upgrade, and exclusive ownership.
///
/// In addition to the shared/exclusive ownership offered by ``std::shared_mutex``, this
/// mutex introduces a third, intermediate "upgrade" ownership level:
///
/// - **Shared** ownership may be held by any number of threads simultaneously. Compatible
///   with other shared owners and with a single upgrade owner.
/// - **Upgrade** ownership may be held by at most one thread at a time. It is compatible
///   with any number of shared owners but excludes other upgrade owners and exclusive
///   owners. An upgrade owner may atomically transition to exclusive ownership via
///   ``unlock_upgrade_and_lock`` (or the ``UpgradeToUniqueLock`` helper).
/// - **Exclusive** ownership may be held by at most one thread at a time and is
///   incompatible with every other kind of ownership.
///
/// The typical use case is a reader that decides, while holding a shared/upgrade lock, that
/// it must perform a write. Because at most one upgrade owner can exist, such an owner can
/// upgrade to exclusive ownership without releasing the lock and racing with another
/// writer.
///
/// This type satisfies the C++ named requirements *Lockable*, *SharedLockable*, and
/// *SharedTimedLockable* is intentionally not provided (no timed operations are exposed).
/// The interface is compatible with ``std::unique_lock`` (exclusive) and
/// ``std::shared_lock`` (shared). Upgrade ownership is managed via ``UpgradeLock``.
///
class UpgradableMutex {
  public:
    UpgradableMutex() = default;
    ~UpgradableMutex() = default;

    // Non-copyable and non-movable.
    UpgradableMutex(const UpgradableMutex&) = delete;
    UpgradableMutex& operator=(const UpgradableMutex&) = delete;
    UpgradableMutex(UpgradableMutex&&) = delete;
    UpgradableMutex& operator=(UpgradableMutex&&) = delete;

    ///// Exclusive ownership

    /// @brief Acquire exclusive ownership, blocking until it is available.
    void lock() {
        std::unique_lock<std::mutex> lock{mutex_};
        // Wait until no other thread holds exclusive or upgrade ownership, then reserve
        // exclusive ownership by setting the write flag. This prevents new shared/upgrade
        // owners from being admitted.
        write_gate_.wait(lock, [this]() {
            return (state_ & (write_entered_ | upgrade_entered_)) == 0;
        });
        state_ |= write_entered_;
        // Drain any remaining shared owners.
        reader_gate_.wait(lock, [this]() { return (state_ & n_readers_) == 0; });
    }

    /// @brief Attempt to acquire exclusive ownership without blocking.
    /// @returns ``true`` if ownership was acquired, ``false`` otherwise.
    bool try_lock() {
        std::lock_guard<std::mutex> lock{mutex_};
        if (state_ != 0) {
            return false;
        }
        state_ = write_entered_;
        return true;
    }

    /// @brief Release exclusive ownership.
    void unlock() {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            state_ = 0;
        }
        write_gate_.notify_all();
    }

    ///// Shared ownership

    /// @brief Acquire shared ownership, blocking until it is available.
    void lock_shared() {
        std::unique_lock<std::mutex> lock{mutex_};
        write_gate_.wait(lock, [this]() {
            return (state_ & write_entered_) == 0 && (state_ & n_readers_) != n_readers_;
        });
        set_readers(readers() + 1);
    }

    /// @brief Attempt to acquire shared ownership without blocking.
    /// @returns ``true`` if ownership was acquired, ``false`` otherwise.
    bool try_lock_shared() {
        std::lock_guard<std::mutex> lock{mutex_};
        if ((state_ & write_entered_) != 0 || (state_ & n_readers_) == n_readers_) {
            return false;
        }
        set_readers(readers() + 1);
        return true;
    }

    /// @brief Release shared ownership.
    void unlock_shared() {
        // Determine which gate to signal before releasing the mutex so the decision is
        // made atomically with the state update, but the actual notify happens after the
        // lock is released. This avoids the notified thread immediately re-contending on
        // mutex_ before it can make progress.
        bool notify_readers = false;
        bool notify_writers = false;
        {
            std::lock_guard<std::mutex> lock{mutex_};
            unsigned num_readers = readers() - 1;
            set_readers(num_readers);
            if ((state_ & write_entered_) != 0) {
                // A writer is waiting to drain the readers.
                notify_readers = (num_readers == 0);
            } else if (num_readers == n_readers_ - 1) {
                // We just freed a reader slot that a would-be reader may be waiting on.
                notify_writers = true;
            }
        }
        if (notify_readers) {
            reader_gate_.notify_one();
        } else if (notify_writers) {
            write_gate_.notify_one();
        }
    }

    ///// Upgrade ownership

    /// @brief Acquire upgrade ownership, blocking until it is available.
    void lock_upgrade() {
        std::unique_lock<std::mutex> lock{mutex_};
        write_gate_.wait(lock, [this]() {
            return (state_ & (write_entered_ | upgrade_entered_)) == 0 &&
                   (state_ & n_readers_) != n_readers_;
        });
        state_ |= upgrade_entered_;
        set_readers(readers() + 1);
    }

    /// @brief Attempt to acquire upgrade ownership without blocking.
    /// @returns ``true`` if ownership was acquired, ``false`` otherwise.
    bool try_lock_upgrade() {
        std::lock_guard<std::mutex> lock{mutex_};
        if ((state_ & (write_entered_ | upgrade_entered_)) != 0 ||
            (state_ & n_readers_) == n_readers_) {
            return false;
        }
        state_ |= upgrade_entered_;
        set_readers(readers() + 1);
        return true;
    }

    /// @brief Release upgrade ownership.
    void unlock_upgrade() {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            // Clear the upgrade flag and decrement the reader count in one write to
            // avoid a redundant read-modify-write on state_.
            state_ = (state_ & ~(upgrade_entered_ | n_readers_)) |
                     ((readers() - 1) & n_readers_);
        }
        write_gate_.notify_all();
    }

    ///// Ownership transitions

    /// @brief Atomically convert upgrade ownership into exclusive ownership.
    ///
    /// Blocks until all other shared owners have released their shared ownership. The
    /// calling thread must already hold upgrade ownership.
    void unlock_upgrade_and_lock() {
        std::unique_lock<std::mutex> lock{mutex_};
        // Drop our own reader slot and the upgrade flag, then reserve exclusive ownership.
        unsigned num_readers = readers() - 1;
        state_ &= ~upgrade_entered_;
        state_ |= write_entered_;
        set_readers(num_readers);
        // Wait for the remaining shared owners to drain.
        reader_gate_.wait(lock, [this]() { return (state_ & n_readers_) == 0; });
    }

    /// @brief Attempt to convert upgrade ownership into exclusive ownership without
    ///     blocking.
    ///
    /// Succeeds only if the calling thread is the sole owner of the mutex.
    /// @returns ``true`` if the transition succeeded, ``false`` otherwise.
    bool try_unlock_upgrade_and_lock() {
        std::lock_guard<std::mutex> lock{mutex_};
        // Success requires that the only reader is our own upgrade ownership.
        if (state_ != (upgrade_entered_ | 1U)) {
            return false;
        }
        state_ = write_entered_;
        return true;
    }

    /// @brief Atomically convert exclusive ownership into upgrade ownership.
    ///
    /// Does not block. The calling thread must already hold exclusive ownership.
    void unlock_and_lock_upgrade() {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            state_ = upgrade_entered_ | 1U;
        }
        write_gate_.notify_all();
    }

    /// @brief Atomically convert upgrade ownership into shared ownership.
    ///
    /// Does not block. The calling thread must already hold upgrade ownership.
    void unlock_upgrade_and_lock_shared() {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            state_ &= ~upgrade_entered_;
        }
        // While upgrade ownership was held, write_entered_ was necessarily 0, so no
        // shared readers were waiting on write_gate_. Only upgrade and exclusive waiters
        // were blocked by upgrade_entered_, and at most one of them can succeed.
        write_gate_.notify_one();
    }

    /// @brief Atomically convert exclusive ownership into shared ownership.
    ///
    /// Does not block. The calling thread must already hold exclusive ownership.
    void unlock_and_lock_shared() {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            state_ = 1U;
        }
        write_gate_.notify_all();
    }

  private:
    // The high bit encodes exclusive ("write") ownership, the next-highest bit encodes
    // upgrade ownership, and the remaining bits count the number of shared owners (an
    // upgrade owner also counts as a shared owner).
    static constexpr unsigned write_entered_ = 1U << (sizeof(unsigned) * CHAR_BIT - 1);
    static constexpr unsigned upgrade_entered_ = write_entered_ >> 1;
    static constexpr unsigned n_readers_ = ~(write_entered_ | upgrade_entered_);

    unsigned readers() const { return state_ & n_readers_; }
    void set_readers(unsigned num_readers) {
        state_ = (state_ & ~n_readers_) | (num_readers & n_readers_);
    }

    std::mutex mutex_{};
    // Notifies threads waiting to acquire shared, upgrade, or exclusive ownership.
    std::condition_variable write_gate_{};
    // Notifies an exclusive/upgrading owner waiting for shared owners to drain.
    std::condition_variable reader_gate_{};
    unsigned state_ = 0;
};

///
/// @brief RAII wrapper managing upgrade ownership of an ``UpgradableMutex``.
///
/// Analogous to ``std::shared_lock`` and ``std::unique_lock``, but for the upgrade
/// ownership level. Supports deferred, try-to-lock, and adopt-lock construction and is
/// movable but not copyable.
///
/// @tparam Mutex The mutex type. Must provide ``lock_upgrade``, ``try_lock_upgrade``, and
///     ``unlock_upgrade``.
///
template <typename Mutex> class UpgradeLock {
  public:
    using mutex_type = Mutex;

    /// @brief Construct without an associated mutex. Owns no lock.
    UpgradeLock() noexcept = default;

    /// @brief Acquire upgrade ownership of ``m``, blocking until available.
    explicit UpgradeLock(mutex_type& m)
        : mutex_{&m}
        , owns_{true} {
        mutex_->lock_upgrade();
    }

    /// @brief Associate with ``m`` without acquiring ownership.
    UpgradeLock(mutex_type& m, std::defer_lock_t) noexcept
        : mutex_{&m}
        , owns_{false} {}

    /// @brief Attempt to acquire upgrade ownership of ``m`` without blocking.
    UpgradeLock(mutex_type& m, std::try_to_lock_t)
        : mutex_{&m}
        , owns_{m.try_lock_upgrade()} {}

    /// @brief Assume that the calling thread already holds upgrade ownership of ``m``.
    UpgradeLock(mutex_type& m, std::adopt_lock_t)
        : mutex_{&m}
        , owns_{true} {}

    UpgradeLock(const UpgradeLock&) = delete;
    UpgradeLock& operator=(const UpgradeLock&) = delete;

    /// @brief Move construct, transferring ownership from ``other``.
    UpgradeLock(UpgradeLock&& other) noexcept
        : mutex_{other.mutex_}
        , owns_{other.owns_} {
        other.mutex_ = nullptr;
        other.owns_ = false;
    }

    /// @brief Move assign, releasing any currently held ownership first.
    UpgradeLock& operator=(UpgradeLock&& other) noexcept {
        if (this != &other) {
            if (owns_) {
                mutex_->unlock_upgrade();
            }
            mutex_ = other.mutex_;
            owns_ = other.owns_;
            other.mutex_ = nullptr;
            other.owns_ = false;
        }
        return *this;
    }

    /// @brief Release upgrade ownership if held.
    ~UpgradeLock() {
        if (owns_) {
            mutex_->unlock_upgrade();
        }
    }

    /// @brief Acquire upgrade ownership, blocking until available.
    void lock() {
        validate();
        mutex_->lock_upgrade();
        owns_ = true;
    }

    /// @brief Attempt to acquire upgrade ownership without blocking.
    /// @returns ``true`` if ownership was acquired, ``false`` otherwise.
    bool try_lock() {
        validate();
        owns_ = mutex_->try_lock_upgrade();
        return owns_;
    }

    /// @brief Release upgrade ownership.
    void unlock() {
        if (!owns_) {
            throw ANNEXCEPTION("Attempting to unlock an UpgradeLock that owns no lock!");
        }
        mutex_->unlock_upgrade();
        owns_ = false;
    }

    /// @brief Disassociate from the mutex without releasing ownership.
    /// @returns A pointer to the associated mutex, or ``nullptr`` if none.
    mutex_type* release() noexcept {
        mutex_type* result = mutex_;
        mutex_ = nullptr;
        owns_ = false;
        return result;
    }

    /// @brief Return the associated mutex, or ``nullptr`` if none.
    mutex_type* mutex() const noexcept { return mutex_; }

    /// @brief Return ``true`` if this object owns upgrade ownership of its mutex.
    bool owns_lock() const noexcept { return owns_; }

    /// @brief Return ``true`` if this object owns upgrade ownership of its mutex.
    explicit operator bool() const noexcept { return owns_; }

  private:
    void validate() const {
        if (mutex_ == nullptr) {
            throw ANNEXCEPTION("UpgradeLock has no associated mutex!");
        }
        if (owns_) {
            throw ANNEXCEPTION("UpgradeLock already owns its lock!");
        }
    }

    mutex_type* mutex_ = nullptr;
    bool owns_ = false;
};

///
/// @brief Scoped helper that temporarily promotes an ``UpgradeLock`` to exclusive
/// ownership.
///
/// On construction, the associated mutex is transitioned from upgrade ownership to
/// exclusive ownership (blocking until other shared owners release). On destruction, the
/// mutex is transitioned back to upgrade ownership. The referenced ``UpgradeLock`` must own
/// upgrade ownership for the lifetime of this object.
///
/// @tparam Mutex The mutex type. Must provide ``unlock_upgrade_and_lock`` and
///     ``unlock_and_lock_upgrade``.
///
template <typename Mutex> class UpgradeToUniqueLock {
  public:
    using mutex_type = Mutex;

    /// @brief Promote ``upgrade_lock`` to exclusive ownership.
    explicit UpgradeToUniqueLock(UpgradeLock<Mutex>& upgrade_lock)
        : source_{&upgrade_lock} {
        if (!source_->owns_lock()) {
            throw ANNEXCEPTION(
                "Cannot promote an UpgradeLock that does not own upgrade ownership!"
            );
        }
        source_->mutex()->unlock_upgrade_and_lock();
    }

    UpgradeToUniqueLock(const UpgradeToUniqueLock&) = delete;
    UpgradeToUniqueLock& operator=(const UpgradeToUniqueLock&) = delete;
    UpgradeToUniqueLock(UpgradeToUniqueLock&&) = delete;
    UpgradeToUniqueLock& operator=(UpgradeToUniqueLock&&) = delete;

    /// @brief Demote back to upgrade ownership.
    ~UpgradeToUniqueLock() { source_->mutex()->unlock_and_lock_upgrade(); }

  private:
    UpgradeLock<Mutex>* source_;
};

} // namespace svs::lib
