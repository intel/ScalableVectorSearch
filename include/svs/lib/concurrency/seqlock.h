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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace svs {

///
/// @brief Per-element sequence lock counter for reader-writer synchronization.
///
/// Uses a uint8_t counter: odd values indicate a write in progress, even values indicate
/// a stable state.
///
/// **Writer-writer serialization is the caller's responsibility.** Only one writer
/// at a time may call ``begin_write``/``end_write`` on a given counter. Use an external
/// lock (e.g., per-node ``SpinLock``) to serialize concurrent writers to the same element.
///
class SeqLockCounter {
    using counter_type = uint8_t;

  public:
    SeqLockCounter() = default;

    SeqLockCounter(const SeqLockCounter& other)
        : seq_(other.seq_.load(std::memory_order_relaxed)) {}

    SeqLockCounter& operator=(const SeqLockCounter& other) {
        seq_.store(other.seq_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    SeqLockCounter(SeqLockCounter&& other) noexcept
        : seq_(other.seq_.load(std::memory_order_relaxed)) {}

    SeqLockCounter& operator=(SeqLockCounter&& other) noexcept {
        seq_.store(other.seq_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    ///
    /// @brief Begin a write operation. Returns the pre-write sequence value.
    ///
    /// Increments the counter to an odd value, signaling to readers that a write is in
    /// progress. The returned value must be passed to ``end_write``.
    ///
    counter_type begin_write() {
        auto seq = seq_.load(std::memory_order_relaxed);
        seq_.store(seq + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        return seq;
    }

    ///
    /// @brief End a write operation.
    ///
    /// @param seq The value returned by the corresponding ``begin_write`` call.
    ///
    /// Increments the counter to an even value, signaling that the write is complete
    /// and data is consistent.
    ///
    void end_write(counter_type seq) { seq_.store(seq + 2, std::memory_order_release); }

    ///
    /// @brief Begin a read operation.
    ///
    /// @returns The current sequence value if it is even (no write in progress),
    ///          or ``std::nullopt`` if a write is in progress.
    ///
    /// The returned value (if present) must be passed to ``read_validate`` after the
    /// read is complete.
    ///
    std::optional<counter_type> read_begin() const {
        auto seq = seq_.load(std::memory_order_acquire);
        if (seq % 2 > 0) {
            return std::nullopt;
        }
        return seq;
    }

    ///
    /// @brief Validate that no write occurred during the read.
    ///
    /// @param seq The value returned by ``read_begin``.
    ///
    /// @returns ``true`` if the data read between ``read_begin`` and ``read_validate``
    ///          is consistent (no concurrent write occurred).
    ///
    bool read_validate(counter_type seq) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return seq_.load(std::memory_order_relaxed) == seq;
    }

  private:
    std::atomic<counter_type> seq_{0};
};

///
/// @brief Array of SeqLock counters, one per element (e.g., one per graph node).
///
class SeqLockArray {
  public:
    SeqLockArray() = default;
    explicit SeqLockArray(size_t n)
        : counters_(n) {}

    SeqLockCounter& operator[](size_t i) { return counters_[i]; }
    const SeqLockCounter& operator[](size_t i) const { return counters_[i]; }

    void resize(size_t n) { counters_.resize(n); }
    size_t size() const { return counters_.size(); }

  private:
    std::vector<SeqLockCounter> counters_;
};

} // namespace svs
