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

#include "svs/lib/boundscheck.h"

#include <atomic>
#include <bit>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace svs::lib {

///
/// @brief An unbounded, grow-stable vector for single-writer/many-reader use.
///
/// Elements live in fixed-size, heap-allocated segments that are never moved once
/// allocated. The *directory* of segment pointers uses a power-of-two bucket layout
/// (Dechev et al. "lock-free dynamic array") so it, too, never relocates: a constant
/// ``kDirBuckets`` (= 64) top-level array of bucket pointers, where directory bucket
/// ``k`` holds ``1 << k`` segment pointers. Segment index ``s`` lives in directory
/// bucket ``floor(log2(s + 1))``. Sixty-four buckets address ``2^64`` segments, so
/// there is no practical size cap.
///
/// Concurrency contract:
/// * **One writer at a time.** ``resize`` / ``shrink_to`` (the only operations that
///   change the structure) must be serialized by the caller (e.g. under a mutex).
/// * **Many concurrent readers.** ``operator[]`` and ``size`` may run concurrently with
///   a writer's ``resize`` *grow*: new segments and directory buckets are published with
///   release stores, ``size_`` is bumped last, and readers acquire-load. A reader that
///   only touches indices ``< size()`` it observed never sees a half-published segment.
/// * **Shrink frees storage.** ``shrink_to`` destroys trailing segments; a reader holding
///   a reference to a freed segment would dangle. The caller must drain readers (e.g. via
///   an exclusive lock) before calling ``shrink_to``.
///
/// This mirrors the std::vector subset used by the dynamic Vamana index: ``operator[]``,
/// ``size``, ``capacity``, ``resize(n)``, ``resize(n, fill)``, ``shrink_to(n)``.
///
template <typename T, std::size_t SegmentSize = 512> class SegmentedVector {
    static_assert(SegmentSize > 0, "SegmentSize must be positive");
    static constexpr std::size_t kDirBuckets = 64;

  public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;

    SegmentedVector() = default;
    explicit SegmentedVector(size_type n) { resize(n); }
    SegmentedVector(size_type n, const T& fill) { resize(n, fill); }

    SegmentedVector(const SegmentedVector& other) { copy_from_(other); }
    SegmentedVector& operator=(const SegmentedVector& other) {
        if (this != &other) {
            destroy_all_();
            copy_from_(other);
        }
        return *this;
    }

    SegmentedVector(SegmentedVector&& other) noexcept { steal_from_(other); }
    SegmentedVector& operator=(SegmentedVector&& other) noexcept {
        if (this != &other) {
            destroy_all_();
            steal_from_(other);
        }
        return *this;
    }

    ~SegmentedVector() { destroy_all_(); }

    ///
    /// @brief Access element ``i``. Precondition: ``i < size()``.
    ///
    /// Safe to call concurrently with a writer's grow ``resize``, provided ``i`` was
    /// ``< size()`` as observed by the reader.
    ///
    const_reference operator[](size_type i) const noexcept {
        auto [b, bo] = locate_segment_(i / SegmentSize);
        SegmentPtr* bucket = dir_[b].load(std::memory_order_acquire);
        T* seg = bucket[bo].ptr.load(std::memory_order_acquire);
        return seg[i % SegmentSize];
    }
    reference operator[](size_type i) noexcept {
        const auto& self = *this;
        return const_cast<reference>(self[i]);
    }

    /// @brief Bounds-checked access (used by ``svs::getindex`` when bounds checking is on).
    reference at(size_type i) {
        if (i >= size()) {
            throw std::out_of_range("SegmentedVector::at index out of range");
        }
        return (*this)[i];
    }
    const_reference at(size_type i) const {
        if (i >= size()) {
            throw std::out_of_range("SegmentedVector::at index out of range");
        }
        return (*this)[i];
    }

    size_type size() const noexcept { return size_.load(std::memory_order_acquire); }
    bool empty() const noexcept { return size() == 0; }

    /// @brief Logical capacity: number of elements addressable without allocating a new
    /// segment. Equals ``allocated_segments * SegmentSize``.
    size_type capacity() const noexcept { return allocated_segments_ * SegmentSize; }

    /// @brief Grow or shrink the logical size. New elements are default-constructed.
    /// Single-writer; concurrent readers safe on grow (see class contract).
    void resize(size_type n) { resize_impl_(n, nullptr); }

    /// @brief Grow or shrink the logical size, filling new elements with ``fill``.
    void resize(size_type n, const T& fill) { resize_impl_(n, &fill); }

    /// @brief Append one element, move-*constructing* it into the new slot.
    ///
    /// Grows logical size by one, allocating a new segment if needed. The slot is
    /// constructed in place via T's move constructor (the slot's default-constructed value
    /// is destroyed first), so types whose move-*assignment* is unavailable or expensive
    /// (e.g. DenseArray, whose move-assign compares allocators) still work. The new slot is
    /// published (segment first, then ``size_``) so a concurrent reader that observes the
    /// new ``size()`` sees the constructed value. Single-writer.
    void push_back(T&& value) {
        size_type i = size_.load(std::memory_order_relaxed);
        size_type seg = i / SegmentSize;
        if (seg >= allocated_segments_) {
            ensure_segment_(seg, nullptr);
            allocated_segments_ = seg + 1;
        }
        T* slot = slot_ptr_(i);
        slot->~T();
        new (slot) T(std::move(value));
        size_.store(i + 1, std::memory_order_release);
    }

    /// @brief Drop the last element (logical only; does not free the segment).
    /// Single-writer; caller must have drained readers if a segment is later freed.
    void pop_back() {
        size_type i = size_.load(std::memory_order_relaxed);
        if (i > 0) {
            size_.store(i - 1, std::memory_order_release);
        }
    }

    /// @brief Shrink to ``n`` elements, freeing now-empty trailing segments.
    /// Single-writer and the caller must have drained concurrent readers.
    void shrink_to(size_type n) {
        if (n > size_.load(std::memory_order_relaxed)) {
            return;
        }
        size_.store(n, std::memory_order_release);
        // Free whole segments strictly above the one containing the last live element.
        size_type needed_segments = (n + SegmentSize - 1) / SegmentSize;
        free_segments_from_(needed_segments);
    }

  private:
    struct SegmentPtr {
        std::atomic<T*> ptr{nullptr};
    };

    // Directory bucket ``k`` holds ``1 << k`` SegmentPtr entries.
    std::atomic<SegmentPtr*> dir_[kDirBuckets] = {};
    std::atomic<size_type> size_{0};
    size_type allocated_segments_{0};

    static constexpr std::pair<size_type, size_type> locate_segment_(size_type seg
    ) noexcept {
        // Segment ``seg`` lives in directory bucket floor(log2(seg+1)); the offset within
        // that bucket is (seg+1) - 2^bucket. Bucket ``k`` has capacity ``1 << k``.
        size_type p = seg + 1;
        size_type bucket = static_cast<size_type>(std::bit_width(p)) - 1;
        size_type offset = p - (size_type{1} << bucket);
        return {bucket, offset};
    }

    // Raw address of slot ``i`` (no atomic publication semantics; single-writer use).
    T* slot_ptr_(size_type i) noexcept {
        auto [b, bo] = locate_segment_(i / SegmentSize);
        SegmentPtr* bucket = dir_[b].load(std::memory_order_relaxed);
        T* seg = bucket[bo].ptr.load(std::memory_order_relaxed);
        return &seg[i % SegmentSize];
    }

    // Ensure directory bucket ``bucket`` is allocated (single-writer).
    SegmentPtr* ensure_dir_bucket_(size_type bucket) {
        SegmentPtr* b = dir_[bucket].load(std::memory_order_relaxed);
        if (b == nullptr) {
            b = new SegmentPtr[size_type{1} << bucket];
            dir_[bucket].store(b, std::memory_order_release);
        }
        return b;
    }

    // Allocate + construct segment ``seg`` if not present and publish it (single-writer).
    // ``fill`` is nullptr for default-construction.
    void ensure_segment_(size_type seg, const T* fill) {
        auto [bucket, offset] = locate_segment_(seg);
        SegmentPtr* b = ensure_dir_bucket_(bucket);
        if (b[offset].ptr.load(std::memory_order_relaxed) != nullptr) {
            return;
        }
        T* seg_data = static_cast<T*>(::operator new[](SegmentSize * sizeof(T)));
        if (fill == nullptr) {
            for (size_type j = 0; j < SegmentSize; ++j) {
                new (&seg_data[j]) T();
            }
        } else {
            for (size_type j = 0; j < SegmentSize; ++j) {
                new (&seg_data[j]) T(*fill);
            }
        }
        b[offset].ptr.store(seg_data, std::memory_order_release);
    }

    void resize_impl_(size_type n, const T* fill) {
        size_type old_size = size_.load(std::memory_order_relaxed);
        if (n <= old_size) {
            // Pure logical shrink (no segment freeing — use shrink_to for that).
            size_.store(n, std::memory_order_release);
            return;
        }
        size_type needed_segments = (n + SegmentSize - 1) / SegmentSize;
        for (size_type s = allocated_segments_; s < needed_segments; ++s) {
            ensure_segment_(s, fill);
        }
        // For the fill variant, also fill the new tail of the last previously-allocated
        // segment (elements in [old_size, allocated_segments_*SegmentSize)). New segments
        // were already fully fill-constructed above; default-constructed segments need no
        // tail handling.
        if (fill != nullptr && old_size < allocated_segments_ * SegmentSize) {
            size_type stop = std::min(n, allocated_segments_ * SegmentSize);
            for (size_type i = old_size; i < stop; ++i) {
                (*this)[i] = *fill;
            }
        }
        if (needed_segments > allocated_segments_) {
            allocated_segments_ = needed_segments;
        }
        // Publish the new size last so readers that observe it see fully-built segments.
        size_.store(n, std::memory_order_release);
    }

    // Free all segments with index >= ``from`` and release any directory buckets that
    // become entirely empty. Single-writer; readers must be drained.
    void free_segments_from_(size_type from) {
        for (size_type s = from; s < allocated_segments_; ++s) {
            auto [bucket, offset] = locate_segment_(s);
            SegmentPtr* b = dir_[bucket].load(std::memory_order_relaxed);
            if (b == nullptr) {
                continue;
            }
            T* seg_data = b[offset].ptr.load(std::memory_order_relaxed);
            if (seg_data != nullptr) {
                for (size_type j = 0; j < SegmentSize; ++j) {
                    seg_data[j].~T();
                }
                ::operator delete[](static_cast<void*>(seg_data));
                b[offset].ptr.store(nullptr, std::memory_order_relaxed);
            }
        }
        // Release directory buckets whose entire range is now above ``from``.
        for (size_type bucket = 0; bucket < kDirBuckets; ++bucket) {
            SegmentPtr* b = dir_[bucket].load(std::memory_order_relaxed);
            if (b == nullptr) {
                continue;
            }
            size_type bucket_first_seg = (size_type{1} << bucket) - 1;
            if (bucket_first_seg >= from) {
                delete[] b;
                dir_[bucket].store(nullptr, std::memory_order_relaxed);
            }
        }
        if (from < allocated_segments_) {
            allocated_segments_ = from;
        }
    }

    void destroy_all_() {
        free_segments_from_(0);
        size_.store(0, std::memory_order_relaxed);
        allocated_segments_ = 0;
    }

    void copy_from_(const SegmentedVector& other) {
        size_type n = other.size_.load(std::memory_order_relaxed);
        for (size_type i = 0; i < n; ++i) {
            // Append via copy-construction so element types whose copy/move-assignment is
            // unavailable still work (mirrors push_back's in-place construction).
            size_type seg = i / SegmentSize;
            if (seg >= allocated_segments_) {
                ensure_segment_(seg, nullptr);
                allocated_segments_ = seg + 1;
            }
            T* slot = slot_ptr_(i);
            slot->~T();
            new (slot) T(other[i]);
        }
        size_.store(n, std::memory_order_release);
    }

    void steal_from_(SegmentedVector& other) noexcept {
        for (size_type bucket = 0; bucket < kDirBuckets; ++bucket) {
            dir_[bucket].store(
                other.dir_[bucket].load(std::memory_order_relaxed),
                std::memory_order_relaxed
            );
            other.dir_[bucket].store(nullptr, std::memory_order_relaxed);
        }
        size_.store(other.size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        allocated_segments_ = other.allocated_segments_;
        other.size_.store(0, std::memory_order_relaxed);
        other.allocated_segments_ = 0;
    }
};

} // namespace svs::lib

namespace svs {
// Opt SegmentedVector into svs::getindex's optional bounds checking.
template <typename T, std::size_t SegmentSize>
inline constexpr bool enable_boundschecking<lib::SegmentedVector<T, SegmentSize>> = true;
} // namespace svs
