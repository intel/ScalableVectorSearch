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

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>

namespace svs::lib {

///
/// @brief An unbounded, grow-stable vector for single-writer/many-reader use.
///
/// Two-level "lock-free dynamic array" (Dechev et al.). A fixed top-level directory of
/// ``kDirBuckets`` bucket pointers; directory bucket ``k`` is a single contiguous
/// heap array of ``kFirstBucket << k`` elements (the first bucket holds ``kFirstBucket``,
/// each subsequent bucket doubles). Grouping elements into chunks of ``kFirstBucket`` and
/// applying the power-of-two layout to the chunk index gives: ``q = i / kFirstBucket``,
/// ``bucket = floor(log2(q + 1))``, with the bucket's first global index
/// ``kFirstBucket * (2^bucket - 1)``. Sixty-four buckets address far more than any real
/// dataset, so there is no practical size cap.
///
/// The directory array is a fixed member (never relocates), and each bucket array is
/// allocated once and never moved or reallocated. Therefore the address of any element
/// ``i < size()`` is stable for the lifetime of that element — a concurrent reader
/// indexing element ``i`` is unaffected by appends that grow the structure past ``i``.
///
/// Concurrency contract:
/// * **One writer at a time.** ``resize`` / ``push_back`` / ``pop_back`` / ``shrink_to``
///   (the only operations that change the structure) must be serialized by the caller
///   (e.g. under a mutex).
/// * **Many concurrent readers.** ``operator[]`` and ``size`` may run concurrently with a
///   writer's *grow*: a new bucket is allocated and its elements constructed, the bucket
///   pointer is published with a release store, and ``size_`` is bumped last (release). A
///   reader does an acquire load of ``size_`` then an acquire load of the bucket pointer,
///   so for any ``i < size()`` it observes, the bucket and element are fully published.
/// * **Shrink frees storage.** ``shrink_to`` destroys trailing elements and frees buckets
///   that lie entirely above the new size; a reader holding a reference to a freed element
///   would dangle, so the caller must drain readers (e.g. via an exclusive lock) first.
///
/// This mirrors the std::vector subset used by the dynamic Vamana index: ``operator[]``,
/// ``at``, ``size``, ``empty``, ``capacity``, ``resize(n)``, ``resize(n, fill)``,
/// ``push_back``, ``pop_back``, ``shrink_to(n)``.
///
template <typename T> class SegmentedVector {
    static constexpr std::size_t kDirBuckets = 64;
    // Number of elements in the first directory bucket. Bucket ``k`` then holds
    // ``kFirstBucket << k`` elements, so a small first bucket means many tiny
    // allocations near the start while a large one front-loads capacity.
    static constexpr std::size_t kFirstBucket = 1;
    static_assert(
        (kFirstBucket & (kFirstBucket - 1)) == 0, "kFirstBucket must be a power of two"
    );

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
    /// Safe to call concurrently with a writer's grow ``resize``/``push_back``, provided
    /// ``i`` was ``< size()`` as observed by the reader.
    ///
    const_reference operator[](size_type i) const noexcept {
        auto [b, off] = locate_(i);
        return dir_[b].load(std::memory_order_acquire)[off];
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
    /// bucket. With ``m`` buckets this is ``kFirstBucket * (2^m - 1)``.
    size_type capacity() const noexcept { return bucket_first_index_(allocated_buckets_); }

    /// @brief Grow or shrink the logical size. New elements are default-constructed.
    /// Single-writer; concurrent readers safe on grow (see class contract).
    void resize(size_type n) { resize_impl_(n, nullptr); }

    /// @brief Grow or shrink the logical size, filling new elements with ``fill``.
    void resize(size_type n, const T& fill) { resize_impl_(n, &fill); }

    /// @brief Append one element, move-*constructing* it into the new slot.
    ///
    /// Grows logical size by one, allocating a new bucket if needed. The element is
    /// constructed in place via T's move constructor, so types whose move-*assignment* is
    /// unavailable or expensive (e.g. DenseArray, whose move-assign compares allocators)
    /// still work. The element is published (bucket pointer first, then ``size_``) so a
    /// concurrent reader that observes the new ``size()`` sees the constructed value.
    /// Single-writer.
    void push_back(T&& value) {
        size_type i = size_.load(std::memory_order_relaxed);
        auto [b, off] = locate_(i);
        T* bucket = ensure_bucket_(b);
        new (&bucket[off]) T(std::move(value));
        size_.store(i + 1, std::memory_order_release);
    }

    /// @brief Drop the last element, destroying it (logical only; does not free the
    /// bucket). Single-writer; caller must have drained readers if a bucket is later freed.
    void pop_back() {
        size_type i = size_.load(std::memory_order_relaxed);
        if (i > 0) {
            auto [b, off] = locate_(i - 1);
            dir_[b].load(std::memory_order_relaxed)[off].~T();
            size_.store(i - 1, std::memory_order_release);
        }
    }

    /// @brief Shrink to ``n`` elements, destroying the dropped elements and freeing buckets
    /// that lie entirely above ``n``. Single-writer; caller must have drained readers.
    void shrink_to(size_type n) {
        size_type old = size_.load(std::memory_order_relaxed);
        if (n >= old) {
            return;
        }
        // Stop readers from seeing the elements about to be destroyed.
        size_.store(n, std::memory_order_release);
        destroy_range_(n, old);
        free_buckets_above_(n);
    }

  private:
    // Fixed top-level directory. Bucket k (when non-null) is a contiguous heap array of
    // (kFirstBucket << k) elements; the elements with global index < size_ are constructed.
    std::atomic<T*> dir_[kDirBuckets] = {};
    std::atomic<size_type> size_{0};
    size_type allocated_buckets_{0};

    // Number of elements held by bucket ``b`` (= kFirstBucket << b).
    static constexpr size_type bucket_size_(size_type b) noexcept {
        return kFirstBucket << b;
    }

    // First global element index held by bucket ``b`` (= kFirstBucket * (2^b - 1)).
    static constexpr size_type bucket_first_index_(size_type b) noexcept {
        return kFirstBucket * ((size_type{1} << b) - 1);
    }

    // Map element index ``i`` to (bucket, offset-within-bucket). Group elements into
    // chunks of kFirstBucket, then apply the power-of-two bucket layout to the chunk
    // index: q = i / kFirstBucket; bucket = floor(log2(q+1)); the bucket's first global
    // index is kFirstBucket * (2^bucket - 1).
    static constexpr std::pair<size_type, size_type> locate_(size_type i) noexcept {
        size_type q = i / kFirstBucket;
        size_type bucket = static_cast<size_type>(std::bit_width(q + 1)) - 1;
        return {bucket, i - bucket_first_index_(bucket)};
    }

    // Ensure bucket ``b`` is allocated (single-writer) and return its base pointer. The
    // bucket's elements are raw storage until constructed by the caller; the pointer is
    // published with release so readers that later observe a matching size see it.
    T* ensure_bucket_(size_type b) {
        T* bucket = dir_[b].load(std::memory_order_relaxed);
        if (bucket == nullptr) {
            bucket = static_cast<T*>(::operator new[](bucket_size_(b) * sizeof(T)));
            dir_[b].store(bucket, std::memory_order_release);
            if (b + 1 > allocated_buckets_) {
                allocated_buckets_ = b + 1;
            }
        }
        return bucket;
    }

    // Construct elements [from, to) in place (single-writer). ``fill`` is nullptr for
    // default-construction. Allocates buckets as needed.
    void construct_range_(size_type from, size_type to, const T* fill) {
        for (size_type i = from; i < to; ++i) {
            auto [b, off] = locate_(i);
            T* bucket = ensure_bucket_(b);
            if (fill == nullptr) {
                new (&bucket[off]) T();
            } else {
                new (&bucket[off]) T(*fill);
            }
        }
    }

    // Destroy elements [from, to) (single-writer). Does not free buckets.
    void destroy_range_(size_type from, size_type to) {
        for (size_type i = from; i < to; ++i) {
            auto [b, off] = locate_(i);
            dir_[b].load(std::memory_order_relaxed)[off].~T();
        }
    }

    void resize_impl_(size_type n, const T* fill) {
        size_type old = size_.load(std::memory_order_relaxed);
        if (n == old) {
            return;
        }
        if (n < old) {
            // Logical-only shrink (no bucket freeing — use shrink_to for reclamation),
            // but still destroy the dropped elements to run their destructors.
            size_.store(n, std::memory_order_release);
            destroy_range_(n, old);
            return;
        }
        // Grow: construct the new elements, then publish the new size last so a reader
        // that observes it sees fully-constructed elements in published buckets.
        construct_range_(old, n, fill);
        size_.store(n, std::memory_order_release);
    }

    // Free every bucket whose entire index range lies at or above ``n`` (single-writer;
    // readers drained). A bucket straddling ``n`` keeps its allocation.
    void free_buckets_above_(size_type n) {
        for (size_type b = allocated_buckets_; b-- > 0;) {
            if (bucket_first_index_(b) < n) {
                break; // this and all lower buckets contain live (or kept) elements
            }
            T* bucket = dir_[b].load(std::memory_order_relaxed);
            if (bucket != nullptr) {
                ::operator delete[](static_cast<void*>(bucket));
                dir_[b].store(nullptr, std::memory_order_relaxed);
            }
            allocated_buckets_ = b;
        }
    }

    void destroy_all_() {
        size_type n = size_.load(std::memory_order_relaxed);
        destroy_range_(0, n);
        for (size_type b = 0; b < allocated_buckets_; ++b) {
            T* bucket = dir_[b].load(std::memory_order_relaxed);
            if (bucket != nullptr) {
                ::operator delete[](static_cast<void*>(bucket));
                dir_[b].store(nullptr, std::memory_order_relaxed);
            }
        }
        size_.store(0, std::memory_order_relaxed);
        allocated_buckets_ = 0;
    }

    void copy_from_(const SegmentedVector& other) {
        size_type n = other.size_.load(std::memory_order_relaxed);
        for (size_type i = 0; i < n; ++i) {
            auto [b, off] = locate_(i);
            T* bucket = ensure_bucket_(b);
            new (&bucket[off]) T(other[i]);
        }
        size_.store(n, std::memory_order_release);
    }

    void steal_from_(SegmentedVector& other) noexcept {
        for (size_type b = 0; b < kDirBuckets; ++b) {
            dir_[b].store(
                other.dir_[b].load(std::memory_order_relaxed), std::memory_order_relaxed
            );
            other.dir_[b].store(nullptr, std::memory_order_relaxed);
        }
        size_.store(other.size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        allocated_buckets_ = other.allocated_buckets_;
        other.size_.store(0, std::memory_order_relaxed);
        other.allocated_buckets_ = 0;
    }
};

} // namespace svs::lib

namespace svs {
// Opt SegmentedVector into svs::getindex's optional bounds checking.
template <typename T>
inline constexpr bool enable_boundschecking<lib::SegmentedVector<T>> = true;
} // namespace svs
