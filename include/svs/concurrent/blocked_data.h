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

///
/// @file
/// @brief Grow-stable blocked storage for the concurrent Vamana index.
///
/// ``svs::data::SimpleData<T, Extent, Blocked<Alloc>>`` keeps its fixed-size blocks in a
/// ``std::vector``, so appending a block may reallocate the outer directory. A lock-free
/// reader evaluating ``blocks_[block_id]`` while a writer grows the dataset would then read
/// through a freed pointer.
///
/// This header supplies the same storage with a grow-stable outer directory
/// (``lib::SegmentedVector``), selected by a distinct allocator tag so that **no existing
/// dataset type changes**: ``SegmentedBlocked<Alloc>`` picks up a partial specialization of
/// ``svs::data::SimpleData`` that is otherwise a line-for-line copy of the ``Blocked``
/// specialization. Element addressing ``(block_id, data_id)`` is unchanged; only the outer
/// block directory becomes a two-level lock-free array.
///
/// Because the result is still a ``svs::data::SimpleData``, every generic facility written
/// against that template — the dataset concepts, the ``extensions`` customization points,
/// ``compact_data``, and the save/load serializer — applies unmodified.
///

#include "svs/core/data/simple.h"
#include "svs/lib/segmented_vector.h"

#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

namespace svs::index::vamana::concurrent {

///
/// @brief Allocator tag selecting grow-stable blocked storage.
///
/// Behaves exactly like ``svs::data::Blocked<Alloc>`` — same blocking parameters, same
/// inner allocator — and differs only in which ``SimpleData`` specialization it selects.
///
template <typename Alloc> class SegmentedBlocked : public svs::data::Blocked<Alloc> {
  public:
    using parent_type = svs::data::Blocked<Alloc>;
    using allocator_type = Alloc;
    using value_type = typename std::allocator_traits<allocator_type>::value_type;

    constexpr SegmentedBlocked() = default;
    explicit SegmentedBlocked(const allocator_type& alloc)
        : parent_type{alloc} {}
    explicit SegmentedBlocked(const svs::data::BlockingParameters& parameters)
        : parent_type{parameters} {}
    explicit SegmentedBlocked(
        const svs::data::BlockingParameters& parameters, const allocator_type& alloc
    )
        : parent_type{parameters, alloc} {}

    // Enable rebinding of allocators.
    template <typename U> friend class SegmentedBlocked;
    template <typename U>
    SegmentedBlocked(const SegmentedBlocked<U>& other)
        : parent_type{other.parameters(), allocator_type(other.get_allocator())} {}
};

template <typename Alloc> inline constexpr bool is_segmented_blocked_v = false;
template <typename Alloc>
inline constexpr bool is_segmented_blocked_v<SegmentedBlocked<Alloc>> = true;

} // namespace svs::index::vamana::concurrent

namespace svs::data {

// ``SegmentedBlocked`` is a blocked allocator for the purposes of the library's
// blocked/non-blocked dispatch, exactly as ``Blocked`` is.
template <typename Alloc>
inline constexpr bool
    is_blocked_v<svs::index::vamana::concurrent::SegmentedBlocked<Alloc>> = true;

} // namespace svs::data

namespace svs::lib::detail {

// Allow rebinding of allocators through the SegmentedBlocked wrapper.
template <typename To, typename Alloc>
struct AllocatorRebinder<To, svs::index::vamana::concurrent::SegmentedBlocked<Alloc>> {
    using type =
        svs::index::vamana::concurrent::SegmentedBlocked<rebind_allocator_t<To, Alloc>>;
};

} // namespace svs::lib::detail

namespace svs::data {

namespace detail {

///
/// @brief ``SimpleData`` specialization with a grow-stable block directory.
///
/// A line-for-line copy of the ``SimpleData<T, Extent, Blocked<Alloc>>`` specialization
/// with two changes:
///
/// 1. ``blocks_`` is a ``lib::SegmentedVector`` rather than a ``std::vector``, so appending
///    a block never relocates the existing block wrappers (or the heap buffers they point
///    at). A concurrent lock-free reader subscripting ``blocks_[block_id]`` is therefore
///    safe against a writer growing the dataset.
/// 2. ``add_block`` move-constructs into the new slot (``push_back``) instead of
///    ``emplace_back``, because ``DenseArray``'s move-*assignment* compares allocators.
///
/// Shrinking still frees storage: ``drop_block`` destroys the trailing block, releasing its
/// heap buffer. A reader inside that block would dangle, so the owning index must drain
/// readers (exclusive lock) before shrinking — the same obligation the ``Blocked``
/// specialization carries.
///
template <typename Alloc>
struct BlockedDataTraits<svs::index::vamana::concurrent::SegmentedBlocked<Alloc>> {
    using inner_allocator_type = Alloc;
    template <typename A>
    using rebind_allocator = svs::index::vamana::concurrent::SegmentedBlocked<A>;
    // Grow-stable directory of fixed-size blocks: appending a block never relocates the
    // existing block wrappers (or the heap buffers they point to), so a concurrent
    // lock-free reader subscripting blocks_[block_id] is safe against a writer growing the
    // dataset. Element addressing (block_id, data_id) is unchanged; only the outer block
    // directory is grow-stable (2-level lock-free array). See svs/lib/segmented_vector.h.
    template <typename Array> using blocks_type = lib::SegmentedVector<Array>;

    // Helper static function to compute blocksize value.
    // If blocking parameters have defined blocksize_elements, use it
    // directly. Otherwise, compute blocksize based on blocksize_bytes.
    static lib::PowerOfTwo blocksize(
        const svs::index::vamana::concurrent::SegmentedBlocked<Alloc>& alloc,
        size_t dimensions
    ) {
        return svs::data::compute_blocksize(
            static_cast<const Blocked<Alloc>&>(alloc), dimensions
        );
    }

    template <typename Array> static void reserve(blocks_type<Array>&, size_t) {}

    template <typename Array, typename Dims>
    static void append(blocks_type<Array>& blocks, Dims&& dims, const Alloc& alloc) {
        blocks.push_back(Array(SVS_FWD(dims), alloc));
    }

    // Acquire, paired with the release store in `resize`: `size_` grows concurrently with
    // lock-free readers, so a plain read would be a data race.
    static size_t load_size(const size_t& size_) {
        return std::atomic_ref<size_t>(const_cast<size_t&>(size_))
            .load(std::memory_order_acquire);
    }

    static void store_size(size_t& size_, size_t new_size) {
        // Release: growth is concurrent with lock-free readers calling `size()`, and
        // publishing the new size must not be reordered before the blocks that back it.
        // Shrinking frees memory, so it only ever runs with readers excluded (the
        // owning index holds `compact_mutex_` exclusive); the atomic store is here for
        // consistency, not for safety, which the exclusion provides.
        std::atomic_ref<size_t>(size_).store(new_size, std::memory_order_release);
    }
};

} // namespace detail

} // namespace svs::data

namespace svs::index::vamana::concurrent {

///
/// @brief Grow-stable analogue of ``svs::data::BlockedData``.
///
/// The dataset type the concurrent index is built on. Interchangeable with
/// ``svs::data::BlockedData`` at every call site; the difference is that growing it does
/// not invalidate a concurrent reader.
///
template <typename T, size_t Extent = Dynamic, typename Alloc = lib::Allocator<T>>
using SegmentedBlockedData = svs::data::SimpleData<T, Extent, SegmentedBlocked<Alloc>>;

} // namespace svs::index::vamana::concurrent
