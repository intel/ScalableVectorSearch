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
        : parent_type{other.parameters(), other.get_allocator()} {}
};

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
template <typename T, size_t Extent, typename Alloc>
class SimpleData<T, Extent, svs::index::vamana::concurrent::SegmentedBlocked<Alloc>> {
  public:
    ///// Static Members
    static constexpr bool supports_saving = true;

    // Type Aliases
    using dim_type = std::tuple<size_t, dim_type_t<Extent>>;
    using allocator_type = svs::index::vamana::concurrent::SegmentedBlocked<Alloc>;
    using inner_allocator_type = Alloc;
    using array_type = DenseArray<T, dim_type, inner_allocator_type>;

    /// Return the underlying allocator.
    const allocator_type& get_allocator() const { return allocator_; }

    // value types
    using element_type = T;
    using value_type = std::span<T, Extent>;
    using const_value_type = std::span<const T, Extent>;

    using lib_alloc_data_type = SimpleData<
        T,
        Extent,
        svs::index::vamana::concurrent::SegmentedBlocked<lib::Allocator<T>>>;
    /// Already blocked, so lib_blocked_alloc_data_type is the same as lib_alloc_data_type.
    using lib_blocked_alloc_data_type = SimpleData<
        T,
        Dynamic,
        svs::index::vamana::concurrent::SegmentedBlocked<lib::Allocator<T>>>;

    ///// Constructors
    SimpleData(size_t n_elements, size_t n_dimensions, const allocator_type& alloc)
        : blocksize_{compute_blocksize(alloc, n_dimensions)}
        , blocks_{}
        , dimensions_{n_dimensions}
        , size_{n_elements}
        , allocator_{alloc} {
        size_t elements_per_block = blocksize_.value();
        size_t num_blocks = lib::div_round_up(n_elements, elements_per_block);
        for (size_t i = 0; i < num_blocks; ++i) {
            add_block();
        }
    }

    SimpleData(size_t n_elements, size_t n_dimensions)
        : SimpleData{n_elements, n_dimensions, allocator_type()} {}

    ///
    /// Convert a linear index into an inner-outer index to access the blocked dataset.
    /// Returns a pair `p` where:
    /// - `p.first` is the block index.
    /// - `p.second` is the index within the block.
    ///
    std::pair<size_t, size_t> resolve(size_t i) const {
        return std::pair<size_t, size_t>{i / blocksize_, i % blocksize_};
    }

    ///
    /// Return the blocksize with reference to the stored data vectors.
    ///
    lib::PowerOfTwo blocksize() const { return blocksize_; }

    ///
    /// Return the blocksize with respect to bytes.
    ///
    lib::PowerOfTwo blocksize_bytes() const {
        return allocator_.parameters().blocksize_bytes;
    }

    ///
    /// Return the number of blocks in the dataset.
    ///
    size_t num_blocks() const { return blocks_.size(); }

    ///
    /// Return the maximum number of data vectors that can be stored before a new block is
    /// required.
    ///
    size_t capacity() const { return num_blocks() * blocksize(); }

    ///
    /// Return an iterator over each index in the dataset.
    ///
    threads::UnitRange<size_t> eachindex() const {
        return threads::UnitRange<size_t>{0, size()};
    }

    ///
    /// Add a new data block to the end of the current collection of blocks.
    ///
    void add_block() {
        blocks_.push_back(array_type(
            make_dims(blocksize().value(), lib::forward_extent<Extent>(dimensions())),
            allocator_.get_allocator()
        ));
    }

    ///
    /// Remove a data block from the end of the block list.
    ///
    void drop_block() {
        if (!blocks_.empty()) {
            blocks_.pop_back();
        }
    }

    ///
    /// Resizing
    ///
    void resize(size_t new_size) {
        if (new_size > size()) {
            // Add blocks until there is sufficient capacity.
            while (new_size > capacity()) {
                add_block();
            }
            // Release: growth is concurrent with lock-free readers calling `size()`, and
            // publishing the new size must not be reordered before the blocks that back it.
            std::atomic_ref<size_t>(size_).store(new_size, std::memory_order_release);
        } else if (new_size < size()) {
            // Reset size then drop blocks until the new size is within the last block.
            // Shrinking frees memory, so it only ever runs with readers excluded (the
            // owning index holds `compact_mutex_` exclusive); the atomic store is here for
            // consistency, not for safety, which the exclusion provides.
            std::atomic_ref<size_t>(size_).store(new_size, std::memory_order_release);
            while (capacity() - blocksize().value() > new_size) {
                drop_block();
            }
        }
    }

    void shrink_to_fit() {
        // We already shrink when down-sizing, so ``shink_to_fit`` becomes a no-op.
    }

    /////
    ///// Dataset API
    /////

    // Acquire, paired with the release store in `resize`: `size_` grows concurrently with
    // lock-free readers, so a plain read would be a data race.
    size_t size() const {
        return std::atomic_ref<size_t>(const_cast<size_t&>(size_))
            .load(std::memory_order_acquire);
    }
    constexpr size_t dimensions() const {
        if constexpr (Extent != Dynamic) {
            return Extent;
        } else {
            return dimensions_;
        }
    }

    size_t element_size() const { return sizeof(element_type) * dimensions(); }

    const_value_type get_datum(size_t i) const {
        auto [block_id, data_id] = resolve(i);
        return getindex(blocks_, block_id).slice(data_id);
    }

    value_type get_datum(size_t i) {
        auto [block_id, data_id] = resolve(i);
        return getindex(blocks_, block_id).slice(data_id);
    }

    void prefetch(size_t i) const { lib::prefetch(get_datum(i)); }

    template <typename U, size_t OtherExtent>
    void set_datum(size_t i, std::span<U, OtherExtent> datum) {
        if constexpr (checkbounds_v) {
            if (datum.size() != dimensions()) {
                throw ANNEXCEPTION(
                    "Datum with dimensions {} is not equal to internal dimensions {}!",
                    datum.size(),
                    dimensions_
                );
            }
        }

        if constexpr (std::is_same_v<T, std::remove_const_t<U>>) {
            std::copy(datum.begin(), datum.end(), get_datum(i).begin());
        } else {
            std::transform(
                datum.begin(),
                datum.end(),
                get_datum(i).begin(),
                [](const U& u) { return lib::relaxed_narrow<T>(u); }
            );
        }
    }

    template <typename U, typename A> void set_datum(size_t i, const std::vector<U, A>& v) {
        set_datum(i, lib::as_span(v));
    }

    ///
    /// Construct an identical copy of the dataset.
    /// Not implemented as a copy constructor to avoid unintentional copies.
    ///
    SimpleData copy() const {
        SimpleData other{size(), dimensions(), allocator_};
        for (const auto& i : eachindex()) {
            other.set_datum(i, get_datum(i));
        }
        return other;
    }

    ///// Compaction
    template <std::integral I, threads::ThreadPool Pool>
    void
    compact(std::span<const I> new_to_old, Pool& threadpool, size_t batchsize = 1'000'000) {
        // Allocate scratch space.
        batchsize = std::min(batchsize, size());
        auto buffer = data::SimpleData<T, Extent>(batchsize, dimensions());
        compact_data(*this, buffer, new_to_old, threadpool);
    }

    template <std::integral I>
    void compact(std::span<const I> new_to_old, size_t batchsize = 1'000'000) {
        auto pool = threads::SequentialThreadPool();
        compact(new_to_old, pool, batchsize);
    }

    ///// Saving
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return GenericSerializer::save(*this, ctx);
    }

    void save(std::ostream& os) const { return GenericSerializer::save(*this, os); }

    lib::SaveTable metadata() const { return GenericSerializer::metadata(*this); }

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return GenericSerializer::check_compatibility(schema, version);
    }

    static SimpleData
    load(const lib::LoadTable& table, const allocator_type& allocator = {}) {
        return GenericSerializer::load<T>(
            table, lib::Lazy([&allocator](size_t n_elements, size_t n_dimensions) {
                return SimpleData(n_elements, n_dimensions, allocator);
            })
        );
    }

    static SimpleData load(
        const lib::ContextFreeLoadTable& table,
        std::istream& is,
        const allocator_type& allocator = {}
    ) {
        return GenericSerializer::load<T>(
            table, is, lib::Lazy([&allocator](size_t n_elements, size_t n_dimensions) {
                return SimpleData(n_elements, n_dimensions, allocator);
            })
        );
    }

    static SimpleData
    load(const std::filesystem::path& path, const allocator_type& allocator = {}) {
        if (detail::is_likely_reload(path)) {
            return lib::load_from_disk<SimpleData>(path, allocator);
        }
        // Try loading directly.
        return io::auto_load<T>(
            path, lib::Lazy([&allocator](size_t n_elements, size_t n_dimensions) {
                return SimpleData(n_elements, n_dimensions, allocator);
            })
        );
    }

  private:
    // Helper static function to compute blocksize value.
    // If blocking parameters have defined blocksize_elements, use it
    // directly. Otherwise, compute blocksize based on blocksize_bytes.
    static lib::PowerOfTwo compute_blocksize(const allocator_type& alloc, size_t dim) {
        if (alloc.parameters().blocksize_elements.has_value()) {
            return alloc.parameters().blocksize_elements.value();
        } else {
            return lib::prevpow2(
                alloc.parameters().blocksize_bytes.value() / (sizeof(T) * dim)
            );
        }
    }

  private:
    // The blocksize in terms of number of vectors.
    lib::PowerOfTwo blocksize_;
    // Grow-stable directory of fixed-size blocks: appending a block never relocates the
    // existing block wrappers (or the heap buffers they point to), so a concurrent
    // lock-free reader subscripting blocks_[block_id] is safe against a writer growing the
    // dataset. Element addressing (block_id, data_id) is unchanged; only the outer block
    // directory is grow-stable (2-level lock-free array). See svs/lib/segmented_vector.h.
    lib::SegmentedVector<array_type> blocks_;
    size_t dimensions_;
    size_t size_;
    allocator_type allocator_;
};

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
