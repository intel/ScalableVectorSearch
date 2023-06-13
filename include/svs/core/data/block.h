/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

#include "svs/concepts/data.h"
#include "svs/core/allocator.h"
#include "svs/core/compact.h"
#include "svs/core/data/simple.h"
#include "svs/lib/array.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/misc.h"
#include "svs/lib/prefetch.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"
#include "svs/third-party/toml.h"

#include <cmath>
#include <span>
#include <vector>

namespace svs {
namespace data {

template <typename T, size_t Extent = Dynamic> class BlockedData {
  public:
    ///// Static Members

    ///
    /// Default block size in bytes.
    ///
    static constexpr size_t default_blocksize_bytes = size_t{1} << size_t{30};
    static constexpr bool supports_saving = true;

    // Type Aliases
    using dim_type = std::tuple<size_t, dim_type_t<Extent>>;
    using storage_type = MMapPtr<T>;
    using allocator_type = HugepageAllocator;
    // using storage_type = lib::DefaultStorage<T>;
    // using allocator_type = lib::DefaultAllocator;
    using array_type = DenseArray<T, dim_type, storage_type>;

    // value types
    using element_type = T;
    using value_type = std::span<T, Extent>;
    using const_value_type = std::span<const T, Extent>;

    // Mode API
    template <AccessMode = DefaultAccess> using mode_const_value_type = const_value_type;
    template <AccessMode = DefaultAccess> using mode_value_type = value_type;

    ///// Constructors
    BlockedData(
        size_t n_elements,
        size_t n_dimensions,
        size_t blocksize_bytes = default_blocksize_bytes
    )
        : blocksize_{lib::prevpow2(blocksize_bytes / (sizeof(T) * n_dimensions))}
        , blocks_{}
        , dimensions_{n_dimensions}
        , size_{n_elements}
        , allocator_{}
        , blocksize_bytes_{lib::prevpow2(blocksize_bytes)} {
        // Begin allocating blocks.
        size_t elements_per_block = blocksize_.value();
        size_t num_blocks = lib::div_round_up(n_elements, elements_per_block);
        blocks_.reserve(num_blocks);
        for (size_t i = 0; i < num_blocks; ++i) {
            add_block();
        }
    }

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
    lib::PowerOfTwo blocksize_bytes() const { return blocksize_bytes_; }

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
        blocks_.emplace_back(
            allocator_, blocksize().value(), meta::forward_extent<Extent>(dimensions())
        );
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
            size_ = new_size;
        } else if (new_size < size()) {
            // Reset size then drop blocks until the new size is within the last block.
            size_ = new_size;
            while (capacity() - blocksize().value() > new_size) {
                drop_block();
            }
        }
    }

    /////
    ///// Dataset API
    /////

    size_t size() const { return size_; }
    constexpr size_t dimensions() const {
        if constexpr (Extent != Dynamic) {
            return Extent;
        } else {
            return dimensions_;
        }
    }

    template <AccessMode Mode = DefaultAccess>
    const_value_type get_datum(size_t i, Mode SVS_UNUSED(mode) = {}) const {
        auto [block_id, data_id] = resolve(i);
        return getindex(blocks_, block_id).slice(data_id);
    }

    template <AccessMode Mode = DefaultAccess>
    value_type get_datum(size_t i, Mode SVS_UNUSED(mode) = {}) {
        auto [block_id, data_id] = resolve(i);
        return getindex(blocks_, block_id).slice(data_id);
    }

    template <AccessMode Mode = DefaultAccess>
    void prefetch(size_t i, Mode SVS_UNUSED(mode) = {}) const {
        lib::prefetch(get_datum(i));
    }

    template <typename U, size_t OtherExtent, AccessMode Mode = DefaultAccess>
    void set_datum(size_t i, std::span<U, OtherExtent> datum, Mode SVS_UNUSED(mode) = {}) {
        if constexpr (checkbounds_v) {
            if (datum.size() != dimensions()) {
                throw ANNEXCEPTION(
                    "Datum with dimensions ",
                    datum.size(),
                    " is not equal to internal dimensions ",
                    dimensions_,
                    '!'
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

    template <typename U, typename Alloc, AccessMode Mode = DefaultAccess>
    void set_datum(size_t i, const std::vector<U, Alloc>& v, Mode SVS_UNUSED(mode) = {}) {
        set_datum(i, lib::as_span(v));
    }

    ///
    /// Construct an identical copy of the dataset.
    /// Not implemented as a copy constructor to avoid unintentional copies.
    ///
    BlockedData copy() const {
        BlockedData other{size(), dimensions(), blocksize_bytes().value()};
        for (const auto& i : eachindex()) {
            other.set_datum(i, get_datum(i));
        }
        return other;
    }

    // Distance Adaptors
    template <typename Distance> static Distance adapt_distance(const Distance& distance) {
        return threads::shallow_copy(distance);
    }

    template <typename Distance> static Distance self_distance(const Distance& distance) {
        return threads::shallow_copy(distance);
    }

    ///// Compaction
    template <typename I, typename Alloc, threads::ThreadPool Pool>
    void compact(
        const std::vector<I, Alloc>& new_to_old,
        Pool& threadpool,
        size_t batchsize = 1'000'000
    ) {
        // Alllocate scratch space.
        auto buffer = data::SimpleData<T, Extent>(batchsize, dimensions());
        compact_data(*this, buffer, new_to_old, threadpool);
    }

    template <typename I, typename Alloc>
    void compact(const std::vector<I, Alloc>& new_to_old, size_t batchsize = 1'000'000) {
        auto pool = threads::SequentialThreadPool();
        compact(new_to_old, pool, batchsize);
    }

    ///// Saving
    lib::SaveType save(const lib::SaveContext& ctx) const {
        return GenericSaver(*this).save(ctx);
    }

  private:
    // The block size for elements in the dataset.
    lib::PowerOfTwo blocksize_;
    std::vector<array_type> blocks_;
    size_t dimensions_;
    size_t size_;
    allocator_type allocator_;

    // The block size in bytes.
    lib::PowerOfTwo blocksize_bytes_;
};

class BlockedBuilder {
  public:
    BlockedBuilder() = default;

    template <typename T, size_t Extent = Dynamic>
    using return_type = data::BlockedData<T, Extent>;

    // Allocate a blocked dataset.
    template <typename T, size_t Extent = Dynamic>
    return_type<T, Extent> build(size_t size, size_t dimensions) const {
        return data::BlockedData<T, Extent>(size, dimensions);
    }

    // TODO: Save blocking parameters to the toml file to allow them to be
    // reloaded.
    void load_hook(const toml::table&) const {}
};

} // namespace data
} // namespace svs
