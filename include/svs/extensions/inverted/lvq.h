/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

// Extensions being defined.
#include "svs/index/inverted/extensions.h"

#include "svs/extensions/vamana/lvq.h"
#include "svs/quantization/lvq/lvq.h"

namespace svs::quantization::lvq {

// Prepare a group of dataset elements for searching across the primary index.
// In practice, this involves decompressing the data to be searched.
//
// TODO: Provide a threadpool to enable multi-threaded decompression.
template <IsLVQDataset Original, std::integral I>
data::SimpleData<float> svs_invoke(
    svs::tag_t<index::inverted::extensions::prepare_index_search>,
    const Original& original,
    std::span<const I> indices
) {
    auto dst = data::SimpleData<float>(indices.size(), original.dimensions());
    auto decompressor = original.decompressor();
    for (size_t i = 0, imax = indices.size(); i < imax; ++i) {
        dst.set_datum(i, decompressor(original.get_datum(indices[i])));
    }
    return dst;
}

// Convert a distance functor to one that can be used for general distance computations
// across elements within `data`.
template <IsLVQDataset Data, typename Distance>
auto svs_invoke(
    svs::tag_t<index::inverted::extensions::clustering_distance>,
    const Data& data,
    const Distance& distance
) {
    return adapt_for_self(data, distance);
}

// A general routine for creating a one-level ScaledBiasedDataset.
template <IsLVQDataset Data, typename Alloc>
auto create_onelevel_from(
    const Data& original, size_t new_size, size_t new_alignment, const Alloc& allocator
) {
    constexpr size_t Bits = Data::primary_bits;
    constexpr size_t Extent = Data::extent;
    using Strategy = typename Data::strategy;

    auto& primary = original.get_primary_dataset();

    return ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>(
        new_size, primary.static_dims(), new_alignment, allocator
    );
}

template <IsLVQDataset Data, typename Alloc>
auto create_lvq_from(const Data& original, size_t new_size, const Alloc& allocator) {
    // At this level, we need to return a full LVQDataset.
    constexpr size_t Bits = Data::primary_bits;
    constexpr size_t Extent = Data::extent;
    using Strategy = typename Data::strategy;
    auto& primary = original.get_primary_dataset();

    // Allocate a new primary dataset of the requested size.
    auto new_primary = ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>(
        new_size, primary.static_dims(), primary.get_alignment(), allocator
    );
    return LVQDataset<Bits, 0, Extent, Strategy, Alloc>(
        std::move(new_primary), *original.view_centroids()
    );
}

// Create a full LVQ dataset.
template <IsLVQDataset Data, typename Alloc>
auto svs_invoke(
    svs::tag_t<index::inverted::extensions::create_auxiliary_dataset>,
    const Data& original,
    size_t new_size,
    const Alloc& allocator
) {
    return create_lvq_from(original, new_size, allocator);
}

template <IsLVQDataset Data, typename Alloc>
auto svs_invoke(
    svs::tag_t<index::inverted::extensions::create_first_level_dataset>,
    const Data& original,
    size_t new_size,
    const Alloc& allocator
) {
    return create_lvq_from(original, new_size, allocator);
}

template <IsLVQDataset Data, typename Alloc>
auto svs_invoke(
    svs::tag_t<index::inverted::extensions::create_sparse_cluster>,
    const Data& original,
    size_t new_size,
    const Alloc& allocator
) {
    return create_onelevel_from(
        original, new_size, original.get_primary_dataset().get_alignment(), allocator
    );
}

template <IsLVQDataset Data, typename Alloc>
auto svs_invoke(
    svs::tag_t<index::inverted::extensions::create_dense_cluster>,
    const Data& original,
    size_t new_size,
    const Alloc& SVS_UNUSED(allocator)
) {
    return create_onelevel_from(
        original,
        new_size,
        original.get_primary_dataset().get_alignment(),
        std::allocator<std::byte>()
    );
}

} // namespace svs::quantization::lvq
