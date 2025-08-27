/*
 * Copyright 2025 Intel Corporation
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

#include "svs/core/distance.h"
#include "svs/core/query_result.h"
#include "svs/index/ivf/common.h"
#include "svs/lib/invoke.h"

namespace svs::index::ivf::extensions {

struct IVFAccessor {
    template <typename Data>
    svs::svs_invoke_result_t<IVFAccessor, Data> operator()(const Data& data) const {
        return svs::svs_invoke(*this, data);
    }
};

inline constexpr IVFAccessor accessor{};

template <typename Data>
data::GetDatumAccessor svs_invoke(svs::tag_t<accessor>, const Data& SVS_UNUSED(data)) {
    return data::GetDatumAccessor{};
}

struct IVFPerThreadBatchSearchSetupType {
    template <typename Data, typename Distance>
    using result_t = svs::
        svs_invoke_result_t<IVFPerThreadBatchSearchSetupType, const Data&, const Distance&>;

    template <typename Data, typename Distance>
    result_t<Data, Distance>
    operator()(const Data& dataset, const Distance& distance) const {
        return svs::svs_invoke(*this, dataset, distance);
    }
};

inline constexpr IVFPerThreadBatchSearchSetupType per_thread_batch_search_setup{};

// Default Implementation.
template <typename Data, typename Distance>
Distance svs_invoke(
    svs::tag_t<per_thread_batch_search_setup>,
    const Data& SVS_UNUSED(dataset),
    const Distance& distance
) {
    return threads::shallow_copy(distance);
}

///
/// @brief Customization point for working with a batch of threads.
///
struct IVFPerThreadBatchSearchType {
    template <
        data::ImmutableMemoryDataset Data,
        typename Cluster,
        typename BufferCentroids,
        typename BufferLeaves,
        typename Scratch,
        data::ImmutableMemoryDataset Queries,
        std::integral I,
        typename SearchCentroids,
        typename SearchLeaves>
    SVS_FORCE_INLINE void operator()(
        const Data& data,
        const Cluster& cluster,
        BufferCentroids& buffer_centroids,
        BufferLeaves& buffer_leaves,
        Scratch& scratch,
        const Queries& queries,
        QueryResultView<I>& result,
        threads::UnitRange<size_t> thread_indices,
        size_t tid,
        const SearchCentroids& search_centroids,
        const SearchLeaves& search_leaves
    ) const {
        svs::svs_invoke(
            *this,
            data,
            cluster,
            buffer_centroids,
            buffer_leaves,
            scratch,
            queries,
            result,
            thread_indices,
            tid,
            search_centroids,
            search_leaves
        );
    }
};

/// @brief Customization point object for batch search.
inline constexpr IVFPerThreadBatchSearchType per_thread_batch_search{};

// Default Implementation
template <
    typename Data,
    typename Cluster,
    typename BufferCentroids,
    typename BufferLeaves,
    typename Distance,
    typename Queries,
    std::integral I,
    typename SearchCentroids,
    typename SearchLeaves>
void svs_invoke(
    svs::tag_t<per_thread_batch_search>,
    const Data& SVS_UNUSED(dataset),
    const Cluster& cluster,
    BufferCentroids& buffer_centroids,
    BufferLeaves& buffer_leaves,
    Distance& distance,
    const Queries& queries,
    QueryResultView<I>& result,
    threads::UnitRange<size_t> thread_indices,
    const size_t tid,
    const SearchCentroids& search_centroids,
    const SearchLeaves& search_leaves
) {
    size_t n_inner_threads = buffer_leaves.size();
    size_t buffer_leaves_size = buffer_leaves[0].capacity();
    size_t num_neighbors = result.n_neighbors();

    // Fallback implementation
    for (auto i : thread_indices) {
        const auto& query = queries.get_datum(i);
        search_centroids(query, buffer_centroids, i);
        search_leaves(query, distance, buffer_centroids, buffer_leaves, tid);

        // Accumulate results from intra-query threads
        for (size_t j = 1; j < n_inner_threads; ++j) {
            for (size_t k = 0; k < buffer_leaves_size; ++k) {
                buffer_leaves[0].insert(buffer_leaves[j][k]);
            }
        }

        for (size_t j = 0; j < buffer_leaves_size; ++j) {
            auto& neighbor = buffer_leaves[0][j];
            auto cluster_id = neighbor.id();
            auto local_id = neighbor.get_local_id();
            auto global_id = cluster.get_global_id(cluster_id, local_id);
            neighbor.set_id(global_id);
        }

        // Store results
        for (size_t j = 0; j < num_neighbors; ++j) {
            result.set(buffer_leaves[0][j], i, j);
        }
    }
}

struct CreateDenseCluster {
    using This = CreateDenseCluster;

    template <typename Data, typename Alloc>
    using return_type = svs::svs_invoke_result_t<This, const Data&, size_t, const Alloc&>;

    template <typename Data, typename Alloc>
    return_type<Data, Alloc>
    operator()(const Data& original, size_t new_size, const Alloc& allocator) const {
        return svs::svs_invoke(*this, original, new_size, allocator);
    }
};

inline constexpr CreateDenseCluster create_dense_cluster{};

template <typename T, size_t Extent, typename Alloc, typename NewAlloc>
svs::data::SimpleData<T, Extent> svs_invoke(
    svs::tag_t<create_dense_cluster>,
    const svs::data::SimpleData<T, Extent, Alloc>& original,
    size_t new_size,
    const NewAlloc& SVS_UNUSED(allocator)
) {
    return svs::data::SimpleData<T, Extent>(new_size, original.dimensions());
}

struct SetDenseCluster {
    template <typename Src, typename Dst, typename Idx>
    void operator()(
        const Src& src, Dst& dst, const std::vector<Idx>& src_ids, std::vector<Idx>& dst_ids
    ) const {
        return svs::svs_invoke(*this, src, dst, src_ids, dst_ids);
    }
};

inline constexpr SetDenseCluster set_dense_cluster{};

template <typename Src, typename Dst, typename Idx>
void svs_invoke(
    svs::tag_t<set_dense_cluster>,
    const Src& src,
    Dst& dst,
    const std::vector<Idx>& src_ids,
    std::vector<Idx>& dst_ids
) {
    size_t i = 0;
    for (auto id : src_ids) {
        dst.set_datum(i, src.get_datum(id));
        dst_ids[i] = id;
        ++i;
    }
}

} // namespace svs::index::ivf::extensions
