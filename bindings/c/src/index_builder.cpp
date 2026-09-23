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
#include "index_builder.hpp"

#include "algorithm.hpp"
#include "data_builder.hpp"
#include "dispatcher_dynamic_vamana.hpp"
#include "dispatcher_vamana.hpp"
#include "error.hpp"
#include "index.hpp"
#include "index_vamana.hpp"
#include "storage.hpp"
#include "threadpool.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/index/vamana/dynamic_index.h>
#include <svs/index/vamana/index.h>
#include <svs/lib/float16.h>
#include <svs/lib/neighbor.h>
#include <svs/orchestrators/vamana.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>

namespace svs::c_runtime {

std::shared_ptr<Index> IndexBuilder::build(const svs::data::ConstSimpleDataView<float>& data
) {
    if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
        auto vamana_algorithm = static_cast<AlgorithmVamana*>(algorithm.get());

        auto index = std::make_shared<IndexVamana>(
            *this,
            // vamana_algorithm,
            dispatch_vamana_index_build(
                vamana_algorithm->build_parameters(),
                data,
                storage.get(),
                to_distance_type(distance_metric),
                pool_builder.build(),
                allocator_builder
            )
        );

        return index;
    }
    return nullptr;
}

std::shared_ptr<Index> IndexBuilder::load(const std::filesystem::path& directory) {
    if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
        auto vamana_algorithm = static_cast<AlgorithmVamana*>(algorithm.get());

        auto index = std::make_shared<IndexVamana>(
            *this,
            dispatch_vamana_index_load(
                vamana_algorithm->build_parameters(),
                directory,
                storage.get(),
                to_distance_type(distance_metric),
                pool_builder.build(),
                allocator_builder
            )
        );

        return index;
    }
    return nullptr;
}

std::shared_ptr<Index> IndexBuilder::copy(const std::shared_ptr<Index>& src_index) {
    const auto& src_builder = src_index->get_builder();

    if (src_builder.algorithm->type != SVS_ALGORITHM_TYPE_VAMANA) {
        throw not_implemented(
            "Only Vamana algorithm is currently supported for index conversion"
        );
    }

    // Validate that the source and destination builders are compatible
    // by comparing:
    // - distance metric
    // - dimensions
    // - algorithm type
    // - key build parameters: alpha, graph_max_degree
    if (src_builder.distance_metric != distance_metric) {
        throw not_implemented(
            "Distance metric mismatch between source and destination builders"
        );
    }
    if (src_builder.dimension != dimension) {
        throw invalid_operation(
            "Dimensions mismatch between source and destination builders"
        );
    }
    const auto src_algorithm = dynamic_cast<AlgorithmVamana*>(src_builder.algorithm.get());
    if (src_algorithm->type != algorithm->type) {
        throw not_implemented(
            "Algorithm type mismatch between source and destination builders"
        );
    }

    const auto dst_algorithm = dynamic_cast<AlgorithmVamana*>(algorithm.get());
    assert(dst_algorithm && "Destination builder must have a valid Vamana algorithm.");

    const auto& src_build_parameters = src_algorithm->build_parameters();
    const auto& dst_build_parameters = dst_algorithm->build_parameters();

    if (src_build_parameters.alpha != dst_build_parameters.alpha ||
        src_build_parameters.graph_max_degree != dst_build_parameters.graph_max_degree) {
        throw not_implemented(
            "Build parameters mismatch between source and destination builders"
        );
    }

    const auto vamana_index = std::dynamic_pointer_cast<IndexVamana>(src_index);
    assert(vamana_index && "Source index must be a valid Vamana index.");

    auto index = std::make_shared<IndexVamana>(
        *this,
        dispatch_vamana_index_copy(
            dst_build_parameters,
            vamana_index->index,
            src_builder.storage.get(),
            storage.get(),
            to_distance_type(distance_metric),
            pool_builder.build(),
            allocator_builder
        )
    );

    return index;
}

std::shared_ptr<DynamicIndex> IndexBuilder::build_dynamic(
    const svs::data::ConstSimpleDataView<float>& data,
    std::span<const size_t> ids,
    size_t blocksize_bytes
) {
    if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
        auto vamana_algorithm = static_cast<AlgorithmVamana*>(algorithm.get());

        auto index = std::make_shared<DynamicIndexVamana>(
            *this,
            // vamana_algorithm,
            dispatch_dynamic_vamana_index_build(
                vamana_algorithm->build_parameters(),
                data,
                ids,
                storage.get(),
                to_distance_type(distance_metric),
                pool_builder.build(),
                allocator_builder,
                blocksize_bytes
            )
        );

        return index;
    }
    return nullptr;
}

std::shared_ptr<DynamicIndex>
IndexBuilder::load_dynamic(const std::filesystem::path& directory, size_t blocksize_bytes) {
    if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
        auto vamana_algorithm = static_cast<AlgorithmVamana*>(algorithm.get());

        auto index = std::make_shared<DynamicIndexVamana>(
            *this,
            // vamana_algorithm,
            dispatch_dynamic_vamana_index_load(
                vamana_algorithm->build_parameters(),
                directory,
                storage.get(),
                to_distance_type(distance_metric),
                pool_builder.build(),
                allocator_builder,
                blocksize_bytes
            )
        );

        return index;
    }
    return nullptr;
}

svs::index::vamana::MemoryBreakdown
IndexBuilder::estimate_memory_breakdown(size_t num_vectors) const {
    NOT_IMPLEMENTED_IF(
        algorithm->type != SVS_ALGORITHM_TYPE_VAMANA,
        "Memory estimation is currently supported only for Vamana algorithm"
    );
    auto vamana_algorithm = static_cast<AlgorithmVamana*>(algorithm.get());
    return dispatch_vamana_memory_estimate(
        vamana_algorithm->build_parameters(),
        num_vectors,
        dimension,
        storage.get(),
        to_distance_type(distance_metric)
    );
}

svs::index::vamana::MemoryBreakdown IndexBuilder::estimate_memory_breakdown_dynamic(
    size_t num_vectors, size_t blocksize_bytes
) const {
    NOT_IMPLEMENTED_IF(
        algorithm->type != SVS_ALGORITHM_TYPE_VAMANA,
        "Memory estimation is currently supported only for Vamana algorithm"
    );
    auto vamana_algorithm = static_cast<AlgorithmVamana*>(algorithm.get());
    return dispatch_dynamic_vamana_memory_estimate(
        vamana_algorithm->build_parameters(),
        num_vectors,
        dimension,
        storage.get(),
        to_distance_type(distance_metric),
        blocksize_bytes
    );
}

namespace {
template <typename SearchBufferType>
size_t estimate_search_memory_vamana(
    const IndexBuilder& builder,
    size_t num_queries,
    size_t num_neighbors,
    const std::shared_ptr<Algorithm::SearchParams>& search_params,
    const IDFilterInterface* id_filter
) {
    if (search_params && search_params->type != builder.algorithm->type) {
        throw std::invalid_argument("Search parameters type does not match algorithm type");
    }

    auto vamana_algorithm = static_cast<AlgorithmVamana*>(builder.algorithm.get());
    auto vamana_search_params = std::static_pointer_cast<AlgorithmVamana::SearchParams>(
        search_params ? search_params : vamana_algorithm->get_default_search_params()
    );

    auto params = vamana_search_params->get_search_parameters();
    size_t buffer_size =
        std::max(params.buffer_config_.get_total_capacity(), num_neighbors);

    // Extra per-worker heap allocated only by filtered searches (via
    // filtered_topk_search() and its per-worker BatchIterator).
    size_t filter_overhead = 0;
    if (id_filter != nullptr) {
        // filtered_topk_search() sizes its first batch to gather enough raw
        // candidates to leave ~num_neighbors survivors after filtering, then adds
        // the default batch-iterator headroom.
        size_t candidates = num_neighbors;
        const double rate = id_filter->filter_rate();
        if (rate > 0.0) {
            const double needed = static_cast<double>(num_neighbors) / rate;
            // A very low filter rate blows up the candidate buffer, which becomes the
            // Vamana search window and makes search() prohibitively slow. Reject such
            // configurations instead of returning a huge, unrepresentative estimate.
            // The cap is a heuristic: search windows beyond ~1M nodes are impractical.
            constexpr size_t MAX_FILTERED_CANDIDATES = 1'000'000;
            if (needed > static_cast<double>(MAX_FILTERED_CANDIDATES)) {
                throw std::invalid_argument(
                    "Filter rate is too low: the estimated candidate buffer would "
                    "exceed the practical search-window limit and make search "
                    "prohibitively slow"
                );
            }
            candidates = static_cast<size_t>(needed);
        }
        buffer_size =
            svs::ITERATOR_EXTRA_BUFFER_CAPACITY_DEFAULT + std::max(buffer_size, candidates);

        // Per-worker BatchIterator allocations bounded by buffer_size:
        //   results_: a batch of candidate neighbors (id + distance).
        //   yielded_: a node-based unordered_set<uint32_t> (stored key + next
        //             pointer per node, plus one bucket pointer per node).
        const size_t yielded_per_node = sizeof(uint32_t) + 2 * sizeof(void*);
        filter_overhead = buffer_size * (sizeof(svs::Neighbor<size_t>) + yielded_per_node);

        // filtered_topk_search() allocates one per-query count vector which total size
        // is num_queries * sizeof(size_t), but it is negligible for estimation.
    }

    // The 'fixed' query distance functor may add up to ~3 * dimension * sizeof(float)
    // per buffer, but that is negligible for estimation and intentionally ignored.
    const size_t per_buffer_size = SearchBufferType::estimate_memory_footprint(
                                       svs::index::vamana::SearchBufferConfig{buffer_size},
                                       params.search_buffer_visited_set_
                                   ) +
                                   filter_overhead;

    const auto threads_num = builder.pool_builder.get_threads_num();
    assert(threads_num > 0 && "Thread pool must have at least one thread");
    const size_t buffers_num = std::min(threads_num, num_queries);

    return per_buffer_size * buffers_num;
}
} // namespace

size_t IndexBuilder::estimate_search_memory(
    size_t num_queries,
    size_t num_neighbors,
    const std::shared_ptr<Algorithm::SearchParams>& search_params,
    const IDFilterInterface* id_filter
) const {
    NOT_IMPLEMENTED_IF(
        algorithm->type != SVS_ALGORITHM_TYPE_VAMANA,
        "Search memory estimation is currently supported only for Vamana algorithm"
    );
    // Cmp template parameter can be ignored - it is not used in the memory estimation.
    return estimate_search_memory_vamana<svs::index::vamana::SearchBuffer<uint32_t>>(
        *this, num_queries, num_neighbors, search_params, id_filter
    );
}

size_t IndexBuilder::estimate_search_memory_dynamic(
    size_t num_queries,
    size_t num_neighbors,
    const std::shared_ptr<Algorithm::SearchParams>& search_params,
    const IDFilterInterface* id_filter,
    size_t SVS_UNUSED(blocksize_bytes)
) const {
    NOT_IMPLEMENTED_IF(
        algorithm->type != SVS_ALGORITHM_TYPE_VAMANA,
        "Search memory estimation is currently supported only for Vamana algorithm"
    );
    // Cmp template parameter can be ignored - it is not used in the memory estimation.
    return estimate_search_memory_vamana<svs::index::vamana::MutableBuffer<uint32_t>>(
        *this, num_queries, num_neighbors, search_params, id_filter
    );
}

} // namespace svs::c_runtime
