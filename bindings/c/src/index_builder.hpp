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

#include "svs/c/svs_c.h"

#include "algorithm.hpp"
#include "data_builder.hpp"
#include "dispatcher_dynamic_vamana.hpp"
#include "dispatcher_vamana.hpp"
#include "index.hpp"
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

#include <filesystem>
#include <memory>

namespace svs::c_runtime {

struct IndexBuilder {
    svs_distance_metric_t distance_metric;
    size_t dimension;
    std::shared_ptr<Algorithm> algorithm;
    std::shared_ptr<Storage> storage;
    ThreadPoolBuilder pool_builder;
    AllocatorHandle<std::byte> allocator_handle;

    IndexBuilder(
        svs_distance_metric_t distance_metric,
        size_t dimension,
        std::shared_ptr<Algorithm> algorithm
    )
        : distance_metric(distance_metric)
        , dimension(dimension)
        , algorithm(std::move(algorithm))
        , storage(std::make_shared<StorageSimple>(SVS_DATA_TYPE_FLOAT32))
        , pool_builder{}
        , allocator_handle{make_allocator_handle(svs::lib::Allocator<std::byte>{})} {}

    ~IndexBuilder() {}

    void set_storage(std::shared_ptr<Storage> storage) {
        this->storage = std::move(storage);
    }

    void set_threadpool_builder(ThreadPoolBuilder threadpool_builder) {
        std::swap(this->pool_builder, threadpool_builder);
    }

    void set_allocator_handle(AllocatorHandle<std::byte> allocator_handle) {
        this->allocator_handle = std::move(allocator_handle);
    }

    std::shared_ptr<Index> build(const svs::data::ConstSimpleDataView<float>& data) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index = std::make_shared<IndexVamana>(
                vamana_algorithm,
                dispatch_vamana_index_build(
                    vamana_algorithm->build_parameters(),
                    data,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build(),
                    allocator_handle
                ),
                pool_builder
            );

            return index;
        }
        return nullptr;
    }

    std::shared_ptr<Index> load(const std::filesystem::path& directory) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index = std::make_shared<IndexVamana>(
                vamana_algorithm,
                dispatch_vamana_index_load(
                    vamana_algorithm->build_parameters(),
                    directory,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build(),
                    allocator_handle
                ),
                pool_builder
            );

            return index;
        }
        return nullptr;
    }

    std::shared_ptr<DynamicIndex> build_dynamic(
        const svs::data::ConstSimpleDataView<float>& data,
        std::span<const size_t> ids,
        size_t blocksize_bytes
    ) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index = std::make_shared<DynamicIndexVamana>(
                vamana_algorithm,
                dispatch_dynamic_vamana_index_build(
                    vamana_algorithm->build_parameters(),
                    data,
                    ids,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build(),
                    allocator_handle,
                    blocksize_bytes
                ),
                pool_builder
            );

            return index;
        }
        return nullptr;
    }

    std::shared_ptr<DynamicIndex>
    load_dynamic(const std::filesystem::path& directory, size_t blocksize_bytes) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index = std::make_shared<DynamicIndexVamana>(
                vamana_algorithm,
                dispatch_dynamic_vamana_index_load(
                    vamana_algorithm->build_parameters(),
                    directory,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build(),
                    allocator_handle,
                    blocksize_bytes
                ),
                pool_builder
            );

            return index;
        }
        return nullptr;
    }

    // Estimate the memory a built static Vamana index would consume
    // for `num_vectors` vectors. Mirrors the accounting done by
    // svs::index::vamana::VamanaIndex::get_memory_breakdown().
    svs::index::vamana::MemoryBreakdown estimate_memory_breakdown(size_t num_vectors
    ) const {
        NOT_IMPLEMENTED_IF(
            algorithm->type != SVS_ALGORITHM_TYPE_VAMANA,
            "Memory estimation is currently supported only for Vamana algorithm"
        );
        auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);
        return dispatch_vamana_memory_estimate(
            vamana_algorithm->build_parameters(),
            num_vectors,
            dimension,
            storage.get(),
            to_distance_type(distance_metric)
        );
    }

    // Estimate the memory a built dynamic Vamana index would consume
    // for `num_vectors` vectors. Mirrors the accounting done by
    // svs::index::vamana::MutableVamanaIndex::get_memory_breakdown().
    svs::index::vamana::MemoryBreakdown
    estimate_memory_breakdown_dynamic(size_t num_vectors, size_t blocksize_bytes) const {
        NOT_IMPLEMENTED_IF(
            algorithm->type != SVS_ALGORITHM_TYPE_VAMANA,
            "Memory estimation is currently supported only for Vamana algorithm"
        );
        auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);
        return dispatch_dynamic_vamana_memory_estimate(
            vamana_algorithm->build_parameters(),
            num_vectors,
            dimension,
            storage.get(),
            to_distance_type(distance_metric),
            blocksize_bytes
        );
    }

    template <typename SearchBufferType>
    size_t estimate_search_memory_vamana(
        size_t num_queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter
    ) const {
        if (search_params && search_params->type != algorithm->type) {
            throw std::invalid_argument(
                "Search parameters type does not match algorithm type"
            );
        }

        auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);
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
            buffer_size = svs::ITERATOR_EXTRA_BUFFER_CAPACITY_DEFAULT +
                          std::max(buffer_size, candidates);

            // Per-worker BatchIterator allocations bounded by buffer_size:
            //   results_: a batch of candidate neighbors (id + distance).
            //   yielded_: a node-based unordered_set<uint32_t> (stored key + next
            //             pointer per node, plus one bucket pointer per node).
            const size_t yielded_per_node = sizeof(uint32_t) + 2 * sizeof(void*);
            filter_overhead =
                buffer_size * (sizeof(svs::Neighbor<size_t>) + yielded_per_node);

            // filtered_topk_search() allocates one per-query count vector which total size
            // is num_queries * sizeof(size_t), but it is negligible for estimation.
        }

        // The 'fixed' query distance functor may add up to ~3 * dimension * sizeof(float)
        // per buffer, but that is negligible for estimation and intentionally ignored.
        const size_t per_buffer_size =
            SearchBufferType::estimate_memory_footprint(
                svs::index::vamana::SearchBufferConfig{buffer_size},
                params.search_buffer_visited_set_
            ) +
            filter_overhead;

        const auto threads_num = pool_builder.get_threads_num();
        assert(threads_num > 0 && "Thread pool must have at least one thread");
        const size_t buffers_num = std::min(threads_num, num_queries);

        return per_buffer_size * buffers_num;
    }

    size_t estimate_search_memory(
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
            num_queries, num_neighbors, search_params, id_filter
        );
    }

    size_t estimate_search_memory_dynamic(
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
            num_queries, num_neighbors, search_params, id_filter
        );
    }
};
} // namespace svs::c_runtime
