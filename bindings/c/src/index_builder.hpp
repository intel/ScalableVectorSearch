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

#include "svs/c_api/svs_c.h"

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

    IndexBuilder(
        svs_distance_metric_t distance_metric,
        size_t dimension,
        std::shared_ptr<Algorithm> algorithm
    )
        : distance_metric(distance_metric)
        , dimension(dimension)
        , algorithm(std::move(algorithm))
        , storage(std::make_shared<StorageSimple>(SVS_DATA_TYPE_FLOAT32))
        , pool_builder{} {}

    ~IndexBuilder() {}

    void set_storage(std::shared_ptr<Storage> storage) {
        this->storage = std::move(storage);
    }

    void set_threadpool_builder(ThreadPoolBuilder threadpool_builder) {
        std::swap(this->pool_builder, threadpool_builder);
    }

    std::shared_ptr<Index> build(const svs::data::ConstSimpleDataView<float>& data) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index = std::make_shared<IndexVamana>(
                dispatch_vamana_index_build(
                    vamana_algorithm->build_parameters(),
                    data,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build()
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
                dispatch_vamana_index_load(
                    vamana_algorithm->build_parameters(),
                    directory,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build()
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
                dispatch_dynamic_vamana_index_build(
                    vamana_algorithm->build_parameters(),
                    data,
                    ids,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build(),
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
                dispatch_dynamic_vamana_index_load(
                    vamana_algorithm->build_parameters(),
                    directory,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build(),
                    blocksize_bytes
                ),
                pool_builder
            );

            return index;
        }
        return nullptr;
    }

    // Estimate the memory a built static Vamana + Simple-storage index would consume
    // for `num_vectors` vectors. Mirrors the accounting done by
    // svs::index::vamana::MutableVamanaIndex::get_memory_breakdown().
    svs::index::vamana::MemoryBreakdown estimate_memory(size_t num_vectors) const {
        NOT_IMPLEMENTED_IF(
            algorithm->type != SVS_ALGORITHM_TYPE_VAMANA,
            "Memory estimation is currently supported only for Vamana algorithm"
        );
        NOT_IMPLEMENTED_IF(
            storage->kind != SVS_STORAGE_KIND_SIMPLE,
            "Memory estimation is currently supported only for Simple storage"
        );
        auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);
        svs::index::vamana::MemoryBreakdown breakdown{};

        // Graph: SimpleData<uint32_t> with num_vectors rows and (max_degree + 1) cols;
        // the +1 slot stores the per-node neighbor count.
        using index_type = uint32_t;
        const size_t max_degree = vamana_algorithm->build_parameters().graph_max_degree;
        using graph_builder_type = svs::SimpleDataBuilder<index_type>;
        breakdown.graph_bytes =
            graph_builder_type{}.estimate_size(num_vectors, (max_degree + 1));

        // Data: SimpleData<T> with num_vectors rows and `dimension` cols.
        breakdown.data_bytes = estimate_data_size(storage.get(), num_vectors, dimension);
        // Metadata: single entry point held as Idx.
        breakdown.metadata_bytes = sizeof(index_type);
        return breakdown;
    }

    // Estimate the memory a built dynamic Vamana + Simple-storage index would consume
    // for `num_vectors` vectors. Mirrors the accounting done by
    // svs::index::vamana::MutableVamanaIndex::get_memory_breakdown().
    svs::index::vamana::MemoryBreakdown
    estimate_memory_dynamic(size_t num_vectors, size_t blocksize_bytes) const {
        NOT_IMPLEMENTED_IF(
            algorithm->type != SVS_ALGORITHM_TYPE_VAMANA,
            "Memory estimation is currently supported only for Vamana algorithm"
        );
        NOT_IMPLEMENTED_IF(
            storage->kind != SVS_STORAGE_KIND_SIMPLE,
            "Memory estimation is currently supported only for Simple storage"
        );
        auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);
        svs::index::vamana::MemoryBreakdown breakdown{};
        // Graph: SimpleBlockedData<uint32_t> with num_vectors rows and (max_degree + 1)
        // cols; the +1 slot stores the per-node neighbor count.
        using index_type = uint32_t;
        const size_t max_degree = vamana_algorithm->build_parameters().graph_max_degree;

        using allocator_type = svs::data::Blocked<svs::lib::Allocator<index_type>>;
        using graph_builder_type = svs::SimpleDataBuilder<index_type, allocator_type>;

        svs::data::BlockingParameters blocking_params{};
        if (blocksize_bytes != 0) {
            blocking_params.blocksize_bytes = svs::lib::prevpow2(blocksize_bytes);
        }
        auto allocator = allocator_type{blocking_params};

        breakdown.graph_bytes =
            graph_builder_type{}.estimate_size(num_vectors, (max_degree + 1), allocator);

        // Data: SimpleData<T> with num_vectors rows and `dimension` cols.
        breakdown.data_bytes = estimate_data_size_blocked(
            storage.get(), num_vectors, dimension, blocksize_bytes
        );

        // Metadata: single entry point held as Idx, plus the SlotMetadata vector, plus the
        // IDTranslator maps.
        size_t metadata_bytes =
            sizeof(index_type) + sizeof(svs::index::vamana::SlotMetadata) * num_vectors;
        // The IDTranslator holds two tsl::robin_map instances (external->internal and
        // internal->external), neither of which exposes its allocated byte count. We
        // approximate the storage as the id pair held in each of the two directions. This
        // ignores the maps' load-factor slack and control bytes, so it is an estimate of
        // the hash-map overhead that is accurate to within a few percent.
        metadata_bytes += 2 * num_vectors *
                          (sizeof(IDTranslator::external_id_type) +
                           sizeof(IDTranslator::internal_id_type));
        breakdown.metadata_bytes = metadata_bytes;
        return breakdown;
    }
};
} // namespace svs::c_runtime
