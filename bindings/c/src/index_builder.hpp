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
#include "allocator.hpp"
#include "data_builder.hpp"
#include "storage.hpp"
#include "threadpool.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/data/simple.h>
#include <svs/index/vamana/index.h>

#include <filesystem>
#include <memory>
#include <span>
#include <utility>

namespace svs::c_runtime {

struct Index;
struct DynamicIndex;

struct IndexBuilder {
    svs_distance_metric_t distance_metric;
    size_t dimension;
    std::unique_ptr<Algorithm> algorithm;
    std::unique_ptr<Storage> storage;
    ThreadPoolBuilder pool_builder;
    AllocatorBuilder allocator_builder;

    IndexBuilder(
        svs_distance_metric_t distance_metric,
        size_t dimension,
        const std::shared_ptr<Algorithm>& algorithm
    )
        : distance_metric(distance_metric)
        , dimension(dimension)
        , algorithm(algorithm->clone())
        , storage(std::make_unique<StorageSimple>(SVS_DATA_TYPE_FLOAT32))
        , pool_builder{}
        , allocator_builder{} {}

    IndexBuilder(const IndexBuilder& other)
        : distance_metric(other.distance_metric)
        , dimension(other.dimension)
        , algorithm(other.algorithm->clone())
        , storage(other.storage->clone())
        , pool_builder(other.pool_builder)
        , allocator_builder(other.allocator_builder) {}

    IndexBuilder& operator=(const IndexBuilder& other) {
        if (this != &other) {
            distance_metric = other.distance_metric;
            dimension = other.dimension;
            algorithm = other.algorithm->clone();
            storage = other.storage->clone();
            pool_builder = other.pool_builder;
            allocator_builder = other.allocator_builder;
        }
        return *this;
    }

    IndexBuilder(IndexBuilder&&) = default;
    IndexBuilder& operator=(IndexBuilder&&) = default;
    ~IndexBuilder() = default;

    void set_storage(const std::shared_ptr<Storage>& storage) {
        this->storage = storage->clone();
    }

    void set_threadpool_builder(ThreadPoolBuilder threadpool_builder) {
        std::swap(this->pool_builder, threadpool_builder);
    }

    void set_allocator_builder(AllocatorBuilder allocator_builder) {
        std::swap(this->allocator_builder, allocator_builder);
    }

    std::shared_ptr<Index> build(const svs::data::ConstSimpleDataView<float>& data);

    std::shared_ptr<Index> load(const std::filesystem::path& directory);

    std::shared_ptr<Index> copy(const std::shared_ptr<Index>& src_index);

    std::shared_ptr<DynamicIndex> build_dynamic(
        const svs::data::ConstSimpleDataView<float>& data,
        std::span<const size_t> ids,
        size_t blocksize_bytes
    );

    std::shared_ptr<DynamicIndex>
    load_dynamic(const std::filesystem::path& directory, size_t blocksize_bytes);

    // Estimate the memory a built static Vamana index would consume
    // for `num_vectors` vectors. Mirrors the accounting done by
    // svs::index::vamana::VamanaIndex::get_memory_breakdown().
    svs::index::vamana::MemoryBreakdown estimate_memory_breakdown(size_t num_vectors) const;

    // Estimate the memory a built dynamic Vamana index would consume
    // for `num_vectors` vectors. Mirrors the accounting done by
    // svs::index::vamana::MutableVamanaIndex::get_memory_breakdown().
    svs::index::vamana::MemoryBreakdown
    estimate_memory_breakdown_dynamic(size_t num_vectors, size_t blocksize_bytes) const;

    size_t estimate_search_memory(
        size_t num_queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter
    ) const;

    size_t estimate_search_memory_dynamic(
        size_t num_queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter,
        size_t blocksize_bytes
    ) const;
};
} // namespace svs::c_runtime
