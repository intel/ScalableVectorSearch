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
#include "threadpool.hpp"
#include "types_support.hpp"

#include <svs/core/data/simple.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/index.h>
#include <svs/lib/misc.h>

#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace svs::c_runtime {

struct IndexBuilder;
struct Index {
    std::unique_ptr<IndexBuilder> builder;
    explicit Index(std::unique_ptr<IndexBuilder> builder)
        : builder(std::move(builder)) {}
    virtual ~Index() = default;
    const IndexBuilder& get_builder() const { return *builder; }
    virtual std::pair<svs::QueryResult<size_t>, std::vector<size_t>> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter = nullptr
    ) = 0;
    virtual void save(const std::filesystem::path& directory) = 0;
    virtual size_t dimensions() const = 0;
    virtual float get_distance(size_t id, std::span<const float> query) const = 0;
    virtual void
    reconstruct_at(svs::data::SimpleDataView<float> dst, std::span<const size_t> ids) = 0;
    virtual size_t get_num_threads() const = 0;
    virtual void set_num_threads(size_t num_threads) = 0;
    virtual svs::index::vamana::MemoryBreakdown get_memory_breakdown() const = 0;
};

struct DynamicIndex : public Index {
    explicit DynamicIndex(std::unique_ptr<IndexBuilder> builder)
        : Index(std::move(builder)) {}
    ~DynamicIndex() = default;

    virtual size_t add_points(
        svs::data::ConstSimpleDataView<float> new_points, std::span<const size_t> ids
    ) = 0;
    virtual size_t delete_points(std::span<const size_t> ids) = 0;
    virtual bool has_id(size_t id) const = 0;
    virtual void consolidate() = 0;
    virtual void compact(size_t batchsize) = 0;
};
} // namespace svs::c_runtime
