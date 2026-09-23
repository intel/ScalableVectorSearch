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

#include "index.hpp"

#include <svs/concepts/data.h>
#include <svs/core/query_result.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/orchestrators/vamana.h>

#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace svs::c_runtime {

struct IndexVamana : public Index {
    svs::Vamana index;
    IndexVamana(const IndexBuilder& builder, svs::Vamana&& index);
    ~IndexVamana() = default;

    std::pair<svs::QueryResult<size_t>, std::vector<size_t>> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter
    ) override;

    void save(const std::filesystem::path& directory) override {
        index.save(directory / "config", directory / "graph", directory / "data");
    }

    size_t dimensions() const override { return index.dimensions(); }

    float get_distance(size_t id, std::span<const float> query) const override {
        return index.get_distance(id, query);
    }

    void reconstruct_at(svs::data::SimpleDataView<float> dst, std::span<const size_t> ids)
        override {
        index.reconstruct_at(dst, ids);
    }

    size_t get_num_threads() const override { return index.get_num_threads(); }

    void set_num_threads(size_t num_threads) override;

    svs::index::vamana::MemoryBreakdown get_memory_breakdown() const override {
        return index.get_memory_breakdown();
    }
};

struct DynamicIndexVamana : public DynamicIndex {
    svs::DynamicVamana index;
    size_t min_id = 0; // Track the minimum ID added to the index
    size_t max_id = 0; // Track the maximum ID added to the index
    DynamicIndexVamana(const IndexBuilder& builder, svs::DynamicVamana&& index);

    ~DynamicIndexVamana() = default;

    std::pair<svs::QueryResult<size_t>, std::vector<size_t>> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter
    ) override;

    void save(const std::filesystem::path& directory) override {
        index.save(directory / "config", directory / "graph", directory / "data");
    }

    size_t dimensions() const override { return index.dimensions(); }

    size_t add_points(
        svs::data::ConstSimpleDataView<float> new_points, std::span<const size_t> ids
    ) override;

    size_t delete_points(std::span<const size_t> ids) override;

    bool has_id(size_t id) const override { return index.has_id(id); }

    float get_distance(size_t id, std::span<const float> query) const override {
        return index.get_distance(id, query);
    }

    void reconstruct_at(svs::data::SimpleDataView<float> dst, std::span<const size_t> ids)
        override {
        index.reconstruct_at(dst, ids);
    }

    void consolidate() override { index.consolidate(); }

    void compact(size_t batchsize) override {
        if (batchsize == 0) {
            index.compact(); // Use default batch size
        } else {
            index.compact(batchsize);
        }
    }

    size_t get_num_threads() const override { return index.get_num_threads(); }

    void set_num_threads(size_t num_threads) override;

    svs::index::vamana::MemoryBreakdown get_memory_breakdown() const override {
        return index.get_memory_breakdown();
    }
};
} // namespace svs::c_runtime
