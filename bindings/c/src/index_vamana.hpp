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

#include <atomic>
#include <filesystem>
#include <memory>
#include <ostream>
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

    void save(std::ostream& stream) override { index.save(stream); }

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

    size_t size() const override { return index.size(); }
};

struct DynamicIndexVamana : public DynamicIndex {
    svs::DynamicVamana index;
    // Bounds of the IDs ever added, used to sample IDs for filtered search. Atomic since
    // ConcurrentIndexVamana adds points under a shared lock.
    std::atomic<size_t> min_id = 0;
    std::atomic<size_t> max_id = 0;
    DynamicIndexVamana(
        const IndexBuilder& builder,
        svs::DynamicVamana&& index,
        svs_sync_kind_t sync_kind = SVS_SYNC_KIND_NONE
    );

    ~DynamicIndexVamana() = default;

    std::pair<svs::QueryResult<size_t>, std::vector<size_t>> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter
    ) override;

    void save(const std::filesystem::path& directory) override;

    void save(std::ostream& stream) override {
        // Saving consolidates and compacts the index, so it requires exclusive access.
        auto lock = write_lock();
        index.save(stream);
    }

    size_t dimensions() const override { return index.dimensions(); }

    size_t add_points(
        svs::data::ConstSimpleDataView<float> new_points, std::span<const size_t> ids
    ) override;

    size_t delete_points(std::span<const size_t> ids) override;

    bool has_id(size_t id) const override {
        auto lock = read_lock();
        return index.has_id(id);
    }

    float get_distance(size_t id, std::span<const float> query) const override {
        auto lock = read_lock();
        return index.get_distance(id, query);
    }

    void reconstruct_at(svs::data::SimpleDataView<float> dst, std::span<const size_t> ids)
        override {
        auto lock = read_lock();
        index.reconstruct_at(dst, ids);
    }

    void consolidate() override;

    void compact(size_t batchsize) override;

    size_t get_num_threads() const override {
        auto lock = read_lock();
        return index.get_num_threads();
    }

    void set_num_threads(size_t num_threads) override;

    svs::index::vamana::MemoryBreakdown get_memory_breakdown() const override {
        auto lock = read_lock();
        return index.get_memory_breakdown();
    }

    size_t size() const override {
        auto lock = read_lock();
        return index.size();
    }

  protected:
    void track_id_range(std::span<const size_t> ids);
    size_t delete_points_unlocked(std::span<const size_t> ids);
    void compact_unlocked(size_t batchsize);
};

/// Dynamic index backed by svs::ConcurrentDynamicVamana.
///
/// The index synchronizes add/delete/consolidate/compact itself, so they take the shared
/// lock alongside searches. Operations that are not thread-safe in the index (save,
/// set_num_threads) and whole-index reads (memory breakdown, use as a conversion source)
/// take the exclusive lock.
struct ConcurrentIndexVamana : public DynamicIndexVamana {
    ConcurrentIndexVamana(
        const IndexBuilder& builder, svs::DynamicVamana&& index, svs_sync_kind_t sync_kind
    );

    size_t add_points(
        svs::data::ConstSimpleDataView<float> new_points, std::span<const size_t> ids
    ) override;

    size_t delete_points(std::span<const size_t> ids) override;

    void consolidate() override {
        auto lock = read_lock();
        index.consolidate();
    }

    void compact(size_t batchsize) override {
        auto lock = read_lock();
        compact_unlocked(batchsize);
    }

    svs::index::vamana::MemoryBreakdown get_memory_breakdown() const override {
        auto lock = write_lock();
        return index.get_memory_breakdown();
    }
};
} // namespace svs::c_runtime
