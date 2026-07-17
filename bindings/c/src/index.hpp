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
#include "filtered_search.hpp"
#include "threadpool.hpp"

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/lib/misc.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/orchestrators/vamana.h>

#include <filesystem>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace svs::c_runtime {

struct Index {
    svs_algorithm_type algorithm;
    ThreadPoolBuilder pool_builder;
    Index(svs_algorithm_type algorithm, ThreadPoolBuilder pool_builder)
        : algorithm(algorithm)
        , pool_builder(pool_builder) {}
    virtual ~Index() = default;
    virtual svs::QueryResult<size_t> search(
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
};

struct DynamicIndex : public Index {
    DynamicIndex(svs_algorithm_type algorithm, ThreadPoolBuilder pool_builder)
        : Index(algorithm, pool_builder) {}
    ~DynamicIndex() = default;

    virtual size_t add_points(
        svs::data::ConstSimpleDataView<float> new_points, std::span<const size_t> ids
    ) = 0;
    virtual size_t delete_points(std::span<const size_t> ids) = 0;
    virtual bool has_id(size_t id) const = 0;
    virtual void consolidate() = 0;
    virtual void compact(size_t batchsize) = 0;
};

struct IndexVamana : public Index {
    svs::Vamana index;
    IndexVamana(svs::Vamana&& index, ThreadPoolBuilder pool_builder)
        : Index{SVS_ALGORITHM_TYPE_VAMANA, pool_builder}
        , index(std::move(index)) {}
    ~IndexVamana() = default;
    svs::QueryResult<size_t> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter
    ) override {
        auto vamana_search_params =
            std::static_pointer_cast<AlgorithmVamana::SearchParams>(search_params);
        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

        auto params = index.get_search_parameters();
        if (vamana_search_params) {
            params = vamana_search_params->get_search_parameters();
        }

        if (id_filter == nullptr) {
            index.search(results.view(), queries, params);
            return results;
        }

        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, index.size() - 1);
        auto sample_generator = [&]() -> size_t { return dist(rng); };

        auto batch_hint =
            std::max(num_neighbors, params.buffer_config_.get_search_window_size());

        filtered_topk_search(
            index, results, queries, batch_hint, id_filter, sample_generator
        );
        return results;
    }

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

    void set_num_threads(size_t num_threads) override {
        pool_builder.resize(num_threads);
        index.set_threadpool(pool_builder.build());
    }
};

struct DynamicIndexVamana : public DynamicIndex {
    svs::DynamicVamana index;
    size_t min_id = 0; // Track the minimum ID added to the index
    size_t max_id = 0; // Track the maximum ID added to the index
    DynamicIndexVamana(svs::DynamicVamana&& index, ThreadPoolBuilder pool_builder)
        : DynamicIndex(SVS_ALGORITHM_TYPE_VAMANA, pool_builder)
        , index(std::move(index)) {
        auto all_ids = this->index.all_ids();
        assert(
            !all_ids.empty() &&
            "DynamicVamana index should have at least one ID after construction."
        );
        auto [min_it, max_it] = std::minmax_element(all_ids.begin(), all_ids.end());
        min_id = (min_it == all_ids.end()) ? 0 : *min_it;
        max_id = (max_it == all_ids.end()) ? 0 : *max_it;
    }
    ~DynamicIndexVamana() = default;

    svs::QueryResult<size_t> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params,
        const IDFilterInterface* id_filter
    ) override {
        auto vamana_search_params =
            std::static_pointer_cast<AlgorithmVamana::SearchParams>(search_params);
        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

        auto params = index.get_search_parameters();
        if (vamana_search_params) {
            params = vamana_search_params->get_search_parameters();
        }

        if (id_filter == nullptr) {
            index.search(results.view(), queries, params);
            return results;
        }

        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(min_id, max_id);
        // DynamicVamana index IDs provided by user and may have any values and gaps, so we
        // need to sample until we find a valid ID.
        // The most reliable way would be get all IDs and sample from them, but that may be
        // expensive for large indexes. So we sample from the range of IDs and check if they
        // exist in the index. If not, we sample again. We limit the number of attempts to
        // avoid infinite loops in case of sparse IDs. The maximum number of
        // attempts is set to the ratio of the ID range to the index size, or at least 4
        // attempts. This ensures that we have a reasonable chance of finding a valid ID
        // without excessive sampling.
        // Note: (index.size() + 1) - to avoid division by zero in case the index is empty.
        const size_t max_attempts =
            std::max((max_id - min_id) / (index.size() + 1), size_t{4});

        auto sample_generator = [&]() -> size_t {
            for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
                size_t id = dist(rng);
                if (index.has_id(id)) {
                    return id;
                }
            }
            return static_cast<size_t>(-1); // Return an invalid ID if no valid ID is found
        };

        auto batch_hint =
            std::max(num_neighbors, params.buffer_config_.get_search_window_size());

        filtered_topk_search(
            index, results, queries, batch_hint, id_filter, sample_generator
        );
        return results;
    }

    void save(const std::filesystem::path& directory) override {
        index.save(directory / "config", directory / "graph", directory / "data");
    }

    size_t dimensions() const override { return index.dimensions(); }

    size_t add_points(
        svs::data::ConstSimpleDataView<float> new_points, std::span<const size_t> ids
    ) override {
        // Track the maximum ID added to the index for ids generator
        auto [min_it, max_it] = std::minmax_element(ids.begin(), ids.end());
        if (min_it != ids.end()) {
            min_id = std::min(min_id, *min_it);
        }
        if (max_it != ids.end()) {
            max_id = std::max(max_id, *max_it);
        }
        auto old_size = index.size();
        index.add_points(new_points, ids);
        // TODO: This is a bit of a hack - we should ideally return the number of points
        // actually added, but for now we can just return index size change.
        return index.size() - old_size;
    }

    size_t delete_points(std::span<const size_t> ids) override {
        std::vector<size_t> ids_to_delete;
        ids_to_delete.reserve(ids.size());

        for (auto id : ids) {
            if (index.has_id(id)) {
                ids_to_delete.push_back(id);
            }
        }

        if (!ids_to_delete.empty()) {
            index.delete_points(svs::lib::as_const_span(ids_to_delete));
        }
        return ids_to_delete.size();
    }

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

    void set_num_threads(size_t num_threads) override {
        pool_builder.resize(num_threads);
        index.set_threadpool(pool_builder.build());
    }
};
} // namespace svs::c_runtime
