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
#include "threadpool.hpp"

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/orchestrators/vamana.h>

#include <filesystem>
#include <memory>
#include <span>

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
        const std::shared_ptr<Algorithm::SearchParams>& search_params
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
        const std::shared_ptr<Algorithm::SearchParams>& search_params
    ) override {
        auto vamana_search_params =
            std::static_pointer_cast<AlgorithmVamana::SearchParams>(search_params);
        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

        auto params = index.get_search_parameters();
        if (vamana_search_params) {
            params = vamana_search_params->get_search_parameters();
        }

        index.search(results.view(), queries, params);
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
    DynamicIndexVamana(svs::DynamicVamana&& index, ThreadPoolBuilder pool_builder)
        : DynamicIndex(SVS_ALGORITHM_TYPE_VAMANA, pool_builder)
        , index(std::move(index)) {}
    ~DynamicIndexVamana() = default;

    svs::QueryResult<size_t> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params
    ) override {
        auto vamana_search_params =
            std::static_pointer_cast<AlgorithmVamana::SearchParams>(search_params);
        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

        auto params = index.get_search_parameters();
        if (vamana_search_params) {
            params = vamana_search_params->get_search_parameters();
        }

        index.search(results.view(), queries, params);
        return results;
    }

    void save(const std::filesystem::path& directory) override {
        index.save(directory / "config", directory / "graph", directory / "data");
    }

    size_t dimensions() const override { return index.dimensions(); }

    size_t add_points(
        svs::data::ConstSimpleDataView<float> new_points, std::span<const size_t> ids
    ) override {
        auto old_size = index.size();
        index.add_points(new_points, ids);
        // TODO: This is a bit of a hack - we should ideally return the number of points
        // actually added, but for now we can just return index size change.
        return index.size() - old_size;
    }

    size_t delete_points(std::span<const size_t> ids) override {
        auto old_size = index.size();
        index.delete_points(ids);
        // TODO: This is a bit of a hack - we should ideally return the number of points
        // actually deleted, but for now we can just return index size change.
        return old_size - index.size();
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
