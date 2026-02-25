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

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/orchestrators/vamana.h>

#include <filesystem>
#include <memory>

namespace svs::c_runtime {
struct Index {
    svs_algorithm_type algorithm;
    Index(svs_algorithm_type algorithm)
        : algorithm(algorithm) {}
    virtual ~Index() = default;
    virtual svs::QueryResult<size_t> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params
    ) = 0;
    virtual void save(const std::filesystem::path& directory) = 0;
};

struct IndexVamana : public Index {
    svs::Vamana index;
    IndexVamana(svs::Vamana&& index)
        : Index{SVS_ALGORITHM_TYPE_VAMANA}
        , index(std::move(index)) {}
    ~IndexVamana() {}
    virtual svs::QueryResult<size_t> search(
        svs::data::ConstSimpleDataView<float> queries,
        size_t num_neighbors,
        const std::shared_ptr<Algorithm::SearchParams>& search_params
    ) {
        auto vamana_search_params =
            std::static_pointer_cast<AlgorithmVamana::SearchParams>(search_params);
        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

        auto params = index.get_search_parameters();
        if (vamana_search_params) {
            params = vamana_search_params->get_search_parameters();
        }

        index.search(results.view(), queries, params);
        return std::move(results);
    }

    void save(const std::filesystem::path& directory) override {
        index.save(directory / "config", directory / "graph", directory / "data");
    }
};
} // namespace svs::c_runtime
