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

#include "index_vamana.hpp"

#include "algorithm.hpp"
#include "filtered_search.hpp"
#include "index.hpp"
#include "index_builder.hpp"

#include <svs/concepts/data.h>
#include <svs/core/query_result.h>
#include <svs/lib/misc.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/orchestrators/vamana.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace svs::c_runtime {

/////////////////////////////////
// IndexVamana Implementation
IndexVamana::IndexVamana(const IndexBuilder& builder, svs::Vamana&& index)
    : Index(std::make_unique<IndexBuilder>(builder))
    , index(std::move(index)) {
    // Apply default search parameters to the index
    auto algorithm_parameters = std::static_pointer_cast<AlgorithmVamana::SearchParams>(
        this->builder->algorithm->get_default_search_params()
    );
    assert(
        algorithm_parameters && "Default search parameters must be set for the algorithm."
    );
    auto params = this->index.get_search_parameters();
    algorithm_parameters->apply_to(params);
    this->index.set_search_parameters(params);
}

std::pair<svs::QueryResult<size_t>, std::vector<size_t>> IndexVamana::search(
    svs::data::ConstSimpleDataView<float> queries,
    size_t num_neighbors,
    const std::shared_ptr<Algorithm::SearchParams>& search_params,
    const IDFilterInterface* id_filter
) {
    auto vamana_search_params =
        std::static_pointer_cast<AlgorithmVamana::SearchParams>(search_params);
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

    auto params = index.get_search_parameters();
    if (vamana_search_params) {
        vamana_search_params->apply_to(params);
    }

    if (id_filter == nullptr) {
        index.search(results.view(), queries, params);
        // assuming all neighbors found
        return {results, std::vector<size_t>(queries.size(), num_neighbors)};
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, index.size() - 1);
    auto sample_generator = [&]() -> size_t { return dist(rng); };

    auto batch_hint =
        std::max(num_neighbors, params.buffer_config_.get_search_window_size());

    auto found_neighbors_per_query = filtered_topk_search(
        index, results, queries, batch_hint, id_filter, sample_generator
    );
    return {results, found_neighbors_per_query};
}

void IndexVamana::set_num_threads(size_t num_threads) {
    builder->pool_builder.resize(num_threads);
    index.set_threadpool(builder->pool_builder.build());
}

/////////////////////////////////////
// DynamicIndexVamana Implementation
DynamicIndexVamana::DynamicIndexVamana(
    const IndexBuilder& builder, svs::DynamicVamana&& index
)
    : DynamicIndex(std::make_unique<IndexBuilder>(builder))
    , index(std::move(index)) {
    auto all_ids = this->index.all_ids();
    assert(
        !all_ids.empty() &&
        "DynamicVamana index should have at least one ID after construction."
    );
    auto [min_it, max_it] = std::minmax_element(all_ids.begin(), all_ids.end());
    min_id = (min_it == all_ids.end()) ? 0 : *min_it;
    max_id = (max_it == all_ids.end()) ? 0 : *max_it;

    // Apply default search parameters to the index
    auto algorithm_parameters = std::static_pointer_cast<AlgorithmVamana::SearchParams>(
        this->builder->algorithm->get_default_search_params()
    );
    assert(
        algorithm_parameters && "Default search parameters must be set for the algorithm."
    );
    auto params = this->index.get_search_parameters();
    algorithm_parameters->apply_to(params);
    this->index.set_search_parameters(params);
}

std::pair<svs::QueryResult<size_t>, std::vector<size_t>> DynamicIndexVamana::search(
    svs::data::ConstSimpleDataView<float> queries,
    size_t num_neighbors,
    const std::shared_ptr<Algorithm::SearchParams>& search_params,
    const IDFilterInterface* id_filter
) {
    auto vamana_search_params =
        std::static_pointer_cast<AlgorithmVamana::SearchParams>(search_params);
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);

    auto params = index.get_search_parameters();
    if (vamana_search_params) {
        vamana_search_params->apply_to(params);
    }

    if (id_filter == nullptr) {
        index.search(results.view(), queries, params);
        return {results, std::vector<size_t>(queries.size(), num_neighbors)};
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
    const size_t max_attempts = std::max((max_id - min_id) / (index.size() + 1), size_t{4});

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

    auto found_neighbors_per_query = filtered_topk_search(
        index, results, queries, batch_hint, id_filter, sample_generator
    );
    return {results, found_neighbors_per_query};
}

size_t DynamicIndexVamana::add_points(
    svs::data::ConstSimpleDataView<float> new_points, std::span<const size_t> ids
) {
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

size_t DynamicIndexVamana::delete_points(std::span<const size_t> ids) {
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

void DynamicIndexVamana::set_num_threads(size_t num_threads) {
    this->builder->pool_builder.resize(num_threads);
    index.set_threadpool(this->builder->pool_builder.build());
}

} // namespace svs::c_runtime
