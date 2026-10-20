// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

// Checks of the filtered top-k search paths that depend on the filter pass rate. Shared by
// the static and the dynamic index tests: both call the same filtered search.

#include "svs/c/svs_c.h"

#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <unordered_set>
#include <vector>

namespace filtered_topk_checks {

// Filter: ids that are multiples of the modulus stored in `self` (pass rate 1 / modulus).
inline bool is_multiple(void* self, size_t id) {
    return id % *static_cast<const size_t*>(self) == 0;
}
inline float rate_unknown(void*) { return 0.0f; }
inline float rate_half(void*) { return 0.5f; }
inline float rate_one_percent(void*) { return 0.01f; }

struct Dataset {
    size_t num_vectors = 5000;
    size_t num_queries = 50;
    size_t dimension = 32;
    size_t k = 10;
    std::vector<float> data;
    std::vector<float> queries;

    Dataset() {
        // Random data: the periodic test generator produces a disconnected graph.
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        data.resize(num_vectors * dimension);
        queries.resize(num_queries * dimension);
        for (auto& x : data) {
            x = uniform(rng);
        }
        for (auto& x : queries) {
            x = uniform(rng);
        }
    }

    // Exact top-k ids of query `q` among the ids with id % modulus == 0.
    std::vector<size_t> ground_truth(size_t q, size_t modulus) const {
        std::vector<std::pair<float, size_t>> dist;
        const float* query = queries.data() + q * dimension;
        for (size_t id = 0; id < num_vectors; id += modulus) {
            float d = 0;
            for (size_t j = 0; j < dimension; ++j) {
                float diff = query[j] - data[id * dimension + j];
                d += diff * diff;
            }
            dist.emplace_back(d, id);
        }
        std::partial_sort(dist.begin(), dist.begin() + k, dist.end());
        std::vector<size_t> top;
        for (size_t i = 0; i < k; ++i) {
            top.push_back(dist[i].second);
        }
        return top;
    }
};

struct Outcome {
    double recall = 0;
    size_t min_found = 0; // fewest results returned for any query
    size_t max_found = 0; // most results returned for any query
};

// Runs a top-k search with the filter "id % modulus == 0" (modulus 0: no filter) and
// compares with the exact filtered results. Checks that every returned id passes the
// filter and, if `check_order`, that distances are non-decreasing (results gathered from
// several iterator batches are sorted within a batch only).
inline Outcome search(
    svs_index_h index,
    const Dataset& ds,
    svs_search_params_h params,
    size_t modulus,
    float (*filter_rate)(void*),
    bool check_order
) {
    svs_error_h error = svs_error_create();
    svs_id_filter_interface_ops ops = SVS_INIT_ID_FILTER_OPS(is_multiple, *filter_rate);
    svs_id_filter_interface filter = SVS_MAKE_INTERFACE(&modulus, ops);
    svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
    CATCH_REQUIRE(svs_index_search_topk(
        index,
        ds.queries.data(),
        ds.num_queries,
        ds.k,
        &results,
        params,
        modulus == 0 ? nullptr : &filter,
        error
    ));
    CATCH_REQUIRE(svs_error_ok(error));
    CATCH_REQUIRE(results.num_queries == ds.num_queries);

    Outcome out;
    out.min_found = ds.k;
    size_t hits = 0;
    size_t expected = 0;
    for (size_t q = 0; q < ds.num_queries; ++q) {
        const size_t begin = results.offsets[q];
        const size_t found = results.offsets[q + 1] - begin;
        CATCH_REQUIRE(found <= ds.k);
        out.min_found = std::min(out.min_found, found);
        out.max_found = std::max(out.max_found, found);
        auto truth = ds.ground_truth(q, std::max<size_t>(modulus, 1));
        std::unordered_set<size_t> truth_set(truth.begin(), truth.end());
        expected += truth.size();
        for (size_t j = 0; j < found; ++j) {
            const size_t id = results.indices[begin + j];
            CATCH_REQUIRE((modulus == 0 || id % modulus == 0));
            if (check_order && j > 0) {
                CATCH_REQUIRE(
                    results.distances[begin + j] >= results.distances[begin + j - 1]
                );
            }
            hits += truth_set.count(id);
        }
    }
    svs_search_results_free(&results);
    svs_error_free(error);
    out.recall = static_cast<double>(hits) / static_cast<double>(expected);
    return out;
}

// `index` holds ds.data with ids 0..num_vectors-1.
inline void run(svs_index_h index, const Dataset& ds) {
    svs_error_h error = svs_error_create();
    svs_search_params_h params = svs_search_params_create_vamana(100, error);
    CATCH_REQUIRE(params != nullptr);

    // Reference: the same search without a filter.
    const auto unfiltered = search(index, ds, params, 0, rate_unknown, true);
    CATCH_REQUIRE(unfiltered.min_found == ds.k);

    // Pass rate 1/2 and 1 (unknown filter rate): the greedy oversampled search. Every query
    // gets k results and the recall is comparable to the unfiltered search.
    for (size_t modulus : {1, 2}) {
        const auto out = search(index, ds, params, modulus, rate_unknown, true);
        CATCH_REQUIRE(out.min_found == ds.k);
        CATCH_REQUIRE(out.recall >= unfiltered.recall - 0.05);
    }

    // Pass rates 1/10 and 1/100: the batch iterator. A window of 400 gathers enough hits in
    // the first batch, so this does not depend on how a search resumes with a larger
    // window. Every query still gets k results at 1/10; at 1/100 all returned ids must
    // pass the filter (checked in search()).
    svs_search_params_h wide = svs_search_params_create_vamana(400, error);
    CATCH_REQUIRE(wide != nullptr);
    const auto tenth = search(index, ds, wide, 10, rate_unknown, false);
    CATCH_REQUIRE(tenth.min_found == ds.k);
    CATCH_REQUIRE(tenth.recall >= 0.8);
    const auto hundredth = search(index, ds, wide, 100, rate_unknown, false);
    CATCH_REQUIRE(hundredth.max_found <= ds.k);
    svs_search_params_free(wide);

    // A positive filter rate above the actual pass rate (1/20) stops the search early with
    // an empty result; a rate below it does not.
    const auto stopped = search(index, ds, params, 20, rate_half, false);
    CATCH_REQUIRE(stopped.max_found == 0);
    const auto allowed = search(index, ds, params, 20, rate_one_percent, false);
    CATCH_REQUIRE(allowed.min_found == ds.k);

    svs_search_params_free(params);
    svs_error_free(error);
}

} // namespace filtered_topk_checks
