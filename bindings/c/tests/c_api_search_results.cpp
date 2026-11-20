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

// C API
#include "svs/c/svs_c.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

// Test utilities
#include "c_api_test_utils.h"

// Standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr size_t DIMENSION = 16;
constexpr size_t NUM_VECTORS = 500;
constexpr size_t NUM_QUERIES = 8;
constexpr size_t K = 10;
// Brute-force neighbors used to pick a radius that yields non-trivial rows.
constexpr size_t NEIGHBORS_IN_RANGE = 20;
// Below NEIGHBORS_IN_RANGE so that some rows are truncated.
constexpr size_t MAX_RESULTS = 5;
constexpr size_t NUM_DELETED = 50;
constexpr size_t BLOCK_SIZE = 1 << 20;

std::vector<float> random_vectors(size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> result(count * DIMENSION);
    for (auto& x : result) {
        x = dist(rng);
    }
    return result;
}

bool filter_is_odd(void* /*self*/, size_t id) { return (id % 2) == 1; }
// Well above the true selectivity of filter_is_odd (0.5).
float filter_rate_overestimated(void* /*self*/) { return 0.9f; }

using Keep = std::function<bool(size_t)>;
const Keep keep_all = [](size_t) { return true; };

struct SearchFixture {
    svs_distance_metric_t metric;
    svs_error_h error = svs_error_create();
    svs_algorithm_h algorithm = nullptr;
    svs_index_builder_h builder = nullptr;
    std::vector<float> data = random_vectors(NUM_VECTORS, 1);
    std::vector<float> queries = random_vectors(NUM_QUERIES, 2);

    explicit SearchFixture(svs_distance_metric_t metric = SVS_DISTANCE_METRIC_EUCLIDEAN)
        : metric{metric} {
        algorithm = svs_algorithm_create_vamana(32, 64, 100, error);
        builder = svs_index_builder_create(metric, DIMENSION, algorithm, error);
        svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_NATIVE, 2, error);
    }

    ~SearchFixture() {
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    SearchFixture(const SearchFixture&) = delete;
    SearchFixture& operator=(const SearchFixture&) = delete;

    svs_index_h build() {
        return svs_index_build(builder, data.data(), NUM_VECTORS, error);
    }

    svs_index_h build_dynamic(svs_sync_kind_t sync_kind) {
        std::vector<size_t> ids(NUM_VECTORS);
        std::iota(ids.begin(), ids.end(), size_t{0});
        svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        params.blocksize_bytes = BLOCK_SIZE;
        params.sync_kind = sync_kind;
        return svs_index_build_dynamic_ex(
            builder, data.data(), ids.data(), NUM_VECTORS, &params, error
        );
    }

    float distance(size_t q, size_t id) const {
        const float* a = queries.data() + q * DIMENSION;
        const float* b = data.data() + id * DIMENSION;
        switch (metric) {
            case SVS_DISTANCE_METRIC_DOT_PRODUCT:
                return inner_product_distance(a, b, DIMENSION);
            case SVS_DISTANCE_METRIC_COSINE:
                return cosine_distance(a, b, DIMENSION);
            default:
                return euclidean_distance(a, b, DIMENSION);
        }
    }

    // L2 reports distances (smaller is closer); inner product and cosine report
    // similarities (larger is closer).
    bool closer(float a, float b) const {
        return metric == SVS_DISTANCE_METRIC_EUCLIDEAN ? a < b : a > b;
    }

    float pick_radius() const {
        std::vector<float> kth;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            std::vector<float> dists(NUM_VECTORS);
            for (size_t id = 0; id < NUM_VECTORS; ++id) {
                dists[id] = distance(q, id);
            }
            auto cmp = [&](float a, float b) { return closer(a, b); };
            std::nth_element(
                dists.begin(), dists.begin() + NEIGHBORS_IN_RANGE - 1, dists.end(), cmp
            );
            kth.push_back(dists[NEIGHBORS_IN_RANGE - 1]);
        }
        auto cmp = [&](float a, float b) { return closer(a, b); };
        std::nth_element(kth.begin(), kth.begin() + NUM_QUERIES / 2, kth.end(), cmp);
        return kth[NUM_QUERIES / 2];
    }

    // A radius no vector can be strictly closer than.
    float empty_radius() const {
        return metric == SVS_DISTANCE_METRIC_EUCLIDEAN ? 0.0f
                                                       : std::numeric_limits<float>::max();
    }

    std::vector<size_t> expected(size_t q, float radius, const Keep& keep) const {
        std::vector<size_t> ids;
        for (size_t id = 0; id < NUM_VECTORS; ++id) {
            if (keep(id) && closer(distance(q, id), radius)) {
                ids.push_back(id);
            }
        }
        return ids;
    }
};

struct RecallCounter {
    size_t found = 0;
    size_t expected = 0;
};

// Validates row `row` of `results` as the range search result of query `q`.
void check_range_row(
    const SearchFixture& fx,
    const svs_search_results_t& results,
    size_t row,
    size_t q,
    float radius,
    const Keep& keep,
    RecallCounter& recall
) {
    const size_t* ids = nullptr;
    const float* dists = nullptr;
    size_t count = 0;
    svs_search_results_row(&results, row, &ids, &dists, &count);
    CATCH_REQUIRE((count == 0 || (ids != nullptr && dists != nullptr)));

    const auto want = fx.expected(q, radius, keep);
    std::vector<size_t> seen;
    for (size_t j = 0; j < count; ++j) {
        CATCH_REQUIRE(ids[j] < NUM_VECTORS);
        CATCH_REQUIRE(keep(ids[j]));
        CATCH_REQUIRE(fx.closer(dists[j], radius));
        const float reference = fx.distance(q, ids[j]);
        CATCH_REQUIRE(
            std::abs(dists[j] - reference) <= 1e-3f * std::max(1.0f, std::abs(reference))
        );
        if (j > 0) {
            CATCH_REQUIRE_FALSE(fx.closer(dists[j], dists[j - 1]));
        }
        if (std::find(want.begin(), want.end(), ids[j]) != want.end()) {
            ++recall.found;
        }
        seen.push_back(ids[j]);
    }
    std::sort(seen.begin(), seen.end());
    CATCH_REQUIRE(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
    recall.expected += want.size();
}

void require_recall(const RecallCounter& recall) {
    CATCH_CAPTURE(recall.found, recall.expected);
    CATCH_REQUIRE(recall.expected > 0);
    CATCH_REQUIRE(recall.found * 10 >= recall.expected * 9);
}

void require_empty_rows(const svs_search_results_t& results, size_t num_queries) {
    CATCH_REQUIRE(results.num_queries == num_queries);
    CATCH_REQUIRE(results.total_results == 0);
    for (size_t q = 0; q <= num_queries; ++q) {
        CATCH_REQUIRE(results.offsets[q] == 0);
    }
}

} // namespace

CATCH_TEST_CASE("C API Search Results Buffers", "[c_api][index][search][results]") {
    SearchFixture fx;
    svs_index_h index = fx.build();
    CATCH_REQUIRE(index != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));

    CATCH_SECTION("Library-owned buffers are reused for the same shape") {
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, fx.error
        ));
        CATCH_REQUIRE(results.owns_buffers);
        CATCH_REQUIRE(results.offsets_capacity >= NUM_QUERIES + 1);
        CATCH_REQUIRE(results.results_capacity >= NUM_QUERIES * K);
        CATCH_REQUIRE(results.total_results == NUM_QUERIES * K);
        CATCH_REQUIRE(results.offsets[NUM_QUERIES] == results.total_results);

        const size_t* offsets = results.offsets;
        const size_t* indices = results.indices;
        const float* distances = results.distances;
        CATCH_REQUIRE(svs_index_search_topk(
            index, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, fx.error
        ));
        CATCH_REQUIRE(results.offsets == offsets);
        CATCH_REQUIRE(results.indices == indices);
        CATCH_REQUIRE(results.distances == distances);

        // A larger request grows the library-owned buffers.
        CATCH_REQUIRE(svs_index_search_topk(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            2 * K,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(results.results_capacity >= NUM_QUERIES * 2 * K);
        CATCH_REQUIRE(results.total_results == NUM_QUERIES * 2 * K);

        svs_search_results_free(&results);
        CATCH_REQUIRE_FALSE(results.owns_buffers);
        CATCH_REQUIRE(results.offsets == nullptr);
        CATCH_REQUIRE(results.indices == nullptr);
        CATCH_REQUIRE(results.distances == nullptr);
        CATCH_REQUIRE(results.offsets_capacity == 0);
        CATCH_REQUIRE(results.results_capacity == 0);
        CATCH_REQUIRE(results.num_queries == 0);
        CATCH_REQUIRE(results.total_results == 0);

        // Freeing twice, or freeing NULL, is a no-op.
        svs_search_results_free(&results);
        svs_search_results_free(nullptr);
    }

    CATCH_SECTION("Caller-owned buffers") {
        std::vector<size_t> offsets(NUM_QUERIES + 1);
        std::vector<size_t> indices(NUM_QUERIES * K);
        std::vector<float> distances(NUM_QUERIES * K);
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS(
            offsets.data(), indices.data(), distances.data(), offsets.size(), indices.size()
        );

        CATCH_REQUIRE(svs_index_search_topk(
            index, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        CATCH_REQUIRE_FALSE(results.owns_buffers);
        CATCH_REQUIRE(results.offsets == offsets.data());
        CATCH_REQUIRE(results.indices == indices.data());
        CATCH_REQUIRE(results.distances == distances.data());
        CATCH_REQUIRE(results.num_queries == NUM_QUERIES);
        CATCH_REQUIRE(results.total_results == NUM_QUERIES * K);
        for (size_t q = 0; q <= NUM_QUERIES; ++q) {
            CATCH_REQUIRE(offsets[q] == q * K);
        }

        // Too small for the request: the call fails instead of reallocating.
        CATCH_REQUIRE_FALSE(svs_index_search_topk(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            K + 1,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE_FALSE(results.owns_buffers);
        CATCH_REQUIRE(results.indices == indices.data());

        CATCH_REQUIRE_FALSE(svs_index_search_topk(
            index,
            fx.queries.data(),
            NUM_QUERIES + 1,
            1,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);

        // Freeing only resets the counts of caller-owned buffers.
        svs_search_results_free(&results);
        CATCH_REQUIRE(results.num_queries == 0);
        CATCH_REQUIRE(results.total_results == 0);
        CATCH_REQUIRE(results.offsets == offsets.data());
        CATCH_REQUIRE(results.indices == indices.data());
        CATCH_REQUIRE(results.distances == distances.data());
        CATCH_REQUIRE(results.offsets_capacity == offsets.size());
        CATCH_REQUIRE(results.results_capacity == indices.size());
    }

    CATCH_SECTION("Incompatible descriptor is rejected") {
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        results.struct_size = sizeof(svs_search_results_t) + 1;
        CATCH_REQUIRE_FALSE(svs_index_search_topk(
            index, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE_FALSE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            1.0f,
            0,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE(results.offsets == nullptr);

        results = SVS_INIT_SEARCH_RESULTS();
        results.version = svs_get_version() + 1;
        CATCH_REQUIRE_FALSE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            1.0f,
            0,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    CATCH_SECTION("Row accessor") {
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, fx.error
        ));

        const size_t* ids = nullptr;
        const float* dists = nullptr;
        size_t count = 0;
        svs_search_results_row(&results, 1, &ids, &dists, &count);
        CATCH_REQUIRE(count == K);
        CATCH_REQUIRE(ids == results.indices + K);
        CATCH_REQUIRE(dists == results.distances + K);

        // Out-of-range rows are empty; all out pointers are optional.
        svs_search_results_row(&results, NUM_QUERIES, &ids, &dists, &count);
        CATCH_REQUIRE(ids == nullptr);
        CATCH_REQUIRE(dists == nullptr);
        CATCH_REQUIRE(count == 0);
        svs_search_results_row(&results, 0, nullptr, nullptr, &count);
        CATCH_REQUIRE(count == K);

        svs_search_results_free(&results);
    }

    svs_index_free(index);
}

CATCH_TEST_CASE("C API Range Search", "[c_api][index][search][range]") {
    const auto metric = GENERATE(
        SVS_DISTANCE_METRIC_EUCLIDEAN,
        SVS_DISTANCE_METRIC_DOT_PRODUCT,
        SVS_DISTANCE_METRIC_COSINE
    );
    CATCH_CAPTURE(metric);

    SearchFixture fx{metric};
    svs_index_h index = fx.build();
    CATCH_REQUIRE(index != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));
    const float radius = fx.pick_radius();
    CATCH_CAPTURE(radius);

    CATCH_SECTION("Matches brute force") {
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        CATCH_REQUIRE(results.num_queries == NUM_QUERIES);
        CATCH_REQUIRE(results.offsets[0] == 0);
        CATCH_REQUIRE(results.offsets[NUM_QUERIES] == results.total_results);

        RecallCounter recall;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            check_range_row(fx, results, q, q, radius, keep_all, recall);
        }
        require_recall(recall);
        svs_search_results_free(&results);
    }

    CATCH_SECTION("Explicit search parameters") {
        svs_search_params_h params = svs_search_params_create_vamana(20, fx.error);
        CATCH_REQUIRE(params != nullptr);
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &results,
            params,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));

        RecallCounter recall;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            check_range_row(fx, results, q, q, radius, keep_all, recall);
        }
        CATCH_REQUIRE(recall.found > 0);
        svs_search_results_free(&results);
        svs_search_params_free(params);
    }

    CATCH_SECTION("With ID filter") {
        svs_id_filter_interface_ops odd_ops = SVS_INIT_ID_FILTER_OPS(filter_is_odd);
        svs_id_filter_interface id_filter = SVS_MAKE_INTERFACE(nullptr, odd_ops);

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &results,
            nullptr,
            &id_filter,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));

        const Keep keep_odd = [](size_t id) { return (id % 2) == 1; };
        RecallCounter recall;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            check_range_row(fx, results, q, q, radius, keep_odd, recall);
        }
        require_recall(recall);
        svs_search_results_free(&results);
    }

    CATCH_SECTION("Filter rate above the observed hit rate") {
        svs_id_filter_interface_ops odd_ops =
            SVS_INIT_ID_FILTER_OPS(filter_is_odd, filter_rate_overestimated);
        svs_id_filter_interface id_filter = SVS_MAKE_INTERFACE(nullptr, odd_ops);

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &results,
            nullptr,
            &id_filter,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        require_empty_rows(results, NUM_QUERIES);
        svs_search_results_free(&results);
    }

    CATCH_SECTION("Radius excluding every vector") {
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            fx.empty_radius(),
            0,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        require_empty_rows(results, NUM_QUERIES);
        svs_search_results_free(&results);
    }

    CATCH_SECTION("Caller-owned buffers") {
        svs_search_results_t reference = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &reference,
            nullptr,
            nullptr,
            fx.error
        ));
        const size_t total = reference.total_results;
        CATCH_REQUIRE(total > 1);

        std::vector<size_t> offsets(NUM_QUERIES + 1);
        std::vector<size_t> indices(total);
        std::vector<float> distances(total);

        // Exactly large enough.
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS(
            offsets.data(), indices.data(), distances.data(), offsets.size(), total
        );
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        CATCH_REQUIRE_FALSE(results.owns_buffers);
        CATCH_REQUIRE(results.total_results == total);
        CATCH_REQUIRE(std::equal(offsets.begin(), offsets.end(), reference.offsets));
        CATCH_REQUIRE(std::equal(indices.begin(), indices.end(), reference.indices));

        // One result short: the call fails and leaves the descriptor unchanged.
        results = SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS(
            offsets.data(), indices.data(), distances.data(), offsets.size(), total - 1
        );
        CATCH_REQUIRE_FALSE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE(results.num_queries == 0);
        CATCH_REQUIRE(results.total_results == 0);
        CATCH_REQUIRE_FALSE(results.owns_buffers);
        CATCH_REQUIRE(results.indices == indices.data());

        // Too few offsets fails before searching.
        results = SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS(
            offsets.data(), indices.data(), distances.data(), NUM_QUERIES, total
        );
        CATCH_REQUIRE_FALSE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &results,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);

        svs_search_results_free(&reference);
    }

    CATCH_SECTION("Result limit") {
        svs_search_results_t unlimited = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &unlimited,
            nullptr,
            nullptr,
            fx.error
        ));

        // Caller-owned buffers sized for the limit always suffice.
        std::vector<size_t> offsets(NUM_QUERIES + 1);
        std::vector<size_t> indices(NUM_QUERIES * MAX_RESULTS);
        std::vector<float> distances(NUM_QUERIES * MAX_RESULTS);
        svs_search_results_t limited = SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS(
            offsets.data(), indices.data(), distances.data(), offsets.size(), indices.size()
        );
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            MAX_RESULTS,
            &limited,
            nullptr,
            nullptr,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));

        RecallCounter recall;
        bool truncated = false;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            check_range_row(fx, limited, q, q, radius, keep_all, recall);

            const size_t* all_ids = nullptr;
            size_t all_count = 0;
            svs_search_results_row(&unlimited, q, &all_ids, nullptr, &all_count);
            const size_t* ids = nullptr;
            size_t count = 0;
            svs_search_results_row(&limited, q, &ids, nullptr, &count);

            // A limited row is the closest prefix of the unlimited one.
            CATCH_REQUIRE(count == std::min(all_count, MAX_RESULTS));
            CATCH_REQUIRE(std::equal(ids, ids + count, all_ids));
            truncated = truncated || all_count > MAX_RESULTS;
        }
        CATCH_REQUIRE(truncated);

        svs_search_results_free(&unlimited);
    }

    CATCH_SECTION("Result limit with ID filter") {
        svs_id_filter_interface_ops odd_ops = SVS_INIT_ID_FILTER_OPS(filter_is_odd);
        svs_id_filter_interface id_filter = SVS_MAKE_INTERFACE(nullptr, odd_ops);

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            MAX_RESULTS,
            &results,
            nullptr,
            &id_filter,
            fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));

        const Keep keep_odd = [](size_t id) { return (id % 2) == 1; };
        RecallCounter recall;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            CATCH_REQUIRE(results.offsets[q + 1] - results.offsets[q] <= MAX_RESULTS);
            check_range_row(fx, results, q, q, radius, keep_odd, recall);
        }
        CATCH_REQUIRE(recall.found > 0);
        svs_search_results_free(&results);
    }

    svs_index_free(index);
}

CATCH_TEST_CASE("C API Range Search Arguments", "[c_api][index][search][range][error]") {
    SearchFixture fx;
    svs_index_h index = fx.build();
    CATCH_REQUIRE(index != nullptr);
    svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();

    CATCH_REQUIRE_FALSE(svs_index_search_range(
        nullptr,
        fx.queries.data(),
        NUM_QUERIES,
        1.0f,
        0,
        &results,
        nullptr,
        nullptr,
        fx.error
    ));
    CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);

    CATCH_REQUIRE_FALSE(svs_index_search_range(
        index, nullptr, NUM_QUERIES, 1.0f, 0, &results, nullptr, nullptr, fx.error
    ));
    CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);

    CATCH_REQUIRE_FALSE(svs_index_search_range(
        index, fx.queries.data(), 0, 1.0f, 0, &results, nullptr, nullptr, fx.error
    ));
    CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);

    CATCH_REQUIRE_FALSE(svs_index_search_range(
        index, fx.queries.data(), NUM_QUERIES, 1.0f, 0, nullptr, nullptr, nullptr, fx.error
    ));
    CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);

    // The error handle is optional.
    CATCH_REQUIRE_FALSE(svs_index_search_range(
        nullptr,
        fx.queries.data(),
        NUM_QUERIES,
        1.0f,
        0,
        &results,
        nullptr,
        nullptr,
        nullptr
    ));

    svs_search_results_free(&results);
    svs_index_free(index);
}

CATCH_TEST_CASE("C API Dynamic Range Search", "[c_api][index][dynamic][search][range]") {
    const auto sync_kind =
        GENERATE(SVS_SYNC_KIND_NONE, SVS_SYNC_KIND_GLOBAL, SVS_SYNC_KIND_FINE_GRAIN);
    CATCH_CAPTURE(sync_kind);
    const bool fine_grain = sync_kind == SVS_SYNC_KIND_FINE_GRAIN;

    SearchFixture fx;
    svs_index_h index = fx.build_dynamic(sync_kind);
    CATCH_REQUIRE(index != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));

    // Leave the deletions unconsolidated so the search must skip them.
    std::vector<size_t> deleted(NUM_DELETED);
    std::iota(deleted.begin(), deleted.end(), size_t{0});
    // FINE_GRAIN dynamic index only supports deleting one point at a time.
    const size_t delete_batch = fine_grain ? 1 : NUM_DELETED;
    for (size_t i = 0; i < NUM_DELETED; i += delete_batch) {
        size_t count = 0;
        CATCH_REQUIRE(svs_index_dynamic_delete_points(
            index, deleted.data() + i, delete_batch, &count, fx.error
        ));
        CATCH_REQUIRE(count == delete_batch);
    }

    const float radius = fx.pick_radius();
    CATCH_CAPTURE(radius);

    // FINE_GRAIN dynamic index only supports one query at a time.
    auto search_all = [&](svs_id_filter_i id_filter, size_t max_results, const Keep& keep) {
        RecallCounter recall;
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        const size_t batch = fine_grain ? 1 : NUM_QUERIES;
        for (size_t q = 0; q < NUM_QUERIES; q += batch) {
            CATCH_REQUIRE(svs_index_search_range(
                index,
                fx.queries.data() + q * DIMENSION,
                batch,
                radius,
                max_results,
                &results,
                nullptr,
                id_filter,
                fx.error
            ));
            CATCH_REQUIRE(svs_error_ok(fx.error));
            CATCH_REQUIRE(results.num_queries == batch);
            for (size_t row = 0; row < batch; ++row) {
                if (max_results != 0) {
                    const size_t count = results.offsets[row + 1] - results.offsets[row];
                    CATCH_REQUIRE(count <= max_results);
                }
                check_range_row(fx, results, row, q + row, radius, keep, recall);
            }
        }
        svs_search_results_free(&results);
        return recall;
    };

    const Keep keep_live = [](size_t id) { return id >= NUM_DELETED; };

    CATCH_SECTION("Skips deleted vectors") {
        require_recall(search_all(nullptr, 0, keep_live));
    }

    CATCH_SECTION("With ID filter") {
        svs_id_filter_interface_ops odd_ops = SVS_INIT_ID_FILTER_OPS(filter_is_odd);
        svs_id_filter_interface id_filter = SVS_MAKE_INTERFACE(nullptr, odd_ops);
        require_recall(search_all(&id_filter, 0, [](size_t id) {
            return id >= NUM_DELETED && (id % 2) == 1;
        }));
    }

    CATCH_SECTION("Result limit") {
        CATCH_REQUIRE(search_all(nullptr, MAX_RESULTS, keep_live).found > 0);
    }

    CATCH_SECTION("Batch size limits") {
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        const bool ok = svs_index_search_range(
            index,
            fx.queries.data(),
            NUM_QUERIES,
            radius,
            0,
            &results,
            nullptr,
            nullptr,
            fx.error
        );
        CATCH_REQUIRE(ok != fine_grain);
        if (fine_grain) {
            CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_NOT_IMPLEMENTED);
        }
        svs_search_results_free(&results);
    }

    svs_index_free(index);
}
