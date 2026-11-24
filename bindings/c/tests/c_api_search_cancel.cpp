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
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t DIMENSION = 32;
constexpr size_t NUM_VECTORS = 1000;
constexpr size_t NUM_QUERIES = 5;
constexpr size_t K = 10;
constexpr size_t SEARCH_WINDOW = 100;
constexpr size_t NUM_THREADS = 4;
constexpr size_t INVALID_ID = std::numeric_limits<size_t>::max();

enum class IndexKind { Static, DynamicNone, DynamicGlobal, DynamicFineGrain };

// Counts every poll. Returns true from poll number `cancel_after + 1` on.
struct CountingCancel {
    std::atomic<size_t> calls{0};
    size_t cancel_after = static_cast<size_t>(-1);
};

bool counting_is_cancelled(void* userdata) {
    auto* state = static_cast<CountingCancel*>(userdata);
    return state->calls.fetch_add(1, std::memory_order_relaxed) >= state->cancel_after;
}

bool filter_is_odd(void* /*self*/, size_t id) { return (id % 2) == 1; }
float filter_rate_odd(void* /*self*/) { return 0.5f; }

// Random data: the periodic generate_test_data() pattern repeats every few rows, which
// starves the filtered search (all top neighbors share a parity) and makes it flaky.
std::vector<float> random_vectors(size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(count * DIMENSION);
    for (auto& x : out) {
        x = dist(rng);
    }
    return out;
}

struct CancelFixture {
    svs_error_h error = svs_error_create();
    svs_algorithm_h algorithm = nullptr;
    svs_index_builder_h builder = nullptr;
    svs_index_h index = nullptr;
    svs_search_params_h params = nullptr;
    svs_distance_metric_t metric;
    std::vector<float> data;
    std::vector<float> queries = random_vectors(NUM_QUERIES, 2);
    // External ID -> row in `data` for every vector that is live in the index.
    std::unordered_map<size_t, size_t> live;
    size_t num_queries;

    // `scattered`: dynamic indexes get non-contiguous external IDs and every fifth
    // vector is deleted after the build.
    CancelFixture(
        IndexKind kind, svs_distance_metric_t metric, size_t num_threads, bool scattered
    )
        // FINE_GRAIN dynamic indexes only search one query at a time.
        : metric(metric)
        , data(random_vectors(NUM_VECTORS, 1))
        , num_queries(kind == IndexKind::DynamicFineGrain ? 1 : NUM_QUERIES) {
        algorithm = svs_algorithm_create_vamana(16, 32, SEARCH_WINDOW, error);
        builder = svs_index_builder_create(metric, DIMENSION, algorithm, error);
        svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, num_threads, error
        );
        if (kind == IndexKind::Static) {
            index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
            for (size_t i = 0; i < NUM_VECTORS; ++i) {
                live[i] = i;
            }
        } else {
            std::vector<size_t> ids(NUM_VECTORS);
            for (size_t i = 0; i < NUM_VECTORS; ++i) {
                ids[i] = scattered ? 5001 + i + i / 3 : i; // a gap every third ID
                live[ids[i]] = i;
            }
            svs_dynamic_index_params_t dyn_params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
            dyn_params.blocksize_bytes = 1 << 20;
            dyn_params.sync_kind = kind == IndexKind::DynamicNone ? SVS_SYNC_KIND_NONE
                                   : kind == IndexKind::DynamicGlobal
                                       ? SVS_SYNC_KIND_GLOBAL
                                       : SVS_SYNC_KIND_FINE_GRAIN;
            index = svs_index_build_dynamic_ex(
                builder, data.data(), ids.data(), NUM_VECTORS, &dyn_params, error
            );
            if (scattered) {
                // One at a time: FINE_GRAIN only supports single-ID deletes.
                for (size_t i = 0; i < NUM_VECTORS; i += 5) {
                    svs_index_dynamic_delete_points(index, &ids[i], 1, nullptr, error);
                    live.erase(ids[i]);
                }
            }
        }
        params = svs_search_params_create_vamana(SEARCH_WINDOW, error);
    }

    ~CancelFixture() {
        svs_search_params_free(params);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CancelFixture(const CancelFixture&) = delete;
    CancelFixture& operator=(const CancelFixture&) = delete;

    bool search(svs_search_results_t* results, bool filtered) {
        svs_id_filter_interface_ops ops =
            SVS_INIT_ID_FILTER_OPS(filter_is_odd, filter_rate_odd);
        svs_id_filter_interface filter = SVS_MAKE_INTERFACE(nullptr, ops);
        return svs_index_search_topk(
            index,
            queries.data(),
            num_queries,
            K,
            results,
            params,
            filtered ? &filter : nullptr,
            error
        );
    }

    bool set_callback(CountingCancel* state) {
        return svs_search_params_set_cancel_callback(
            params, state ? counting_is_cancelled : nullptr, state, error
        );
    }

    float distance(size_t query, size_t id) const {
        const float* q = queries.data() + query * DIMENSION;
        const float* x = data.data() + live.at(id) * DIMENSION;
        return metric == SVS_DISTANCE_METRIC_EUCLIDEAN
                   ? euclidean_distance(q, x, DIMENSION)
                   : inner_product_distance(q, x, DIMENSION);
    }

    // Every returned entry must be a live external ID with its true distance, rows must
    // be nearest first, and the library's invalid-neighbor marker must never leak out.
    void
    require_valid_partial_rows(const svs_search_results_t& results, bool filtered) const {
        CATCH_REQUIRE(results.num_queries == num_queries);
        CATCH_REQUIRE(results.offsets[0] == 0);
        CATCH_REQUIRE(results.offsets[num_queries] == results.total_results);
        for (size_t q = 0; q < num_queries; ++q) {
            const size_t begin = results.offsets[q];
            const size_t end = results.offsets[q + 1];
            CATCH_REQUIRE(begin <= end);
            CATCH_REQUIRE(end - begin <= K);
            std::unordered_set<size_t> seen;
            for (size_t i = begin; i < end; ++i) {
                const size_t id = results.indices[i];
                CATCH_REQUIRE(id != INVALID_ID);
                CATCH_REQUIRE(live.count(id) == 1);
                CATCH_REQUIRE(seen.insert(id).second);
                if (filtered) {
                    CATCH_REQUIRE(filter_is_odd(nullptr, id));
                }
                const float expected = distance(q, id);
                CATCH_REQUIRE(
                    std::abs(results.distances[i] - expected) <=
                    1e-4f + 1e-3f * std::abs(expected)
                );
                // Filtered rows are concatenated iterator batches and are only ordered
                // within a batch (also when not cancelled), so check order unfiltered.
                if (i > begin && !filtered) {
                    if (metric == SVS_DISTANCE_METRIC_EUCLIDEAN) {
                        CATCH_REQUIRE(results.distances[i - 1] <= results.distances[i]);
                    } else {
                        CATCH_REQUIRE(results.distances[i - 1] >= results.distances[i]);
                    }
                }
            }
        }
    }
};

void require_same_results(
    const svs_search_results_t& actual, const svs_search_results_t& expected
) {
    CATCH_REQUIRE(actual.num_queries == expected.num_queries);
    CATCH_REQUIRE(actual.total_results == expected.total_results);
    for (size_t q = 0; q <= expected.num_queries; ++q) {
        CATCH_REQUIRE(actual.offsets[q] == expected.offsets[q]);
    }
    for (size_t i = 0; i < expected.total_results; ++i) {
        CATCH_REQUIRE(actual.indices[i] == expected.indices[i]);
        CATCH_REQUIRE(actual.distances[i] == expected.distances[i]);
    }
}

void require_cancelled(CancelFixture& fx, bool success) {
    CATCH_REQUIRE_FALSE(success);
    CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_CANCELLED);
}

} // namespace

CATCH_TEST_CASE("C API Search Cancel Callback", "[c_api][index][search][cancel]") {
    const auto kind = GENERATE(
        IndexKind::Static,
        IndexKind::DynamicNone,
        IndexKind::DynamicGlobal,
        IndexKind::DynamicFineGrain
    );
    const auto metric =
        GENERATE(SVS_DISTANCE_METRIC_EUCLIDEAN, SVS_DISTANCE_METRIC_DOT_PRODUCT);
    const bool filtered = GENERATE(false, true);
    const size_t num_threads = GENERATE(size_t{1}, NUM_THREADS);
    // Non-contiguous IDs plus deletions (dynamic indexes only).
    const bool scattered = GENERATE(false, true);
    if (kind == IndexKind::Static && scattered) {
        return; // scattered IDs and deletions only apply to dynamic indexes
    }
    CancelFixture fx{kind, metric, num_threads, scattered};
    CATCH_REQUIRE(fx.index != nullptr);
    CATCH_REQUIRE(fx.params != nullptr);
    // With several threads concurrent polls can overshoot the cancel point.
    const size_t max_overshoot = num_threads - 1;

    svs_search_results_t baseline = SVS_INIT_SEARCH_RESULTS();
    CATCH_REQUIRE(fx.search(&baseline, filtered));
    CATCH_REQUIRE(svs_error_ok(fx.error));
    CATCH_INFO(
        "kind=" << int(kind) << " metric=" << int(metric) << " filtered=" << filtered
                << " threads=" << num_threads << " scattered=" << scattered
    );
    CATCH_REQUIRE(baseline.total_results > 0);
    fx.require_valid_partial_rows(baseline, filtered);

    // Number of polls a complete, uncancelled search makes.
    CountingCancel never;
    CATCH_REQUIRE(fx.set_callback(&never));
    svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
    CATCH_REQUIRE(fx.search(&results, filtered));
    CATCH_REQUIRE(svs_error_ok(fx.error));
    const size_t uncancelled_calls = never.calls.load();
    CATCH_REQUIRE(uncancelled_calls > 0);

    CATCH_SECTION("Callback that never cancels returns the baseline results") {
        require_same_results(results, baseline);
    }

    CATCH_SECTION("NULL callback is identical to no callback") {
        CATCH_REQUIRE(fx.set_callback(nullptr));
        CATCH_REQUIRE(fx.search(&results, filtered));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        require_same_results(results, baseline);
        CATCH_REQUIRE(never.calls.load() == uncancelled_calls);
    }

    CATCH_SECTION("Set then unset") {
        CountingCancel always;
        always.cancel_after = 0;
        CATCH_REQUIRE(fx.set_callback(&always));
        CATCH_REQUIRE(fx.set_callback(nullptr));
        CATCH_REQUIRE(fx.search(&results, filtered));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        require_same_results(results, baseline);
        CATCH_REQUIRE(always.calls.load() == 0);
    }

    CATCH_SECTION("Cancel before the search starts returns no results") {
        CountingCancel always;
        always.cancel_after = 0;
        CATCH_REQUIRE(fx.set_callback(&always));
        require_cancelled(fx, fx.search(&results, filtered));
        CATCH_REQUIRE(results.num_queries == fx.num_queries);
        CATCH_REQUIRE(results.total_results == 0);
        for (size_t q = 0; q <= fx.num_queries; ++q) {
            CATCH_REQUIRE(results.offsets[q] == 0);
        }
        // The first poll latches the request, so the callback is not consulted again
        // (at most one concurrent first poll per thread).
        CATCH_REQUIRE(always.calls.load() >= 1);
        CATCH_REQUIRE(always.calls.load() <= 1 + max_overshoot);
    }

    CATCH_SECTION("Cancel in the middle of the search returns the neighbors found so far") {
        size_t partial_total = 0;
        bool any_fewer = false;
        // Poll counts the counting callback lets through before it asks to cancel.
        const size_t cancel_points[] = {1, 5, uncancelled_calls / 2, uncancelled_calls - 1};
        for (size_t cancel_after : cancel_points) {
            CountingCancel mid;
            mid.cancel_after = cancel_after;
            CATCH_REQUIRE(fx.set_callback(&mid));
            svs_search_results_free(&results);
            results = SVS_INIT_SEARCH_RESULTS();
            bool ok = fx.search(&results, filtered);
            CATCH_INFO("cancel_after=" << cancel_after << " calls=" << mid.calls.load());
            require_cancelled(fx, ok);
            fx.require_valid_partial_rows(results, filtered);
            // A row may already be full when the cancel arrives.
            CATCH_REQUIRE(results.total_results <= baseline.total_results);
            CATCH_REQUIRE(mid.calls.load() >= cancel_after + 1);
            CATCH_REQUIRE(mid.calls.load() <= cancel_after + 1 + max_overshoot);
            CATCH_REQUIRE(mid.calls.load() <= uncancelled_calls + max_overshoot);
            partial_total += results.total_results;
            any_fewer |= results.total_results < baseline.total_results;
        }
        // At least one cancel point must have left some neighbors behind.
        CATCH_REQUIRE(partial_total > 0);
        // With several queries, the ones that never started return nothing.
        CATCH_REQUIRE((fx.num_queries == 1 || any_fewer));
    }

    CATCH_SECTION("A search after a cancelled one is unaffected") {
        CountingCancel mid;
        mid.cancel_after = uncancelled_calls / 2;
        CATCH_REQUIRE(fx.set_callback(&mid));
        require_cancelled(fx, fx.search(&results, filtered));
        CATCH_REQUIRE(fx.set_callback(nullptr));
        CATCH_REQUIRE(fx.search(&results, filtered));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        require_same_results(results, baseline);
    }

    svs_search_results_free(&results);
    svs_search_results_free(&baseline);
}

CATCH_TEST_CASE(
    "C API Search Cancel Callback argument validation", "[c_api][search_params][cancel]"
) {
    svs_error_h error = svs_error_create();

    CATCH_SECTION("NULL params") {
        CATCH_REQUIRE_FALSE(svs_search_params_set_cancel_callback(
            nullptr, counting_is_cancelled, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    CATCH_SECTION("NULL error handle") {
        svs_search_params_h params = svs_search_params_create_vamana(100, nullptr);
        CATCH_REQUIRE(params != nullptr);
        CATCH_REQUIRE(svs_search_params_set_cancel_callback(
            params, counting_is_cancelled, nullptr, nullptr
        ));
        svs_search_params_free(params);
    }

    svs_error_free(error);
}
