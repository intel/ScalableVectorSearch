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

// Tests for the partial results returned by a cancelled Vamana search.

// header under test
#include "svs/index/vamana/extensions.h"

// svs
#include "svs/concurrent/dynamic_index.h"
#include "svs/core/distance.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/orchestrators/vamana.h"

// stl
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <unordered_set>
#include <vector>

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

namespace {

namespace cc = svs::index::vamana::concurrent;

constexpr size_t NUM_NEIGHBORS = 10;
constexpr size_t SEARCH_WINDOW = 20;
constexpr size_t MARKER_ID = std::numeric_limits<size_t>::max();

// External ID of the vector stored in row `i` of the base dataset. Deliberately
// non-contiguous so that internal and external IDs cannot be confused.
size_t external_id(size_t i) { return 1000 + 3 * i; }
size_t row_of(size_t external) { return (external - 1000) / 3; }

// A predicate that returns `false` for the first `allowed` polls and `true` afterwards.
struct CountingCancel {
    explicit CountingCancel(size_t allowed)
        : allowed_{allowed} {}
    bool operator()() { return polls_.fetch_add(1) >= allowed_; }
    size_t allowed_;
    std::atomic<size_t> polls_{0};
};

svs::index::vamana::VamanaSearchParameters search_parameters() {
    svs::index::vamana::VamanaSearchParameters sp{};
    sp.buffer_config_ = svs::index::vamana::SearchBufferConfig{SEARCH_WINDOW};
    return sp;
}

// Describes a metric: how to compute the exact distance of a result, the distance stored
// in the invalid-neighbor marker, and whether the best neighbor has the smallest distance.
struct Metric {
    std::function<float(std::span<const float>, std::span<const float>)> distance;
    float marker_distance;
    bool smaller_is_better;
};

float dot(std::span<const float> a, std::span<const float> b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += double(a[i]) * double(b[i]);
    }
    return float(sum);
}

// The marker distance is the worst possible one: the largest float for L2 and the lowest
// float for similarity metrics.
Metric l2_metric() {
    return Metric{
        [](auto a, auto b) {
            double sum = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                double d = double(a[i]) - double(b[i]);
                sum += d * d;
            }
            return float(sum);
        },
        std::numeric_limits<float>::max(),
        true};
}

Metric ip_metric() { return Metric{dot, std::numeric_limits<float>::lowest(), false}; }

Metric cosine_metric() {
    return Metric{
        [](auto a, auto b) { return dot(a, b) / std::sqrt(dot(a, a) * dot(b, b)); },
        std::numeric_limits<float>::lowest(),
        false};
}

// Number of valid entries in row `q`: the length of the prefix without the marker.
// Requires that the marker only appears as a suffix.
size_t check_row(
    const svs::QueryResult<size_t>& results,
    size_t q,
    const svs::data::SimpleData<float>& base,
    const svs::data::SimpleData<float>& queries,
    const std::function<bool(size_t)>& is_live,
    const std::function<size_t(size_t)>& to_row,
    const Metric& metric
) {
    size_t valid = 0;
    while (valid < results.n_neighbors() && results.index(q, valid) != MARKER_ID) {
        ++valid;
    }
    float previous = metric.smaller_is_better ? -std::numeric_limits<float>::infinity()
                                              : std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < results.n_neighbors(); ++j) {
        size_t id = results.index(q, j);
        float distance = results.distance(q, j);
        if (j < valid) {
            // Valid entry: live ID, exact distance, sorted nearest first.
            CATCH_REQUIRE(is_live(id));
            auto expected =
                metric.distance(base.get_datum(to_row(id)), queries.get_datum(q));
            CATCH_REQUIRE(distance == Catch::Approx(expected).epsilon(1e-4));
            CATCH_REQUIRE(
                (metric.smaller_is_better ? distance >= previous : distance <= previous)
            );
            previous = distance;
        } else {
            // Marker entries only after the valid ones.
            CATCH_REQUIRE(id == MARKER_ID);
            CATCH_REQUIRE(distance == metric.marker_distance);
        }
    }
    return valid;
}

using SearchFn =
    std::function<void(svs::QueryResultView<size_t>, const svs::lib::DefaultPredicate&)>;

// `search(results, cancel)` performs a batch search with the index under test, returning
// `k` neighbors per query.
void check_cancel_semantics(
    const SearchFn& search,
    const svs::data::SimpleData<float>& base,
    const svs::data::SimpleData<float>& queries,
    const std::function<bool(size_t)>& is_live,
    const std::function<size_t(size_t)>& to_row,
    bool single_thread,
    const Metric& metric = l2_metric(),
    size_t k = NUM_NEIGHBORS
) {
    auto make_results = [&]() { return svs::QueryResult<size_t>(queries.size(), k); };
    auto check = [&](const svs::QueryResult<size_t>& results, size_t q) {
        return check_row(results, q, base, queries, is_live, to_row, metric);
    };

    // (d) A predicate that never fires: identical to searching without a predicate.
    {
        auto expected = make_results();
        search(expected.view(), svs::lib::Returns(svs::lib::Const<false>()));
        auto actual = make_results();
        CountingCancel never{std::numeric_limits<size_t>::max()};
        search(actual.view(), std::ref(never));
        CATCH_REQUIRE(never.polls_.load() > 0);
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t j = 0; j < k; ++j) {
                CATCH_REQUIRE(actual.index(q, j) == expected.index(q, j));
                CATCH_REQUIRE(actual.distance(q, j) == expected.distance(q, j));
            }
        }
    }

    // (a) Cancelled before anything started: every slot of every row is the marker.
    {
        auto results = make_results();
        // Fill with junk to make sure the search overwrites everything.
        std::fill(results.indices().begin(), results.indices().end(), size_t{7});
        std::fill(results.distances().begin(), results.distances().end(), -1.0f);
        CountingCancel cancel{0};
        search(results.view(), std::ref(cancel));
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t j = 0; j < k; ++j) {
                CATCH_REQUIRE(results.index(q, j) == MARKER_ID);
                CATCH_REQUIRE(results.distance(q, j) == metric.marker_distance);
            }
        }
    }

    // (b), (c) Cancelled at various points. Rows must be a sorted prefix of live,
    // exactly-measured neighbors followed by markers.
    size_t partial_rows = 0;
    size_t rows_with_neighbors = 0;
    for (size_t allowed : {1, 2, 3, 5, 20, 100, 400, 1500, 6000}) {
        auto results = make_results();
        std::fill(results.indices().begin(), results.indices().end(), size_t{7});
        std::fill(results.distances().begin(), results.distances().end(), -1.0f);
        CountingCancel cancel{allowed};
        search(results.view(), std::ref(cancel));
        for (size_t q = 0; q < queries.size(); ++q) {
            size_t valid = check(results, q);
            rows_with_neighbors += (valid > 0);
            partial_rows += (valid > 0 && valid < k);
        }
    }
    CATCH_REQUIRE(rows_with_neighbors > 0);
    if (single_thread) {
        // A deterministic interruption: with one thread, the first query's single_search
        // poll passes and the first visited-node poll fires. The row then holds only what
        // initialization put into the search buffer. Later queries never start.
        auto results = make_results();
        CountingCancel cancel{1};
        search(results.view(), std::ref(cancel));
        size_t valid = check(results, 0);
        CATCH_REQUIRE(valid >= 1);
        CATCH_REQUIRE(valid < k);
        for (size_t q = 1; q < queries.size(); ++q) {
            CATCH_REQUIRE(check(results, q) == 0);
        }
        CATCH_REQUIRE(partial_rows > 0);
    }
}

// Delete every fourth vector.
std::vector<size_t> ids_to_delete(size_t n) {
    std::vector<size_t> ids;
    for (size_t i = 0; i < n; i += 4) {
        ids.push_back(external_id(i));
    }
    return ids;
}

// Delete all but every twentieth vector, so that few live vectors remain (95% deleted).
std::vector<size_t> ids_to_delete_most(size_t n) {
    std::vector<size_t> ids;
    for (size_t i = 0; i < n; ++i) {
        if (i % 20 != 0) {
            ids.push_back(external_id(i));
        }
    }
    return ids;
}

std::vector<size_t> all_external_ids(size_t n) {
    std::vector<size_t> ids(n);
    for (size_t i = 0; i < n; ++i) {
        ids[i] = external_id(i);
    }
    return ids;
}

// Whether `id` is an external ID of a live vector of the dynamic test indexes.
std::function<bool(size_t)>
is_live_external(size_t n, const std::unordered_set<size_t>& deleted) {
    return [n, &deleted](size_t id) {
        return id >= 1000 && (id - 1000) % 3 == 0 && row_of(id) < n &&
               !deleted.contains(id);
    };
}

// Static index test for any distance functor.
template <typename Distance>
void check_static(const Metric& metric, size_t k = NUM_NEIGHBORS) {
    auto base = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();
    // MIP and cosine require alpha <= 1.
    svs::index::vamana::VamanaBuildParameters parameters{
        metric.smaller_is_better ? 1.2f : 0.95f, 64, 10, 20, 10, true};

    for (size_t num_threads : {1, 3}) {
        CATCH_SECTION("Threads: " + std::to_string(num_threads)) {
            auto index = svs::Vamana::build<float>(
                parameters,
                svs::data::SimpleData<float>::load(test_dataset::data_svs_file()),
                Distance(),
                num_threads
            );
            auto sp = search_parameters();
            check_cancel_semantics(
                [&](auto results, const auto& cancel) {
                    index.search(results, queries.cview(), sp, cancel);
                },
                base,
                queries,
                [&](size_t id) { return id < base.size(); },
                [](size_t id) { return id; },
                num_threads == 1,
                metric,
                k
            );
        }
    }
}

// Checks of a dynamic index in which 95% of the vectors have been deleted. The search
// buffer then often holds fewer candidates than the window, which is when an uncancelled
// search tops it up with live vectors found outside the graph traversal. A cancelled
// search must not do that.
void check_mostly_deleted(
    const SearchFn& search,
    const svs::data::SimpleData<float>& base,
    const svs::data::SimpleData<float>& queries,
    const std::function<bool(size_t)>& is_live,
    bool single_thread
) {
    const Metric metric = l2_metric();
    auto make_results = [&]() {
        return svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);
    };

    // The uncancelled search returns a full row of live IDs (thanks to the top-up).
    {
        auto results = make_results();
        search(results.view(), svs::lib::Returns(svs::lib::Const<false>()));
        for (size_t q = 0; q < queries.size(); ++q) {
            CATCH_REQUIRE(
                check_row(results, q, base, queries, is_live, row_of, metric) ==
                NUM_NEIGHBORS
            );
        }
    }

    // Cancelled searches only ever return live IDs, and rows may be incomplete.
    size_t incomplete_rows = 0;
    for (size_t allowed : {1, 2, 3, 5, 20, 100, 400, 1500}) {
        auto results = make_results();
        CountingCancel cancel{allowed};
        search(results.view(), std::ref(cancel));
        for (size_t q = 0; q < queries.size(); ++q) {
            incomplete_rows +=
                (check_row(results, q, base, queries, is_live, row_of, metric) <
                 NUM_NEIGHBORS);
        }
    }
    CATCH_REQUIRE(incomplete_rows > 0);

    if (single_thread) {
        // Interrupted right after initialization. The buffer then holds at most the entry
        // point, so a full row can only come from a top-up, which must not happen.
        auto results = make_results();
        CountingCancel cancel{1};
        search(results.view(), std::ref(cancel));
        CATCH_REQUIRE(
            check_row(results, 0, base, queries, is_live, row_of, metric) < NUM_NEIGHBORS
        );
        for (size_t q = 1; q < queries.size(); ++q) {
            CATCH_REQUIRE(
                check_row(results, q, base, queries, is_live, row_of, metric) == 0
            );
        }
    }
}

// Single-query search with caller-provided scratch space. `external` converts an internal
// ID in the search buffer to the vector's row in the base dataset.
template <typename Index>
void check_single_query(
    Index& index,
    const svs::data::SimpleData<float>& base,
    const svs::data::SimpleData<float>& queries,
    const std::function<bool(const svs::Neighbor<uint32_t>&)>& is_live,
    const std::function<size_t(const svs::Neighbor<uint32_t>&)>& to_row
) {
    auto sp = search_parameters();
    auto scratch = index.scratchspace(sp);
    auto metric = l2_metric();
    auto query = queries.get_datum(0);

    // Uncancelled: a full window of live candidates.
    index.search(query, scratch);
    const size_t full_size = scratch.buffer.size();
    CATCH_REQUIRE(full_size >= NUM_NEIGHBORS);

    // A predicate that never fires leaves the same candidates.
    {
        std::vector<std::pair<size_t, float>> expected;
        for (const auto& n : scratch.buffer) {
            expected.emplace_back(n.id(), n.distance());
        }
        CountingCancel never{std::numeric_limits<size_t>::max()};
        index.search(query, scratch, std::ref(never));
        CATCH_REQUIRE(never.polls_.load() > 0);
        CATCH_REQUIRE(scratch.buffer.size() == expected.size());
        for (size_t j = 0; j < expected.size(); ++j) {
            CATCH_REQUIRE(scratch.buffer[j].id() == expected[j].first);
            CATCH_REQUIRE(scratch.buffer[j].distance() == expected[j].second);
        }
    }

    // Cancelled before the search starts: stale candidates are dropped, nothing is added
    // (in particular no top-up).
    {
        CountingCancel cancel{0};
        index.search(query, scratch, std::ref(cancel));
        CATCH_REQUIRE(scratch.buffer.size() == 0);
    }

    // Cancelled right after initialization: a partial, valid buffer without top-up.
    {
        CountingCancel cancel{1};
        index.search(query, scratch, std::ref(cancel));
        CATCH_REQUIRE(scratch.buffer.size() >= 1);
        CATCH_REQUIRE(scratch.buffer.size() < full_size);
        CATCH_REQUIRE(scratch.buffer.size() < SEARCH_WINDOW);
        for (const auto& n : scratch.buffer) {
            CATCH_REQUIRE(is_live(n));
            auto expected = metric.distance(base.get_datum(to_row(n)), query);
            CATCH_REQUIRE(n.distance() == Catch::Approx(expected).epsilon(1e-4));
        }
    }
}

} // namespace

CATCH_TEST_CASE("Static Vamana Cancelled Search", "[vamana][cancel]") {
    check_static<svs::distance::DistanceL2>(l2_metric());
}

CATCH_TEST_CASE("Static Vamana Cancelled Search IP", "[vamana][cancel]") {
    check_static<svs::distance::DistanceIP>(ip_metric());
}

CATCH_TEST_CASE("Static Vamana Cancelled Search Cosine", "[vamana][cancel]") {
    check_static<svs::distance::DistanceCosineSimilarity>(cosine_metric());
}

CATCH_TEST_CASE("Static Vamana Cancelled Search k > window", "[vamana][cancel]") {
    // More neighbors than the search window: the buffer is enlarged to `k`.
    static_assert(40 > SEARCH_WINDOW);
    check_static<svs::distance::DistanceL2>(l2_metric(), 40);
}

CATCH_TEST_CASE("Dynamic Vamana Cancelled Search", "[vamana][dynamic_index][cancel]") {
    using Distance = svs::distance::DistanceL2;
    auto base = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();
    auto ids = all_external_ids(base.size());
    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, test_dataset::data_blocked_f32(), ids, Distance(), 1
    );
    auto deleted = ids_to_delete(base.size());
    index.delete_entries(deleted);
    std::unordered_set<size_t> deleted_set(deleted.begin(), deleted.end());

    auto sp = search_parameters();
    for (size_t num_threads : {1, 3}) {
        CATCH_SECTION("Threads: " + std::to_string(num_threads)) {
            index.set_threadpool(svs::threads::DefaultThreadPool(num_threads));
            auto search = [&](auto results, const auto& cancel) {
                index.search(results, queries.cview(), sp, cancel);
            };
            auto is_live = is_live_external(base.size(), deleted_set);
            check_cancel_semantics(
                search, base, queries, is_live, row_of, num_threads == 1
            );
            // k larger than the search window.
            check_cancel_semantics(
                search, base, queries, is_live, row_of, num_threads == 1, l2_metric(), 40
            );
        }
    }
}

CATCH_TEST_CASE(
    "Dynamic Vamana Cancelled Search, Mostly Deleted", "[vamana][dynamic_index][cancel]"
) {
    using Distance = svs::distance::DistanceL2;
    auto base = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();
    auto ids = all_external_ids(base.size());
    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, test_dataset::data_blocked_f32(), ids, Distance(), 1
    );
    auto deleted = ids_to_delete_most(base.size());
    index.delete_entries(deleted);
    std::unordered_set<size_t> deleted_set(deleted.begin(), deleted.end());
    CATCH_REQUIRE(index.size() > 2 * SEARCH_WINDOW);

    auto sp = search_parameters();
    for (size_t num_threads : {1, 3}) {
        CATCH_SECTION("Threads: " + std::to_string(num_threads)) {
            index.set_threadpool(svs::threads::DefaultThreadPool(num_threads));
            check_mostly_deleted(
                [&](auto results, const auto& cancel) {
                    index.search(results, queries.cview(), sp, cancel);
                },
                base,
                queries,
                is_live_external(base.size(), deleted_set),
                num_threads == 1
            );
        }
    }
}

CATCH_TEST_CASE("Single Query Cancelled Search", "[vamana][dynamic_index][cancel]") {
    auto base = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();

    CATCH_SECTION("Static") {
        auto index = test_dataset::vamana::load_test_index(svs::distance::DistanceL2());
        check_single_query(
            index,
            base,
            queries,
            [&](const auto& n) { return n.id() < base.size(); },
            [](const auto& n) -> size_t { return n.id(); }
        );
    }

    CATCH_SECTION("Dynamic") {
        auto ids = all_external_ids(base.size());
        svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
        auto index = svs::index::vamana::MutableVamanaIndex(
            parameters,
            test_dataset::data_blocked_f32(),
            ids,
            svs::distance::DistanceL2(),
            1
        );
        // A buffer holding only the entry point would be topped up if not cancelled.
        auto deleted = ids_to_delete(base.size());
        index.delete_entries(deleted);
        std::unordered_set<size_t> deleted_set(deleted.begin(), deleted.end());
        auto is_live = is_live_external(base.size(), deleted_set);
        // Buffer entries hold internal IDs.
        check_single_query(
            index,
            base,
            queries,
            [&](const auto& n) { return is_live(index.translate_internal_id(n.id())); },
            [&](const auto& n) -> size_t {
                return row_of(index.translate_internal_id(n.id()));
            }
        );
    }
}

CATCH_TEST_CASE(
    "Concurrent Vamana Cancelled Search", "[vamana][concurrent][dynamic_index][cancel]"
) {
    using Distance = svs::distance::DistanceL2;
    auto base = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();
    auto ids = all_external_ids(base.size());
    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = cc::MutableVamanaIndex(
        parameters,
        cc::SegmentedBlockedData<float>::load(test_dataset::data_svs_file()),
        ids,
        Distance(),
        1
    );
    auto deleted = ids_to_delete(base.size());
    index.delete_entries(deleted);
    std::unordered_set<size_t> deleted_set(deleted.begin(), deleted.end());

    auto sp = search_parameters();
    for (size_t num_threads : {1, 3}) {
        CATCH_SECTION("Threads: " + std::to_string(num_threads)) {
            index.set_threadpool(svs::threads::DefaultThreadPool(num_threads));
            check_cancel_semantics(
                [&](auto results, const auto& cancel) {
                    index.search(results, queries.cview(), sp, cancel);
                },
                base,
                queries,
                is_live_external(base.size(), deleted_set),
                row_of,
                num_threads == 1
            );
        }
    }
}
