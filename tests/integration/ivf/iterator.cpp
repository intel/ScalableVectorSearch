/*
 * Copyright 2025 Intel Corporation
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

// svs
#include "svs/orchestrators/dynamic_ivf.h"
#include "svs/orchestrators/ivf.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"

// stl
#include <unordered_set>
#include <vector>

namespace {

const size_t NUM_CLUSTERS = 10;

// Helper to build a static IVF index using the orchestrator
svs::IVF make_static_ivf_index() {
    auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto distance = svs::DistanceL2{};
    size_t num_threads = 2;
    size_t intra_query_threads = 2;

    // Build clustering
    auto build_params = svs::index::ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto clustering =
        svs::IVF::build_clustering<float>(build_params, data, distance, num_threads);

    // Assemble the index
    return svs::IVF::assemble_from_clustering<float>(
        std::move(clustering), data, distance, num_threads, intra_query_threads
    );
}

// Helper to build a dynamic IVF index using the orchestrator
svs::DynamicIVF make_dynamic_ivf_index() {
    auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto distance = svs::DistanceL2{};
    size_t num_threads = 2;
    size_t intra_query_threads = 2;

    // Generate IDs for all data points
    std::vector<size_t> initial_ids(data.size());
    std::iota(initial_ids.begin(), initial_ids.end(), 0);

    // Build clustering
    auto build_params = svs::index::ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto clustering =
        svs::IVF::build_clustering<float>(build_params, data, distance, num_threads);

    // Assemble the dynamic index
    return svs::DynamicIVF::assemble_from_clustering<float>(
        std::move(clustering),
        std::move(data),
        initial_ids,
        distance,
        num_threads,
        intra_query_threads
    );
}

void test_static_iterator() {
    auto index = make_static_ivf_index();
    auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());

    // Set batch size
    size_t batchsize = 10;

    // Create a batch iterator over the index for the first query
    auto query = queries.get_datum(0);
    auto query_span = std::span<const float>(query.data(), query.size());
    auto itr = index.batch_iterator(query_span);

    // Ensure the iterator is initialized correctly. No search happens at this point.
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 0);
    CATCH_REQUIRE(!itr.done());

    // Get first batch
    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == batchsize);
    CATCH_REQUIRE(itr.batch_number() == 1);
    CATCH_REQUIRE(!itr.done());

    // Get results and verify no duplicates
    std::unordered_set<size_t> seen_ids;
    auto results = itr.results();
    CATCH_REQUIRE(results.size() == batchsize);
    for (const auto& neighbor : results) {
        CATCH_REQUIRE(seen_ids.find(neighbor.id()) == seen_ids.end());
        seen_ids.insert(neighbor.id());
    }

    // Get second batch
    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == batchsize);
    CATCH_REQUIRE(itr.batch_number() == 2);

    // Verify no duplicates across batches
    results = itr.results();
    for (const auto& neighbor : results) {
        CATCH_REQUIRE(seen_ids.find(neighbor.id()) == seen_ids.end());
        seen_ids.insert(neighbor.id());
    }

    // Continue until done
    size_t max_iterations = index.size() / batchsize + 10;
    size_t iterations = 2;
    while (!itr.done() && iterations < max_iterations) {
        itr.next(batchsize);
        for (const auto& neighbor : itr.results()) {
            CATCH_REQUIRE(seen_ids.find(neighbor.id()) == seen_ids.end());
            seen_ids.insert(neighbor.id());
        }
        ++iterations;
    }

    // Should eventually be done
    CATCH_REQUIRE(itr.done());

    // Test update with new query
    auto query2 = queries.get_datum(1);
    auto query2_span = std::span<const float>(query2.data(), query2.size());
    itr.update(query2_span);

    // Iterator should be reset
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 0);
    CATCH_REQUIRE(!itr.done());

    // Should be able to get results for new query
    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == batchsize);
    CATCH_REQUIRE(itr.batch_number() == 1);
}

void test_dynamic_iterator() {
    auto index = make_dynamic_ivf_index();
    auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());

    // Set batch size
    size_t batchsize = 10;

    // Create a batch iterator over the index for the first query
    auto query = queries.get_datum(0);
    auto query_span = std::span<const float>(query.data(), query.size());
    auto itr = index.batch_iterator(query_span);

    // Ensure the iterator is initialized correctly
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 0);
    CATCH_REQUIRE(!itr.done());

    // Get first batch
    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == batchsize);
    CATCH_REQUIRE(itr.batch_number() == 1);
    CATCH_REQUIRE(!itr.done());

    // Verify all returned IDs are valid (exist in the index)
    auto results = itr.results();
    for (const auto& neighbor : results) {
        CATCH_REQUIRE(index.has_id(neighbor.id()));
    }

    // Get second batch and verify no duplicates
    std::unordered_set<size_t> seen_ids;
    for (const auto& neighbor : results) {
        seen_ids.insert(neighbor.id());
    }

    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == batchsize);
    CATCH_REQUIRE(itr.batch_number() == 2);

    results = itr.results();
    for (const auto& neighbor : results) {
        CATCH_REQUIRE(seen_ids.find(neighbor.id()) == seen_ids.end());
        CATCH_REQUIRE(index.has_id(neighbor.id()));
        seen_ids.insert(neighbor.id());
    }
}

void test_iterator_restart() {
    auto index = make_static_ivf_index();
    auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());

    size_t batchsize = 10;
    auto query = queries.get_datum(0);
    auto query_span = std::span<const float>(query.data(), query.size());
    auto itr = index.batch_iterator(query_span);

    // Get first batch
    itr.next(batchsize);
    CATCH_REQUIRE(itr.batch_number() == 1);

    auto first_results = std::vector<size_t>();
    for (const auto& neighbor : itr.results()) {
        first_results.push_back(neighbor.id());
    }

    // Signal restart
    itr.restart_next_search();

    // Get next batch
    itr.next(batchsize);
    CATCH_REQUIRE(itr.batch_number() == 2);

    // After restart, the new batch should not duplicate any IDs from first batch
    for (const auto& neighbor : itr.results()) {
        CATCH_REQUIRE(
            std::find(first_results.begin(), first_results.end(), neighbor.id()) ==
            first_results.end()
        );
    }
}

void test_iterator_extra_buffer_capacity() {
    auto index = make_static_ivf_index();
    auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());

    auto query = queries.get_datum(0);
    auto query_span = std::span<const float>(query.data(), query.size());

    // Create iterator with custom extra buffer capacity
    size_t extra_buffer = 50;
    auto itr = index.batch_iterator(query_span, extra_buffer);

    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 0);

    // Get first batch
    itr.next(20);
    CATCH_REQUIRE(itr.size() == 20);
    CATCH_REQUIRE(itr.batch_number() == 1);
    CATCH_REQUIRE(!itr.done());
}

} // namespace

CATCH_TEST_CASE("IVF Iterator Integration", "[integration][ivf][iterator]") {
    CATCH_SECTION("Static IVF Iterator") { test_static_iterator(); }

    CATCH_SECTION("Dynamic IVF Iterator") { test_dynamic_iterator(); }

    CATCH_SECTION("Iterator Restart") { test_iterator_restart(); }

    CATCH_SECTION("Iterator Extra Buffer Capacity") {
        test_iterator_extra_buffer_capacity();
    }
}
