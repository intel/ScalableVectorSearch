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

// header under test
#include "svs/index/ivf/iterator.h"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/index/ivf/clustering.h"
#include "svs/index/ivf/dynamic_ivf.h"
#include "svs/index/ivf/index.h"

// tests
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

const size_t QUERIES_TO_CHECK = 10;
const size_t NUM_CLUSTERS = 10;
const size_t N = 128; // dimensions

// Common test routines for the static and dynamic indexes.
template <typename Index, typename IDChecker = svs::lib::Returns<svs::lib::Const<true>>>
void check(
    Index& index,
    svs::data::ConstSimpleDataView<float> queries,
    svs::data::ConstSimpleDataView<uint32_t> groundtruth,
    IDChecker& checker
) {
    const size_t num_neighbors = 100;
    const auto batchsizes = std::vector<size_t>{{10, 20, 25, 50, 100}};

    CATCH_REQUIRE(index.size() > num_neighbors);

    auto id_to_distance = std::unordered_map<size_t, float>();
    auto id_buffer = std::vector<size_t>();

    CATCH_REQUIRE(checker(id_to_distance));

    auto from_iterator = std::unordered_set<size_t>();
    for (size_t query_index = 0; query_index < QUERIES_TO_CHECK; ++query_index) {
        auto query = queries.get_datum(query_index);

        // Set up search parameters for full search
        auto search_params = svs::index::ivf::IVFSearchParameters();
        search_params.n_probes_ = NUM_CLUSTERS; // Search all clusters for accuracy
        search_params.k_reorder_ = num_neighbors;

        // Perform a single, full-precision search to obtain reference results.
        auto scratch = index.scratchspace(search_params, num_neighbors);
        index.search(query, scratch);
        auto& buffer = scratch.buffer_leaves[0];
        buffer.sort();

        id_to_distance.clear();
        id_buffer.clear();
        for (const auto& neighbor : buffer) {
            size_t id = [&]() -> size_t {
                if constexpr (Index::needs_id_translation) {
                    return index.translate_internal_id(neighbor.id());
                } else {
                    return neighbor.id();
                }
            }();
            id_to_distance.insert({id, neighbor.distance()});
            id_buffer.push_back(id);
        }

        // Ensure we have reasonable recall between.
        CATCH_REQUIRE(
            svs::lib::count_intersect(id_buffer, groundtruth.get_datum(query_index)) >=
            0.8 * num_neighbors
        );

        // Begin performing batch searches.
        for (auto batchsize : batchsizes) {
            CATCH_REQUIRE(num_neighbors % batchsize == 0);
            size_t num_batches = num_neighbors / batchsize;

            auto iterator = index.make_batch_iterator(query);
            CATCH_REQUIRE(iterator.size() == 0);
            iterator.next(batchsize);

            from_iterator.clear();
            size_t similar_count = 0;

            // IDs returned from the most recent batch.
            auto ids_returned_this_batch = std::vector<size_t>();
            for (size_t batch = 0; batch < num_batches; ++batch) {
                // Make sure the batch number is the same.
                CATCH_REQUIRE(iterator.batch_number() == batch + 1);
                ids_returned_this_batch.clear();
                for (auto i : iterator) {
                    auto id = i.id();
                    // Make sure that this ID has not been returned yet.
                    CATCH_REQUIRE(!from_iterator.contains(id));
                    auto itr = id_to_distance.find(id);
                    if (itr != id_to_distance.end()) {
                        // Make sure the returned distances match.
                        CATCH_REQUIRE(itr->second == i.distance());
                        ++similar_count;
                    }

                    // Insert the ID into the `from_iterator` container to detect for
                    // duplicates from future calls.
                    from_iterator.insert(id);
                    ids_returned_this_batch.push_back(id);
                }

                // The number of IDs returned should equal the number of IDs reported
                // by the iterator.
                CATCH_REQUIRE(ids_returned_this_batch.size() == iterator.size());
                CATCH_REQUIRE(ids_returned_this_batch.size() == batchsize);

                iterator.next(batchsize);
            }

            // Make sure the expected number of neighbors has been obtained.
            CATCH_REQUIRE(from_iterator.size() == num_neighbors);

            // Ensure that the results returned by the iterator are "substantively similar"
            // to those returned from the full search.
            CATCH_REQUIRE(similar_count >= 0.95 * num_neighbors);
        }

        // Invoke the checker on the IDs returned from the iterator.
        CATCH_REQUIRE(checker(from_iterator));
    }
}

template <typename Index>
void check(
    Index& index,
    svs::data::ConstSimpleDataView<float> queries,
    svs::data::ConstSimpleDataView<uint32_t> groundtruth
) {
    auto checker = svs::lib::Returns<svs::lib::Const<true>>();
    check(index, queries, groundtruth, checker);
}

struct DynamicChecker {
    DynamicChecker(const std::unordered_set<size_t>& valid_ids)
        : valid_ids_{valid_ids} {}

    // Check whether `id` is valid or not.
    bool check(size_t id) {
        seen_.insert(id);
        return valid_ids_.contains(id);
    }

    template <std::integral I> bool operator()(const std::unordered_map<I, float>& ids) {
        for (const auto& itr : ids) {
            if (!check(itr.first)) {
                return false;
            }
        }
        return true;
    }

    template <std::integral I> bool operator()(const std::unordered_set<I>& ids) {
        for (auto itr : ids) {
            if (!check(itr)) {
                return false;
            }
        }
        return true;
    }

    void clear() { seen_.clear(); }

    // Valid IDs
    const std::unordered_set<size_t>& valid_ids_;
    std::unordered_set<size_t> seen_;
};

// Helper to build a static IVF index from test data
auto build_static_ivf_index() {
    namespace ivf = svs::index::ivf;

    auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto distance = svs::distance::DistanceL2();
    size_t num_threads = 2;
    size_t num_inner_threads = 2;

    // Build clustering
    auto build_params = ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = ivf::build_clustering<svs::BFloat16>(
        build_params, data, distance, threadpool, false
    );

    // Create clustered dataset
    auto centroids = clustering.centroids();
    using Idx = uint32_t;
    auto cluster = ivf::DenseClusteredDataset<decltype(centroids), Idx, decltype(data)>(
        clustering, data, threadpool, svs::lib::Allocator<std::byte>()
    );

    // Build IVF index
    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);

    return ivf::IVFIndex(
        std::move(centroids),
        std::move(cluster),
        distance,
        std::move(threadpool_for_index),
        num_inner_threads
    );
}

// Helper to build a dynamic IVF index from test data
auto build_dynamic_ivf_index() {
    namespace ivf = svs::index::ivf;
    using Eltype = float;
    using DataType = svs::data::SimpleData<Eltype, N>;
    using Idx = uint32_t;
    using Distance = svs::distance::DistanceL2;

    auto data = DataType::load(test_dataset::data_svs_file());
    auto distance = Distance();
    size_t num_threads = 2;
    size_t num_inner_threads = 2;

    // Generate IDs for all data points
    std::vector<Idx> initial_indices(data.size());
    std::iota(initial_indices.begin(), initial_indices.end(), 0);

    // Build clustering
    auto build_params = ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering =
        ivf::build_clustering<Eltype>(build_params, data, distance, threadpool, false);

    // Create dynamic clustered dataset using DenseClusteredDataset
    auto centroids = clustering.centroids();
    auto dense_clusters = ivf::DenseClusteredDataset<decltype(centroids), Idx, DataType>(
        clustering, data, threadpool, svs::lib::Allocator<std::byte>()
    );

    // Build Dynamic IVF index
    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = ivf::DynamicIVFIndex<
        decltype(centroids),
        decltype(dense_clusters),
        Distance,
        decltype(threadpool_for_index)>;

    return IndexType(
        std::move(centroids),
        std::move(dense_clusters),
        initial_indices,
        Distance(),
        std::move(threadpool_for_index),
        num_inner_threads
    );
}

} // namespace

CATCH_TEST_CASE("IVF Iterator", "[ivf][iterator]") {
    // This tests the general behavior of the iterator for correctness.
    // It is not concerned with whether the returned neighbors are accurate.
    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();

    CATCH_SECTION("Static Index") {
        auto index = build_static_ivf_index();
        check(index, queries.cview(), gt.cview());
    }

    CATCH_SECTION("Static Index - Update Query") {
        auto index = build_static_ivf_index();

        // Create an iterator with the first query
        auto query0 = std::span<const float>(queries.get_datum(0));
        auto iterator = index.make_batch_iterator(query0);

        // Get first batch
        iterator.next(10);
        CATCH_REQUIRE(iterator.size() == 10);
        CATCH_REQUIRE(iterator.batch_number() == 1);

        // Store results from first query
        auto first_query_results = std::vector<size_t>();
        for (auto n : iterator) {
            first_query_results.push_back(n.id());
        }

        // Update to second query
        auto query1 = std::span<const float>(queries.get_datum(1));
        iterator.update(query1);

        // Verify iterator is reset
        CATCH_REQUIRE(iterator.size() == 0);
        CATCH_REQUIRE(iterator.batch_number() == 0);

        // Get first batch of second query
        iterator.next(10);
        CATCH_REQUIRE(iterator.size() == 10);
        CATCH_REQUIRE(iterator.batch_number() == 1);

        // The results should be different (assuming different queries)
        auto second_query_results = std::vector<size_t>();
        for (auto n : iterator) {
            second_query_results.push_back(n.id());
        }

        // Results should be different (not necessarily completely)
        // Just check that update() actually reset the iterator state
        CATCH_REQUIRE(!iterator.done());
    }

    CATCH_SECTION("Static Index - Done Condition") {
        auto index = build_static_ivf_index();

        auto query = std::span<const float>(queries.get_datum(0));
        auto iterator = index.make_batch_iterator(query);

        // Initially not done
        CATCH_REQUIRE(!iterator.done());

        // Keep fetching until done
        size_t total_fetched = 0;
        while (!iterator.done() && total_fetched < index.size() + 100) {
            iterator.next(10);
            total_fetched += iterator.size();
        }

        // Should eventually be done
        CATCH_REQUIRE(iterator.done());
    }

    CATCH_SECTION("Dynamic Index") {
        auto index = build_dynamic_ivf_index();

        std::unordered_set<size_t> valid_ids;
        for (size_t i = 0; i < index.size(); ++i) {
            valid_ids.insert(i);
        }
        auto checker = DynamicChecker(valid_ids);
        check(index, queries.cview(), gt.cview(), checker);
    }

    CATCH_SECTION("Dynamic Index - Delete and Search") {
        auto index = build_dynamic_ivf_index();
        auto original = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        std::unordered_set<size_t> valid_ids;
        for (size_t i = 0; i < index.size(); ++i) {
            valid_ids.insert(i);
        }
        auto checker = DynamicChecker(valid_ids);

        // Delete the best candidate for each of the first few queries
        auto ids_to_delete = std::vector<size_t>();
        for (size_t i = 0; i < std::min<size_t>(5, QUERIES_TO_CHECK); ++i) {
            auto nearest_neighbor = gt.get_datum(i).front();
            auto it =
                std::find(ids_to_delete.begin(), ids_to_delete.end(), nearest_neighbor);
            if (it == ids_to_delete.end()) {
                ids_to_delete.push_back(nearest_neighbor);
                CATCH_REQUIRE(valid_ids.erase(nearest_neighbor) == 1);
            }
        }

        fmt::print("Deleting {} entries\n", ids_to_delete.size());
        index.delete_entries(ids_to_delete);
        checker.clear();
        check(index, queries.cview(), gt.cview(), checker);

        // Verify deleted IDs are not returned
        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(!checker.seen_.contains(id));
        }
    }

    CATCH_SECTION("Iterator Contents and Span") {
        auto index = build_static_ivf_index();

        auto query = std::span<const float>(queries.get_datum(0));
        auto iterator = index.make_batch_iterator(query);

        // Get a batch
        iterator.next(20);
        CATCH_REQUIRE(iterator.size() == 20);

        // Test contents() returns a valid span
        auto contents = iterator.contents();
        CATCH_REQUIRE(contents.size() == 20);

        // Verify contents match iteration
        size_t idx = 0;
        for (auto n : iterator) {
            CATCH_REQUIRE(n.id() == contents[idx].id());
            CATCH_REQUIRE(n.distance() == contents[idx].distance());
            ++idx;
        }
    }

    CATCH_SECTION("Restart Search") {
        auto index = build_static_ivf_index();

        auto query = std::span<const float>(queries.get_datum(0));
        auto iterator = index.make_batch_iterator(query);

        // Get first batch
        iterator.next(10);
        CATCH_REQUIRE(iterator.batch_number() == 1);

        auto first_results = std::vector<size_t>();
        for (auto n : iterator) {
            first_results.push_back(n.id());
        }

        // Force restart
        iterator.restart_next_search();

        // Get another batch
        iterator.next(10);
        CATCH_REQUIRE(iterator.batch_number() == 2);

        // After restart, the new batch should not duplicate any IDs from first batch
        for (auto n : iterator) {
            CATCH_REQUIRE(
                std::find(first_results.begin(), first_results.end(), n.id()) ==
                first_results.end()
            );
        }
    }
}
