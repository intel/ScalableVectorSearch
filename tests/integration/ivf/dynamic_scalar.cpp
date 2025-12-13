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
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/recall.h"
#include "svs/extensions/ivf/scalar.h"
#include "svs/index/ivf/clustering.h"
#include "svs/orchestrators/dynamic_ivf.h"
#include "svs/quantization/scalar/scalar.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"

// fmt
#include "fmt/core.h"

// stl
#include <numeric>
#include <random>

namespace sc = svs::quantization::scalar;

namespace {

constexpr size_t NUM_NEIGHBORS = 10;
constexpr size_t NUM_CLUSTERS = 10;
constexpr size_t EXTENT = 128;

///
/// Test Dynamic IVF with Scalar Quantization
///
template <typename ElementType, typename Distance>
void test_dynamic_ivf_scalar(const Distance& distance) {
    size_t num_threads = 2;
    size_t intra_query_threads = 2;

    // Load test dataset
    auto data = svs::data::SimpleData<float, EXTENT>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();

    // Build clustering on UNCOMPRESSED data
    auto build_params = svs::index::ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<float>(
        build_params, data, distance, threadpool, false
    );

    // Compress the data with Scalar Quantization
    auto compressed_data = sc::SQDataset<ElementType, EXTENT>::compress(data);

    // Generate external IDs for the data
    std::vector<size_t> ids(data.size());
    std::iota(ids.begin(), ids.end(), 0);

    auto index = svs::DynamicIVF::assemble_from_clustering<float>(
        std::move(clustering),
        compressed_data,
        ids,
        distance,
        svs::threads::as_threadpool(num_threads),
        intra_query_threads
    );

    // Search
    auto search_params = svs::index::ivf::IVFSearchParameters(
        NUM_CLUSTERS, // n_probes
        NUM_NEIGHBORS // k_reorder
    );

    auto results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);
    index.search(
        results.view(),
        svs::data::ConstSimpleDataView<float>{
            queries.data(), queries.size(), queries.dimensions()},
        search_params
    );

    // Check recall
    auto recall = svs::k_recall_at_n(gt, results, NUM_NEIGHBORS, NUM_NEIGHBORS);

    // Set expected recall thresholds based on quantization level
    CATCH_REQUIRE(recall > 0.9);
}

///
/// Test Dynamic IVF with Scalar Quantization - Add/Delete/Compact stress test
///
template <typename ElementType, typename Distance>
void test_dynamic_ivf_scalar_stress(const Distance& distance) {
    size_t num_threads = 2;
    size_t intra_query_threads = 2;

    // Load test dataset
    auto data = svs::data::SimpleData<float, EXTENT>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();

    // Start with half the data
    size_t initial_size = data.size() / 2;
    auto initial_data = svs::data::SimpleData<float, EXTENT>(initial_size, EXTENT);
    for (size_t i = 0; i < initial_size; ++i) {
        initial_data.set_datum(i, data.get_datum(i));
    }

    // Build clustering on initial data
    auto build_params = svs::index::ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<float>(
        build_params, initial_data, distance, threadpool, false
    );

    // Compress with Scalar Quantization
    auto compressed_data = sc::SQDataset<ElementType, EXTENT>::compress(initial_data);

    // Generate external IDs
    std::vector<size_t> ids(initial_size);
    std::iota(ids.begin(), ids.end(), 0);

    auto index = svs::DynamicIVF::assemble_from_clustering<float>(
        std::move(clustering),
        compressed_data,
        ids,
        distance,
        svs::threads::as_threadpool(num_threads),
        intra_query_threads
    );

    auto search_params = svs::index::ivf::IVFSearchParameters(NUM_CLUSTERS, NUM_NEIGHBORS);
    auto results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);

    // Perform add/delete/compact cycles
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> idx_dist(0, initial_size - 1);

    for (size_t cycle = 0; cycle < 3; ++cycle) {
        // Delete some entries
        std::vector<size_t> to_delete;
        for (size_t i = 0; i < 20 && i < ids.size(); ++i) {
            size_t idx = idx_dist(rng) % ids.size();
            to_delete.push_back(ids[idx]);
        }
        if (!to_delete.empty()) {
            index.delete_points(to_delete);
        }

        // Add new entries (uncompressed - index will compress them)
        size_t num_to_add = 30;
        auto new_data = svs::data::SimpleData<float, EXTENT>(num_to_add, EXTENT);
        std::vector<size_t> new_ids;
        size_t new_base_id = 100000 + cycle * 1000;

        for (size_t i = 0; i < num_to_add; ++i) {
            new_ids.push_back(new_base_id + i);
            new_data.set_datum(i, data.get_datum(i % data.size()));
        }

        // Pass uncompressed data as ConstSimpleDataView - index will compress
        auto new_data_view = svs::data::ConstSimpleDataView<float>{
            new_data.data(), new_data.size(), new_data.dimensions()
        };
        index.add_points(new_data_view, new_ids, false);

        // Search after modifications
        index.search(
            results.view(),
            svs::data::ConstSimpleDataView<float>{
                queries.data(), queries.size(), queries.dimensions()},
            search_params
        );

        // Verify no deleted IDs appear in results
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t k = 0; k < NUM_NEIGHBORS; ++k) {
                auto result_id = results.index(q, k);
                for (auto deleted_id : to_delete) {
                    CATCH_REQUIRE(result_id != deleted_id);
                }
            }
        }

        // Compact every cycle
        index.compact(50);

        // Search after compaction
        index.search(
            results.view(),
            svs::data::ConstSimpleDataView<float>{
                queries.data(), queries.size(), queries.dimensions()},
            search_params
        );

        // Verify all results are valid
        for (size_t q = 0; q < queries.size(); ++q) {
            CATCH_REQUIRE(results.index(q, 0) != std::numeric_limits<size_t>::max());
        }
    }
}

} // anonymous namespace

CATCH_TEST_CASE("Dynamic IVF with Scalar Quantization", "[integration][dynamic_ivf][scalar]") {
    auto distance = svs::DistanceL2();

    CATCH_SECTION("int8 quantization") {
        test_dynamic_ivf_scalar<int8_t>(distance);
    }

    CATCH_SECTION("int8 stress test") {
        test_dynamic_ivf_scalar_stress<int8_t>(distance);
    }
}
