/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// stl
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

// svs
#include "svs/core/recall.h"
#include "svs/lib/saveload.h"
#include "svs/orchestrators/vamana.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

// fmt
#include "fmt/core.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

namespace {

void verify_reconstruction(svs::Vamana& index, const svs::data::SimpleData<float>& data) {
    // Construct a scrambled version of the ids to retrieve.
    auto ids = std::vector<uint64_t>(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        ids.at(i) = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);

    auto dst = svs::data::SimpleData<float>(data.size(), data.dimensions());
    index.reconstruct_at(dst.view(), ids);
    for (size_t i = 0; i < data.size(); ++i) {
        auto id = ids.at(i);
        auto d = data.get_datum(id);
        auto o = dst.get_datum(i);
        CATCH_REQUIRE(d.size() == o.size());
        CATCH_REQUIRE(std::equal(d.begin(), d.end(), o.begin()));
    }

    // Check error handling.
    // (A) Wrong dimensionality.
    // (A1) Too big.
    dst = svs::data::SimpleData<float>(data.size(), 2 * data.dimensions());
    CATCH_REQUIRE_THROWS_MATCHES(
        index.reconstruct_at(dst.view(), ids),
        svs::ANNException,
        svs_test::ExceptionMatcher(
            Catch::Matchers::ContainsSubstring("Destination has dimensions")
        )
    );

    // (A2) Too small.
    dst = svs::data::SimpleData<float>(data.size(), data.dimensions() - 1);
    CATCH_REQUIRE_THROWS_MATCHES(
        index.reconstruct_at(dst.view(), ids),
        svs::ANNException,
        svs_test::ExceptionMatcher(
            Catch::Matchers::ContainsSubstring("Destination has dimensions")
        )
    );

    // (B) Size mismatch.
    // (B1) Too big.
    dst = svs::data::SimpleData<float>(data.size() + 1, data.dimensions());
    CATCH_REQUIRE_THROWS_MATCHES(
        index.reconstruct_at(dst.view(), ids),
        svs::ANNException,
        svs_test::ExceptionMatcher(Catch::Matchers::ContainsSubstring("IDs span has size"))
    );

    // (B2) Too small.
    dst = svs::data::SimpleData<float>(data.size() - 1, data.dimensions());
    CATCH_REQUIRE_THROWS_MATCHES(
        index.reconstruct_at(dst.view(), ids),
        svs::ANNException,
        svs_test::ExceptionMatcher(Catch::Matchers::ContainsSubstring("IDs span has size"))
    );

    // (C) IDs out of range.
    // Make sure that the contents of `dst` are unmodified until the bounds of all indices
    // are checked.
    dst = svs::data::SimpleData<float>(2, data.dimensions());
    for (size_t i = 0; i < dst.size(); ++i) {
        auto a = dst.get_datum(i);
        std::fill(a.begin(), a.end(), 0);
    }
    ids = {0, data.size()};
    CATCH_REQUIRE_THROWS_MATCHES(
        index.reconstruct_at(dst.view(), ids),
        svs::ANNException,
        svs_test::ExceptionMatcher(Catch::Matchers::ContainsSubstring("ID 1 with value"))
    );
    for (size_t i = 0; i < dst.size(); ++i) {
        auto a = dst.get_datum(i);
        CATCH_REQUIRE(std::all_of(a.begin(), a.end(), [](float v) { return v == 0; }));
    }
}

void run_tests(
    svs::Vamana& index,
    const svs::data::SimpleData<float>& queries_all,
    const svs::data::SimpleData<uint32_t>& groundtruth_all,
    const std::vector<svsbenchmark::vamana::ConfigAndResult>& expected_results,
    bool test_calibration = false
) {
    // If we make a change that somehow improves accuracy, we'll want to know.
    // Use `epsilon` to add to the expected results to set an upper bound on the
    // achieved accuracy.
    double epsilon = 0.0005;

    CATCH_REQUIRE(index.can_change_threads());
    CATCH_REQUIRE(index.get_num_threads() == 2);
    index.set_num_threads(1);
    CATCH_REQUIRE(index.get_num_threads() == 1);

    index.set_search_window_size(10);
    CATCH_REQUIRE(index.get_search_window_size() == 10);

    // Make sure we get errors if we try to feed in an unsupported query type.
    auto queries_f16_all =
        svs::data::SimpleData<svs::Float16>{queries_all.size(), queries_all.dimensions()};
    svs::data::copy(queries_all, queries_f16_all);

    {
        auto mock_queries_u8 = svs::data::SimpleData<uint8_t>(10, queries_all.dimensions());
        CATCH_REQUIRE_THROWS_AS(index.search(mock_queries_u8, 10), svs::ANNException);
    }

    // Ensure we have at least one entry in the expected results.
    CATCH_REQUIRE(!expected_results.empty());

    const auto queries_in_test_set = expected_results.at(0).num_queries_;

    auto queries = test_dataset::get_test_set(queries_all, queries_in_test_set);
    auto queries_f16 = test_dataset::get_test_set(queries_f16_all, queries_in_test_set);
    auto groundtruth = test_dataset::get_test_set(groundtruth_all, queries_in_test_set);

    // End to end queries.
    bool first = false;
    for (const auto& expected : expected_results) {
        // Update the query set if needed.
        auto num_queries = expected.num_queries_;
        if (num_queries != queries.size()) {
            queries = test_dataset::get_test_set(queries_all, num_queries);
            queries_f16 = test_dataset::get_test_set(queries_f16_all, num_queries);
            groundtruth = test_dataset::get_test_set(groundtruth_all, num_queries);
        }

        // Configure the index with the current parameters.
        // Ensure that the result sticks.
        index.set_search_parameters(expected.search_parameters_);
        CATCH_REQUIRE(index.get_search_parameters() == expected.search_parameters_);

        for (auto num_threads : {1, 2}) {
            index.set_num_threads(num_threads);
            // Float32
            auto results = index.search(queries, expected.num_neighbors_);
            auto recall = svs::k_recall_at_n(
                groundtruth, results, expected.num_neighbors_, expected.recall_k_
            );
            CATCH_REQUIRE(recall > expected.recall_ - epsilon);
            CATCH_REQUIRE(recall < expected.recall_ + epsilon);

            // Test Float16 results, but only on the first iteration.
            // Otherwise, skip it to keep run times down.
            if (first) {
                results = index.search(queries_f16, expected.num_neighbors_);
                recall = svs::k_recall_at_n(
                    groundtruth, results, expected.num_neighbors_, expected.recall_k_
                );
                CATCH_REQUIRE(recall > expected.recall_ - epsilon);
                CATCH_REQUIRE(recall < expected.recall_ + epsilon);
                first = false;
            }
        }
    }

    // Make sure calibration works.
    if (!test_calibration) {
        return;
    }

    index.set_search_parameters(svs::index::vamana::VamanaSearchParameters());
    // Select the first target recall as we know it should be achievable.
    if (queries_in_test_set != queries.size()) {
        queries = test_dataset::get_test_set(queries_all, queries_in_test_set);
        groundtruth = test_dataset::get_test_set(groundtruth_all, queries_in_test_set);
    }

    auto first_result = expected_results.at(0);
    auto c = svs::index::vamana::CalibrationParameters();
    c.search_window_size_upper_ = 30;
    c.search_window_capacity_upper_ = 30;
    c.train_prefetchers_ = false;

    index.experimental_calibrate(
        queries, groundtruth, first_result.num_neighbors_, first_result.recall_, c
    );
    auto recall = svs::k_recall_at_n(
        groundtruth,
        index.search(queries, first_result.num_neighbors_),
        first_result.num_neighbors_,
        first_result.recall_k_
    );
    CATCH_REQUIRE(recall >= first_result.recall_);
}
} // namespace

CATCH_TEST_CASE("Uncompressed Vamana Search", "[integration][search][vamana]") {
    auto distances = std::to_array<svs::DistanceType>({svs::L2, svs::MIP, svs::Cosine});
    const auto queries = test_dataset::queries();
    auto temp_dir = svs_test::temp_directory();

    auto original_data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

    bool first = true;
    for (auto distance_type : distances) {
        auto groundtruth = test_dataset::load_groundtruth(distance_type);
        auto expected_results = test_dataset::vamana::expected_search_results(
            distance_type, svsbenchmark::Uncompressed(svs::DataType::float32)
        );

        auto index = svs::Vamana::assemble<svs::lib::Types<float, svs::Float16>>(
            test_dataset::vamana_config_file(),
            svs::GraphLoader(test_dataset::graph_file()),
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()),
            distance_type,
            2
        );

        CATCH_REQUIRE(index.size() == test_dataset::VECTORS_IN_DATA_SET);
        CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);
        CATCH_REQUIRE(
            index.query_types() ==
            std::vector<svs::DataType>{svs::DataType::float32, svs::DataType::float16}
        );

        if (first) {
            verify_reconstruction(index, original_data);
            first = false;
        }

        run_tests(index, queries, groundtruth, expected_results.config_and_recall_, true);

        // Save and reload.
        svs_test::prepare_temp_directory();

        // Set variables to ensure they are saved and reloaded properly.
        index.set_search_window_size(123);
        index.set_alpha(1.2);
        index.set_construction_window_size(456);
        index.set_max_candidates(1001);

        auto config_dir = temp_dir / "config";
        auto graph_dir = temp_dir / "graph";
        auto data_dir = temp_dir / "data";

        index.save(config_dir, graph_dir, data_dir);
        index = svs::Vamana::assemble<svs::lib::Types<float, svs::Float16>>(
            config_dir,
            svs::GraphLoader(graph_dir),
            svs::VectorDataLoader<float>(data_dir),
            distance_type,
            1
        );
        // Data Properties
        CATCH_REQUIRE(index.size() == test_dataset::VECTORS_IN_DATA_SET);
        CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);
        // Index Properties
        CATCH_REQUIRE(index.get_search_window_size() == 123);
        CATCH_REQUIRE(index.get_alpha() == 1.2f);
        CATCH_REQUIRE(index.get_construction_window_size() == 456);
        CATCH_REQUIRE(index.get_max_candidates() == 1001);

        index.set_num_threads(2);
        run_tests(index, queries, groundtruth, expected_results.config_and_recall_);
    }
}
