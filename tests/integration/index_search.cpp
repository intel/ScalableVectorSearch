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

// catch2
#include "catch2/catch_test_macros.hpp"

// fmt
#include "fmt/core.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

namespace {
const bool PRINT_RESULTS = false;
void run_tests(
    svs::Vamana& index,
    const svs::data::SimpleData<float>& queries,
    const svs::data::SimpleData<uint32_t>& groundtruth,
    const std::map<size_t, double>& expected_results
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
    auto mock_queries_f16 =
        svs::data::SimpleData<svs::Float16>{queries.size(), queries.dimensions()};
    CATCH_REQUIRE_THROWS_AS(index.search(mock_queries_f16, 10), svs::ANNException);

    // End to end queries.
    for (const auto& [search_window_size, expected_accuracy] : expected_results) {
        index.set_search_window_size(search_window_size);
        CATCH_REQUIRE(index.get_search_window_size() == search_window_size);

        // Perform several runs with the visited set disabled.
        index.disable_visited_set();
        CATCH_REQUIRE(index.visited_set_enabled() == false);
        for (auto num_threads : {1, 2}) {
            index.set_num_threads(num_threads);
            const auto results = index.search(queries, search_window_size);

            // Compute and compare accuracy.
            double recall = svs::k_recall_at_n(groundtruth, results);
            if constexpr (PRINT_RESULTS) {
                fmt::print(
                    "Window size {}, Expected {}, Got {}\n",
                    search_window_size,
                    expected_accuracy,
                    recall
                );
            }
            CATCH_REQUIRE(recall >= expected_accuracy);
            CATCH_REQUIRE(recall <= expected_accuracy + epsilon);
        }

        // Perform one run with the visited set enabled.
        {
            CATCH_REQUIRE(index.visited_set_enabled() == false);
            index.enable_visited_set();
            CATCH_REQUIRE(index.visited_set_enabled() == true);
            const auto results = index.search(queries, search_window_size);
            double recall = svs::k_recall_at_n(groundtruth, results);
            CATCH_REQUIRE(recall >= expected_accuracy - epsilon);
            CATCH_REQUIRE(recall <= expected_accuracy + epsilon);
        }
        index.disable_visited_set();

        // // Make sure changing the number of threads actually does something.
        // // Only run in non-debug mode since this can take a little while.
        // //
        // // Also - limit the maximum size to avoid consuming too much run time.
        // if (search_window_size < 40) {
        //     auto timer = [&, search_window_size = search_window_size] {
        //         return svs_test::timed(2, true, [&] {
        //             index.search(queries, search_window_size);
        //         });
        //     };

        //     index.set_num_threads(1);
        //     double base_time = timer();

        //     index.set_num_threads(2);
        //     double new_time = timer();

        //     if constexpr (PRINT_RESULTS) {
        //         fmt::print(
        //             "Window Size: {}, Base Time: {}, New Time: {}\n",
        //             search_window_size,
        //             base_time,
        //             new_time
        //         );
        //     }

        //     // Don't expect perfect speedup, but speedup should generally be pretty
        //     // good. The trickiest case the the one with the smallest window size. In
        //     // this case, the overhead of threading is larger so we don't get as close
        //     // to a perfect 2x speedup.
        //     CATCH_REQUIRE(1.3 * new_time < base_time);
        // }
    }
}
} // namespace

CATCH_TEST_CASE("Testing Search", "[integration][search]") {
    auto distances = std::to_array<svs::DistanceType>({svs::L2, svs::MIP, svs::Cosine});

    // Construct a map from window size to expected accuracy:
    // To boot-strap this process, I saved the results of each query window size
    // to file and used an external known-good program to compute the accuracy.
    //
    // Going forward (assuming the code is in a working place), one can use the
    // code itself to generated the expected accuracies using simple prints and then
    // hard-code those results into the tests.
    const std::map<size_t, double> result_map_l2{
        {2, 0.4595}, {3, 0.537333}, {4, 0.60025}, {5, 0.643}, {10, 0.7585}, {20, 0.86}};
    // {50, 0.94662},
    // {100, 0.97724}};

    const std::map<size_t, double> result_map_mip{
        {2, 0.1405}, {3, 0.167}, {4, 0.18575}, {5, 0.2064}, {10, 0.3076}, {20, 0.4242}};
    // {50, 0.6148},
    // {100, 0.76227}};

    const std::map<size_t, double> result_map_cosine{
        {2, 0.207}, {3, 0.255}, {4, 0.289}, {5, 0.3196}, {10, 0.4299}, {20, 0.5551}};
    // {50, 0.72514},
    // {100, 0.84135}};

    const auto result_map = std::map<svs::DistanceType, std::map<size_t, double>>{
        {svs::L2, result_map_l2},
        {svs::MIP, result_map_mip},
        {svs::Cosine, result_map_cosine}};

    // Note: can't construct from an initializer list because initializer lists want to
    // copy.
    auto groundtruth_map =
        std::unordered_map<svs::DistanceType, svs::data::SimpleData<uint32_t>>{};
    groundtruth_map.emplace(svs::L2, test_dataset::groundtruth_euclidean());
    groundtruth_map.emplace(svs::MIP, test_dataset::groundtruth_mip());
    groundtruth_map.emplace(svs::Cosine, test_dataset::groundtruth_cosine());

    const auto queries = test_dataset::queries();
    auto temp_dir = svs_test::temp_directory();

    for (auto distance_type : distances) {
        const auto& groundtruth = groundtruth_map.at(distance_type);
        const auto& results = result_map.at(distance_type);

        auto index = svs::Vamana::assemble<float>(
            test_dataset::vamana_config_file(),
            svs::GraphLoader(test_dataset::graph_file()),
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()),
            distance_type,
            2
        );
        CATCH_REQUIRE(index.size() == test_dataset::VECTORS_IN_DATA_SET);
        CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);
        run_tests(index, queries, groundtruth, results);

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
        index = svs::Vamana::assemble<float>(
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
        run_tests(index, queries, groundtruth, results);
    }
}
