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
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>

// svs
#include "svs/core/data/simple.h"
#include "svs/core/recall.h"
#include "svs/lib/timing.h"
#include "svs/orchestrators/vamana.h"

// fmt
#include "fmt/core.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

namespace {

template <typename E, size_t D = svs::Dynamic>
svs::Vamana build_index(
    const std::string& vecs_filename,
    const size_t build_search_window_size,
    const size_t max_degree,
    const size_t max_candidate_pool_size,
    float alpha,
    const size_t n_threads,
    const svs::DistanceType dist_type
) {
    svs::index::vamana::VamanaBuildParameters parameters{
        alpha,
        max_degree,
        build_search_window_size,
        max_candidate_pool_size,
        max_degree,
        true};

    auto tic = svs::lib::now();
    svs::Vamana index = svs::Vamana::build<E>(
        parameters, svs::VectorDataLoader<E, D>(vecs_filename), dist_type, n_threads
    );

    auto diff = svs::lib::time_difference(tic);
    std::cout << "Indexing time: " << diff << "s\n";

    // Make sure the number of threads was propagated correctly.
    CATCH_REQUIRE(index.get_num_threads() == n_threads);
    return index;
}
} // namespace

CATCH_TEST_CASE("Test Building", "[integration][build]") {
    auto distances = std::to_array<svs::DistanceType>({svs::L2, svs::MIP, svs::Cosine});

    // Boot-strapped accuracies from previous runs.
    const std::map<size_t, double> result_map_l2{
        {2, 0.217},
        {3, 0.2657},
        {4, 0.306},
        {5, 0.332},
        {10, 0.4379},
        {20, 0.54005},
        {50, 0.66526},
        {100, 0.74538}};

    const std::map<size_t, double> result_map_mip{
        {2, 0.09},
        {3, 0.1156667},
        {4, 0.143},
        {5, 0.1642},
        {10, 0.242},
        {20, 0.35485},
        {50, 0.53504},
        {100, 0.68658}};

    const std::map<size_t, double> result_map_cosine{
        {2, 0.0725},
        {3, 0.0976666},
        {4, 0.1165},
        {5, 0.1392},
        {10, 0.2136},
        {20, 0.32545},
        {50, 0.51474},
        {100, 0.67853}};

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

    const auto alpha_map = std::map<svs::DistanceType, float>{
        {svs::L2, 1.2f}, {svs::MIP, 1 / 1.2f}, {svs::Cosine, 1 / 1.2f}};

    for (auto distance_type : distances) {
        CATCH_REQUIRE(svs_test::prepare_temp_directory());
        size_t num_threads = 2;
        svs::Vamana index = build_index<float>(
            test_dataset::data_svs_file(),
            40,
            30,
            30,
            alpha_map.at(distance_type),
            num_threads,
            distance_type
        );

        const auto& groundtruth = groundtruth_map.at(distance_type);
        const auto& expected_results = result_map.at(distance_type);
        const double epsilon = 0.01;

        const auto queries = svs::load_data<float>(test_dataset::query_file());
        for (auto [windowsize, expected_recall] : expected_results) {
            index.set_search_window_size(windowsize);
            auto results = index.search(queries, windowsize);
            double recall = svs::k_recall_at_n(groundtruth, results);

            fmt::print(
                "Window Size: {}, Expected Recall: {}, Actual Recall: {}\n",
                windowsize,
                expected_recall,
                recall
            );
            CATCH_REQUIRE(recall >= expected_recall - epsilon);
            CATCH_REQUIRE(recall <= expected_recall + epsilon);
        }
    }
}
