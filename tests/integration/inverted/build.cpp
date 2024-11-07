/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// svs
#include "svs/lib/timing.h"
#include "svs/orchestrators/inverted.h"

// svs-benchmark
#include "svs-benchmark/datasets.h"

// tests
#include "tests/utils/inverted_reference.h"
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <filesystem>

namespace {

template <typename E, typename Distance, typename ClusterStrategy, size_t D = svs::Dynamic>
svs::Inverted build_index(
    const svs::index::inverted::InvertedBuildParameters& build_parameters,
    const std::filesystem::path& data_path,
    size_t num_threads,
    Distance distance,
    ClusterStrategy strategy
) {
    auto tic = svs::lib::now();
    svs::Inverted index = svs::Inverted::build<E>(
        build_parameters,
        svs::data::SimpleData<E, D>::load(data_path),
        distance,
        num_threads,
        strategy
    );
    fmt::print("Indexing time: {}s\n", svs::lib::time_difference(tic));
    CATCH_REQUIRE(index.get_num_threads() == num_threads);
    return index;
}

template <typename Distance, typename Strategy, typename Queries>
void run_test(const Queries& queries) {
    auto distance = Distance();
    auto strategy = Strategy();

    // Distance between the obtained results and reference ressults.
    const double epsilon = 0.005;
    size_t num_threads = 2;
    constexpr svs::DistanceType distance_type = svs::distance_type_v<decltype(distance)>;
    auto expected_results = test_dataset::inverted::expected_build_results(
        distance_type, svsbenchmark::Uncompressed(svs::DataType::float32)
    );

    svs::Inverted index = build_index<float>(
        expected_results.build_parameters_.value(),
        test_dataset::data_svs_file(),
        num_threads,
        distance,
        strategy
    );

    auto groundtruth = test_dataset::load_groundtruth(distance_type);
    for (const auto& expected : expected_results.config_and_recall_) {
        const auto& sp = expected.search_parameters_;
        CATCH_REQUIRE(index.get_search_parameters() != sp);
        auto these_queries = test_dataset::get_test_set(queries, expected.num_queries_);
        auto these_groundtruth =
            test_dataset::get_test_set(groundtruth, expected.num_queries_);
        index.set_search_parameters(sp);
        CATCH_REQUIRE(index.get_search_parameters() == sp);
        for (size_t num_threads : {1, 2}) {
            index.set_num_threads(num_threads);
            CATCH_REQUIRE(index.get_num_threads() == num_threads);

            auto results = index.search(these_queries, expected.num_neighbors_);
            double recall = svs::k_recall_at_n(
                these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
            );

            fmt::print(
                "Expected Recall: {}, Actual Recall: {}\n", expected.recall_, recall
            );
            CATCH_REQUIRE(recall > expected.recall_ - epsilon);
            CATCH_REQUIRE(recall < expected.recall_ + epsilon);
        }
    }
}

} // namespace

CATCH_TEST_CASE("Test Inverted Building", "[integration][build][inverted]") {
    auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    run_test<svs::DistanceL2, svs::index::inverted::SparseStrategy>(queries);
    run_test<svs::DistanceL2, svs::index::inverted::DenseStrategy>(queries);

    run_test<svs::DistanceIP, svs::index::inverted::SparseStrategy>(queries);
    run_test<svs::DistanceIP, svs::index::inverted::SparseStrategy>(queries);
}
