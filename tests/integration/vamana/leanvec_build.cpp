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

// svs
#include "svs/core/recall.h"
#include "svs/extensions/vamana/leanvec.h"
#include "svs/lib/timing.h"
#include "svs/orchestrators/vamana.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/datasets/leanvec.h"

// fmt
#include "fmt/core.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// stl
#include <array>
#include <filesystem>
#include <string>
#include <utility>

namespace leanvec = svs::leanvec;
namespace {

template <
    typename T1,
    typename T2,
    size_t LeanVecDims = svs::Dynamic,
    size_t D = svs::Dynamic,
    typename Distance>
svs::Vamana build_index(
    const svs::index::vamana::VamanaBuildParameters parameters,
    const std::filesystem::path& data_path,
    const size_t num_threads,
    const Distance& dist_type,
    bool is_pca = true
) {
    auto tic = svs::lib::now();
    auto loader = svs::lib::Lazy([=]() {
        auto data = svs::VectorDataLoader<float, D>(data_path).load();
        if (!is_pca) {
            auto data_matrix = test_dataset::leanvec_data_matrix<LeanVecDims>();
            auto query_matrix = test_dataset::leanvec_query_matrix<LeanVecDims>();
            return leanvec::LeanDataset<T1, T2, LeanVecDims, D>::reduce(
                data,
                std::optional(
                    leanvec::LeanVecMatrices<LeanVecDims>(data_matrix, query_matrix)
                )
            );
        }
        return leanvec::LeanDataset<T1, T2, LeanVecDims, D>::reduce(data);
    });

    auto index = svs::Vamana::build<float>(parameters, loader, dist_type, num_threads);

    fmt::print("Indexing time: {}s\n", svs::lib::time_difference(tic));

    // Make sure the number of threads was propagated correctly.
    CATCH_REQUIRE(index.get_num_threads() == num_threads);
    return index;
}

template <typename T, typename Distance>
void test_build(const Distance& distance, bool is_pca = true) {
    // How far these results may deviate from previously generated results.
    const double epsilon = 0.005;
    const auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    CATCH_REQUIRE(svs_test::prepare_temp_directory());
    size_t num_threads = 2;
    auto kind = svsbenchmark::LeanVec(
        svsbenchmark::leanvec_kind_v<T>, svsbenchmark::leanvec_kind_v<T>, 64
    );

    if (!is_pca) {
        kind.data_matrix_ = test_dataset::leanvec_data_matrix_file();
        kind.query_matrix_ = test_dataset::leanvec_query_matrix_file();
    }

    auto expected_result =
        test_dataset::vamana::expected_build_results(svs::distance_type_v<Distance>, kind);
    svs::Vamana index = build_index<T, T, 64>(
        expected_result.build_parameters_.value(),
        test_dataset::data_svs_file(),
        num_threads,
        distance,
        is_pca
    );

    auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);
    for (const auto& expected : expected_result.config_and_recall_) {
        auto these_queries = test_dataset::get_test_set(queries, expected.num_queries_);
        auto these_groundtruth =
            test_dataset::get_test_set(groundtruth, expected.num_queries_);
        index.set_search_parameters(expected.search_parameters_);
        auto results = index.search(these_queries, expected.num_neighbors_);
        double recall = svs::k_recall_at_n(
            these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );

        fmt::print(
            "Window Size: {}, Expected Recall: {}, Actual Recall: {}\n",
            index.get_search_window_size(),
            expected.recall_,
            recall
        );
        CATCH_REQUIRE(recall > expected.recall_ - epsilon);
        CATCH_REQUIRE(recall < expected.recall_ + epsilon);
    }
}

} // namespace

CATCH_TEST_CASE("LeanVec Vamana Build", "[integration][build][leanvec][vamana]") {
    test_build<float>(svs::DistanceL2());
    test_build<float>(svs::DistanceIP());
    test_build<leanvec::UsingLVQ<8>>(svs::DistanceL2());
    test_build<leanvec::UsingLVQ<8>>(svs::DistanceIP());

    // LeanVec OOD
    test_build<leanvec::UsingLVQ<8>>(svs::DistanceL2(), false);
    test_build<leanvec::UsingLVQ<8>>(svs::DistanceIP(), false);
}
