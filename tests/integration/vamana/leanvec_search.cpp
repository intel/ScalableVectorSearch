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
#include "svs/lib/saveload.h"
#include "svs/orchestrators/vamana.h"

// tests
#include "tests/utils/lvq_reconstruction.h" // To check LVQ reconstruction.
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/datasets/leanvec.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace leanvec = svs::leanvec;
namespace {

template <size_t N>
void check_reconstruction(
    svs::Vamana& index,
    svs::data::ConstSimpleDataView<float, N> original,
    size_t lvq_bits // zero for non-lvq
) {
    auto ids = svs_test::permute_indices(original.size());
    auto dst = svs::data::SimpleData<float>(original.size(), original.dimensions());
    index.reconstruct_at(dst.view(), ids);
    auto shuffled = svs::data::SimpleData<float>(original.size(), original.dimensions());
    for (size_t i = 0; i < original.size(); ++i) {
        shuffled.set_datum(i, original.get_datum(ids.at(i)));
    }

    // Uncompressed residual
    if (lvq_bits == 0) {
        for (size_t i = 0, imax = original.size(); i < imax; ++i) {
            auto r = dst.get_datum(i);
            auto o = shuffled.get_datum(i);
            CATCH_REQUIRE(r.size() == o.size());
            for (size_t j = 0, jmax = r.size(); j < jmax; ++j) {
                CATCH_REQUIRE(std::abs(r[j] - o[j]) <= 1.0e-4);
            }
        }
    } else {
        svs_test::check_lvq_reconstruction(shuffled.cview(), dst.cview(), lvq_bits, 0);
    }
}

void run_search(
    svs::Vamana& index,
    const svs::data::SimpleData<float>& queries_all,
    const svs::data::SimpleData<uint32_t>& groundtruth_all,
    const std::vector<svsbenchmark::vamana::ConfigAndResult>& expected_results,
    bool test_calibration
) {
    double epsilon = 0.0005;
    for (const auto& expected : expected_results) {
        auto num_queries = expected.num_queries_;
        auto queries = test_dataset::get_test_set(queries_all, num_queries);
        auto groundtruth = test_dataset::get_test_set(groundtruth_all, num_queries);

        index.set_search_parameters(expected.search_parameters_);
        CATCH_REQUIRE(index.get_search_parameters() == expected.search_parameters_);

        for (auto num_threads : {1, 2}) {
            index.set_num_threads(num_threads);
            auto results = index.search(queries, expected.num_neighbors_);
            auto recall = svs::k_recall_at_n(
                groundtruth, results, expected.num_neighbors_, expected.recall_k_
            );
            CATCH_REQUIRE(recall > expected.recall_ - epsilon);
            CATCH_REQUIRE(recall < expected.recall_ + epsilon);
        }
    }

    // Return early if calibration is not being tested.
    if (test_calibration == false) {
        return;
    }

    index.set_search_parameters(svs::index::vamana::VamanaSearchParameters());
    auto first_result = expected_results.at(0);
    size_t num_queries = first_result.num_queries_;
    auto queries = test_dataset::get_test_set(queries_all, num_queries);
    auto groundtruth = test_dataset::get_test_set(groundtruth_all, num_queries);

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

template <leanvec::IsLeanDataset Data, typename Distance>
void test_search(
    Data data,
    const svs::data::SimpleData<float, Data::extent>& original,
    const Distance& distance,
    const svs::data::SimpleData<float>& queries,
    const svs::data::SimpleData<uint32_t>& groundtruth,
    bool is_pca = true,
    bool try_calibration = false
) {
    size_t num_threads = 2;

    auto secondary_kind = svsbenchmark::leanvec_kind_v<typename Data::secondary_data_type>;
    auto kind = svsbenchmark::LeanVec(
        svsbenchmark::leanvec_kind_v<typename Data::primary_data_type>,
        secondary_kind,
        data.inner_dimensions()
    );
    if (!is_pca) {
        kind.data_matrix_ = test_dataset::leanvec_data_matrix_file();
        kind.query_matrix_ = test_dataset::leanvec_query_matrix_file();
    }

    // Find the expected results for this dataset.
    auto expected_results =
        test_dataset::vamana::expected_search_results(svs::distance_type_v<Distance>, kind);

    auto index = svs::Vamana::assemble<float>(
        test_dataset::vamana_config_file(),
        svs::GraphLoader(test_dataset::graph_file()),
        std::move(data),
        distance,
        num_threads
    );
    CATCH_REQUIRE(index.get_num_threads() == num_threads);

    run_search(
        index, queries, groundtruth, expected_results.config_and_recall_, try_calibration
    );
    CATCH_REQUIRE(index.size() == test_dataset::VECTORS_IN_DATA_SET);
    CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);

    svs_test::prepare_temp_directory();
    auto dir = svs_test::temp_directory();

    auto config_dir = dir / "config";
    auto graph_dir = dir / "graph";
    auto data_dir = dir / "data";
    index.save(config_dir, graph_dir, data_dir);

    // Reload
    auto reloaded = svs::Vamana::assemble<float>(
        config_dir,
        svs::GraphLoader(graph_dir),
        svs::lib::Lazy([&]() { return svs::lib::load_from_disk<Data>(data_dir); }),
        distance,
        num_threads
    );
    CATCH_REQUIRE(reloaded.get_num_threads() == num_threads);
    CATCH_REQUIRE(reloaded.size() == test_dataset::VECTORS_IN_DATA_SET);
    CATCH_REQUIRE(reloaded.dimensions() == test_dataset::NUM_DIMENSIONS);
    run_search(index, queries, groundtruth, expected_results.config_and_recall_, false);

    // Determine the number of LVQ bits.
    size_t lvq_bits = 0;
    switch (secondary_kind) {
        // Order from highest number of LVQ bits to the lowest number so if we forget a
        // fallthrough, at least the test fails.
        case svsbenchmark::LeanVecKind::lvq8: {
            lvq_bits = 8;
            break;
        }
        case svsbenchmark::LeanVecKind::lvq4: {
            lvq_bits = 4;
            break;
        }
        case svsbenchmark::LeanVecKind::float32:
        case svsbenchmark::LeanVecKind::float16: {
            lvq_bits = 0;
            break;
        }
    }
    check_reconstruction(index, original.cview(), lvq_bits);
}

} // namespace

CATCH_TEST_CASE("LeanVec Vamana Search", "[integration][search][vamana][leanvec]") {
    namespace vamana = svs::index::vamana;

    CATCH_SECTION("Calibration Extensions") {
        CATCH_REQUIRE(vamana::extensions::calibration_uses_reranking<
                      leanvec::LeanDataset<float, float, 64, 128>>());
        CATCH_REQUIRE(vamana::extensions::calibration_uses_reranking<
                      leanvec::LeanDataset<svs::Float16, svs::Float16, 64, 64>>());
        CATCH_REQUIRE(vamana::extensions::calibration_uses_reranking<
                      leanvec::LeanDataset<leanvec::UsingLVQ<8>, svs::Float16, 32, 64>>());
        CATCH_REQUIRE(
            vamana::extensions::calibration_uses_reranking<
                leanvec::LeanDataset<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, 32, 64>>()
        );
        CATCH_REQUIRE(vamana::extensions::calibration_uses_reranking<
                      leanvec::LeanDataset<float, leanvec::UsingLVQ<8>, 32, 64>>());
    }

    const size_t N = 128;
    auto datafile = test_dataset::data_svs_file();
    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();

    auto extents = std::make_tuple(svs::lib::Val<N>(), svs::lib::Val<svs::Dynamic>());

    svs::lib::foreach (extents, [&]<size_t E>(svs::lib::Val<E> /*unused*/) {
        fmt::print("LeanVec Search - Extent {}\n", E);
        auto distance = svs::distance::DistanceL2();
        auto data = svs::data::SimpleData<float, E>::load(datafile);
        bool try_calibration = (E == svs::Dynamic);

        test_search(
            leanvec::LeanDataset<float, float, 64, E>::reduce(data),
            data,
            distance,
            queries,
            gt,
            true, // PCA
            try_calibration
        );

        auto leanvec_dims = svs::lib::MaybeStatic<svs::Dynamic>(64);
        test_search(
            leanvec::LeanDataset<float, float, svs::Dynamic, E>::reduce(
                data, 1, 0, leanvec_dims
            ),
            data,
            distance,
            queries,
            gt
        );
        test_search(
            leanvec::LeanDataset<leanvec::UsingLVQ<4>, leanvec::UsingLVQ<4>, 64, E>::reduce(
                data
            ),
            data,
            distance,
            queries,
            gt
        );
        test_search(
            leanvec::LeanDataset<leanvec::UsingLVQ<4>, leanvec::UsingLVQ<8>, 64, E>::reduce(
                data
            ),
            data,
            distance,
            queries,
            gt
        );
        test_search(
            leanvec::LeanDataset<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<4>, 64, E>::reduce(
                data
            ),
            data,
            distance,
            queries,
            gt
        );
        test_search(
            leanvec::LeanDataset<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, 64, E>::reduce(
                data
            ),
            data,
            distance,
            queries,
            gt
        );
        test_search(
            leanvec::LeanDataset<float, float, 96, E>::reduce(data),
            data,
            distance,
            queries,
            gt
        );
        test_search(
            leanvec::LeanDataset<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, 96, E>::reduce(
                data
            ),
            data,
            distance,
            queries,
            gt
        );

        // LeanVec OOD
        const size_t LeanVecDims = 64;
        auto data_matrix = test_dataset::leanvec_data_matrix<LeanVecDims>();
        auto query_matrix = test_dataset::leanvec_query_matrix<LeanVecDims>();
        auto matrices =
            std::optional(leanvec::LeanVecMatrices<LeanVecDims>(data_matrix, query_matrix));

        test_search(
            leanvec::LeanDataset<float, float, LeanVecDims, E>::reduce(data, matrices),
            data,
            distance,
            queries,
            gt,
            false // Not PCA
        );
        test_search(
            leanvec::
                LeanDataset<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, LeanVecDims, E>::
                    reduce(data, matrices),
            data,
            distance,
            queries,
            gt,
            false // Not PCA
        );
    });
}
