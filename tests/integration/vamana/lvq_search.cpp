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
#include "svs/extensions/vamana/lvq.h"
#include "svs/lib/saveload.h"
#include "svs/orchestrators/vamana.h"

// tests
#include "tests/utils/lvq_reconstruction.h" // To check LVQ reconstruction.
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/datasets/lvq.h"

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

namespace lvq = svs::quantization::lvq;
namespace {

template <lvq::IsLVQDataset Data> consteval bool is_turbo_compatible() {
    return Data::primary_bits == 4 &&
           (Data::residual_bits == 0 || Data::residual_bits == 8);
}

void check_reconstruction(
    svs::Vamana& index,
    const svs::data::SimpleData<float>& original,
    size_t primary,
    size_t residual
) {
    auto ids = svs_test::permute_indices(original.size());

    // Reconstruct
    auto dst = svs::data::SimpleData<float>(original.size(), original.dimensions());
    index.reconstruct_at(dst.view(), ids);

    auto shuffled = svs::data::SimpleData<float>(original.size(), original.dimensions());
    for (size_t i = 0; i < original.size(); ++i) {
        shuffled.set_datum(i, original.get_datum(ids.at(i)));
    }

    svs_test::check_lvq_reconstruction(shuffled.cview(), dst.cview(), primary, residual);
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

template <lvq::IsLVQDataset Data, typename Distance>
void test_search(
    Data data,
    const svs::data::SimpleData<float>& original,
    const Distance& distance,
    const svs::data::SimpleData<float>& queries,
    const svs::data::SimpleData<uint32_t>& groundtruth,
    bool try_calibration = false
) {
    size_t num_threads = 2;

    // Ensure the incoming data is not aligned.
    // Reloading with alignment is done when reloading.
    CATCH_REQUIRE(data.primary_dataset_alignment() == 0);

    // Find the expected results for this dataset.
    auto expected_results = test_dataset::vamana::expected_search_results(
        svs::distance_type_v<Distance>,
        svsbenchmark::LVQ(
            Data::primary_bits,
            Data::residual_bits,
            svsbenchmark::LVQPackingStrategy::Sequential // not-checked
        )
    );

    // Make a copy of the original data to use for reconstruction comparison.
    auto index = svs::Vamana::assemble<float>(
        test_dataset::vamana_config_file(),
        svs::GraphLoader(test_dataset::graph_file()),
        std::move(data),
        distance,
        num_threads
    );
    CATCH_REQUIRE(index.get_num_threads() == num_threads);
    check_reconstruction(index, original, Data::primary_bits, Data::residual_bits);

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
    {
        auto reloaded_data = svs::lib::load_from_disk<Data>(data_dir, 32);
        CATCH_REQUIRE(reloaded_data.primary_dataset_alignment() == 32);
        auto reloaded = svs::Vamana::assemble<float>(
            config_dir,
            svs::GraphLoader(graph_dir),
            std::move(reloaded_data),
            distance,
            num_threads
        );
        CATCH_REQUIRE(reloaded.get_num_threads() == num_threads);
        CATCH_REQUIRE(reloaded.size() == test_dataset::VECTORS_IN_DATA_SET);
        CATCH_REQUIRE(reloaded.dimensions() == test_dataset::NUM_DIMENSIONS);
        run_search(index, queries, groundtruth, expected_results.config_and_recall_, false);
    }

    // Test switching strategies if appropriate.
    if constexpr (is_turbo_compatible<Data>()) {
        using ReloadStrategy = std::
            conditional_t<lvq::UsesSequential<Data>, lvq::Turbo<16, 8>, lvq::Sequential>;

        using ReloadData = lvq::LVQDataset<
            Data::primary_bits,
            Data::residual_bits,
            Data::extent,
            ReloadStrategy,
            typename Data::allocator_type>;

        auto reloaded = svs::Vamana::assemble<float>(
            config_dir,
            svs::GraphLoader(graph_dir),
            svs::lib::Lazy([&]() { return svs::lib::load_from_disk<ReloadData>(data_dir); }
            ),
            distance,
            num_threads
        );
        run_search(index, queries, groundtruth, expected_results.config_and_recall_, false);
    }
}
} // namespace

CATCH_TEST_CASE("LVQ Vamana Search", "[integration][search][vamana][lvq]") {
    namespace vamana = svs::index::vamana;

    CATCH_SECTION("Calibration Extensions") {
        CATCH_REQUIRE(vamana::extensions::calibration_uses_reranking<
                      lvq::LVQDataset<8, 8, svs::Dynamic>>());
        CATCH_REQUIRE(vamana::extensions::calibration_uses_reranking<
                      lvq::LVQDataset<4, 8, svs::Dynamic>>());
        CATCH_REQUIRE(!vamana::extensions::calibration_uses_reranking<
                      lvq::LVQDataset<8, 0, svs::Dynamic>>());
        CATCH_REQUIRE(!vamana::extensions::calibration_uses_reranking<
                      lvq::LVQDataset<4, 0, svs::Dynamic>>());
    }

    const size_t N = 128;
    auto datafile = test_dataset::data_svs_file();
    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();

    auto extents = std::make_tuple(svs::lib::Val<N>(), svs::lib::Val<svs::Dynamic>());
    auto original = svs::data::SimpleData<float>::load(datafile);

    svs::lib::foreach (extents, [&]<size_t E>(svs::lib::Val<E> /*unused*/) {
        fmt::print("LVQ Search - Extent {}\n", E);
        auto distance = svs::distance::DistanceL2();
        auto data = svs::data::SimpleData<float, E>::load(datafile);
        bool try_calibration = (E == svs::Dynamic);

        // Sequential tests
        test_search(
            lvq::LVQDataset<8, 0, E>::compress(data), original, distance, queries, gt
        );
        test_search(
            lvq::LVQDataset<4, 0, E>::compress(data), original, distance, queries, gt
        );

        test_search(
            lvq::LVQDataset<4, 4, E>::compress(data),
            original,
            distance,
            queries,
            gt,
            try_calibration
        );
        test_search(
            lvq::LVQDataset<4, 8, E>::compress(data), original, distance, queries, gt
        );
        test_search(
            lvq::LVQDataset<8, 8, E>::compress(data), original, distance, queries, gt
        );

        // Turbo tests
        test_search(
            lvq::LVQDataset<4, 0, E, lvq::Turbo<16, 8>>::compress(data),
            original,
            distance,
            queries,
            gt
        );
        test_search(
            lvq::LVQDataset<4, 8, E, lvq::Turbo<16, 8>>::compress(data),
            original,
            distance,
            queries,
            gt
        );
    });
}
