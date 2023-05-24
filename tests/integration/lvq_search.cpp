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
#include "svs/index/vamana/index.h"
#include "svs/lib/saveload.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

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
const bool CALIBRATE = false;
constexpr std::array<size_t, 5> windowsizes{2, 3, 5, 10, 20};

namespace {

// We need a scheme to map compression techniques to collections of pre-determined recall
// values.
//
// Build up a set of "NameBuilder" objects to convert the various compression techniques
// to unique string names that are suitable for dictionary keys.
template <typename T> struct NameBuilder;

template <size_t Primary, size_t Dims>
struct NameBuilder<lvq::OneLevelWithBias<Primary, Dims>> {
    static std::string key() { return fmt::format("LVQ{}", Primary); }
};
template <size_t Primary, size_t Residual, size_t Dims>
struct NameBuilder<lvq::TwoLevelWithBias<Primary, Residual, Dims>> {
    static std::string key() { return fmt::format("LVQ{}x{}", Primary, Residual); }
};
template <size_t Primary, size_t Dims>
struct NameBuilder<lvq::GlobalOneLevelWithBias<Primary, Dims>> {
    static std::string key() { return fmt::format("Global{}", Primary); }
};
template <size_t Primary, size_t Residual, size_t Dims>
struct NameBuilder<lvq::GlobalTwoLevelWithBias<Primary, Residual, Dims>> {
    static std::string key() { return fmt::format("Global{}x{}", Primary, Residual); }
};

template <typename T> std::string get_key() { return NameBuilder<T>::key(); }

// Expected recall dictionary based on calibration.
using RecallMapType =
    std::unordered_map<std::string, std::vector<std::pair<size_t, double>>>;
std::vector<std::pair<size_t, double>> get_recall(const std::string& key) {
    static auto map = RecallMapType{
        {"LVQ8", {{2, 0.4575}, {3, 0.53833}, {5, 0.6438}, {10, 0.7584}, {20, 0.85925}}},
        {"LVQ4", {{2, 0.4225}, {3, 0.498}, {5, 0.5966}, {10, 0.7055}, {20, 0.7883}}},
        {"LVQ4x4", {{2, 0.4225}, {3, 0.498}, {5, 0.5966}, {10, 0.7055}, {20, 0.7883}}},
        {"LVQ4x8", {{2, 0.4225}, {3, 0.498}, {5, 0.5966}, {10, 0.7055}, {20, 0.7883}}},
        {"LVQ8x8", {{2, 0.4575}, {3, 0.53833}, {5, 0.6438}, {10, 0.7584}, {20, 0.85925}}},
        {"Global8", {{2, 0.4565}, {3, 0.538}, {5, 0.6406}, {10, 0.7582}, {20, 0.8602}}},
        {"Global4x4", {{2, 0.377}, {3, 0.458}, {5, 0.5472}, {10, 0.6396}, {20, 0.7049}}}};
    return map.at(key);
};

template <typename Index>
void run_search(
    Index& index,
    const std::vector<std::pair<size_t, double>> window_recall,
    const svs::data::SimplePolymorphicData<float>& queries,
    const svs::data::SimplePolymorphicData<uint32_t>& groundtruth
) {
    double epsilon = 0.0001;
    for (auto [windowsize, recall] : window_recall) {
        index.set_search_window_size(windowsize);
        CATCH_REQUIRE(index.get_search_window_size() == windowsize);
        auto results = index.search(queries, windowsize);
        double achieved_recall =
            svs::k_recall_at_n(groundtruth, results, windowsize, windowsize);
        CATCH_REQUIRE(achieved_recall >= recall);
        CATCH_REQUIRE(achieved_recall <= recall + epsilon);
    }
}

template <typename DataProto, typename Distance>
void test_search(
    const DataProto& data_proto,
    const Distance& distance,
    const svs::data::SimplePolymorphicData<float>& queries,
    const svs::data::SimplePolymorphicData<uint32_t>& groundtruth
) {
    size_t num_threads = 2;
    auto index = svs::index::vamana::auto_assemble(
        test_dataset::vamana_config_file(),
        svs::GraphLoader(test_dataset::graph_file()),
        data_proto,
        distance,
        num_threads
    );
    CATCH_REQUIRE(index.get_num_threads() == num_threads);

    if constexpr (CALIBRATE) {
        auto recalls = std::vector<double>{};
        for (size_t windowsize : windowsizes) {
            index.set_search_window_size(windowsize);
            auto results = index.search(queries, windowsize);
            recalls.push_back(
                svs::k_recall_at_n(groundtruth, results, windowsize, windowsize)
            );
        }

        CATCH_REQUIRE(recalls.size() == windowsizes.size());
        fmt::print("{{\"{}\", {{", get_key<DataProto>());
        for (size_t i = 0, imax = recalls.size(); i < imax; ++i) {
            fmt::print("{}{{{}, {}}}", i == 0 ? "" : ", ", windowsizes[i], recalls[i]);
        }
        fmt::print("}}}}\n");
        return;
    }
    run_search(index, get_recall(get_key<DataProto>()), queries, groundtruth);
    CATCH_REQUIRE(index.size() == test_dataset::VECTORS_IN_DATA_SET);
    CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);

    svs_test::prepare_temp_directory();
    auto dir = svs_test::temp_directory();

    auto config_dir = dir / "config";
    auto graph_dir = dir / "graph";
    auto data_dir = dir / "data";
    index.save(config_dir, graph_dir, data_dir);

    // Reload
    auto reloaded = svs::index::vamana::auto_assemble(
        config_dir,
        svs::GraphLoader(graph_dir),
        DataProto{svs::quantization::lvq::Reload(data_dir)},
        distance,
        num_threads
    );
    CATCH_REQUIRE(reloaded.get_num_threads() == num_threads);
    CATCH_REQUIRE(reloaded.size() == test_dataset::VECTORS_IN_DATA_SET);
    CATCH_REQUIRE(reloaded.dimensions() == test_dataset::NUM_DIMENSIONS);
    run_search(reloaded, get_recall(get_key<DataProto>()), queries, groundtruth);
}
} // namespace

CATCH_TEST_CASE("Testing Search", "[integration][lvq_search]") {
    namespace vamana = svs::index::vamana;

    const size_t N = 128;
    auto data_file = test_dataset::data_svs_file();

    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();

    auto extents = std::make_tuple(svs::meta::Val<N>(), svs::meta::Val<svs::Dynamic>());

    svs::lib::foreach (extents, [&]<size_t E>(svs::meta::Val<E> /*unused*/) {
        fmt::print("LVQ Search - Extent {}\n", E);
        auto loader = svs::VectorDataLoader<float, E>(data_file);
        auto distance = svs::distance::DistanceL2();

        // Local
        test_search(lvq::OneLevelWithBias<8, E>(loader), distance, queries, gt);
        test_search(lvq::OneLevelWithBias<4, E>(loader), distance, queries, gt);
        test_search(lvq::TwoLevelWithBias<4, 4, E>(loader), distance, queries, gt);
        test_search(lvq::TwoLevelWithBias<4, 8, E>(loader), distance, queries, gt);
        test_search(lvq::TwoLevelWithBias<8, 8, E>(loader), distance, queries, gt);

        // Global
        test_search(lvq::GlobalOneLevelWithBias<8, E>(loader), distance, queries, gt);
        test_search(lvq::GlobalTwoLevelWithBias<4, 4, E>(loader), distance, queries, gt);
    });
}
