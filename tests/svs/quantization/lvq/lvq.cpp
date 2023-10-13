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

// Header under test.
#include "svs/quantization/lvq/lvq.h"

// Extras
#include "svs/lib/saveload.h"

// test utilities
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace lvq = svs::quantization::lvq;
using DistanceL2 = svs::distance::DistanceL2;
using DistanceIP = svs::distance::DistanceIP;

namespace {

// // A combination of reference data to put in an LVQ dataset and centroids that should be
// // able to test centroid-selection logic.
// struct ReferenceAndCentroids {
//     svs::data::SimpleData<float> reference_;
//     svs::data::SimpleData<float> centroids_;
// };
//
// ReferenceAndCentroids generate_reference_data(
//     size_t size, size_t dimensions, size_t primary_bits, size_t residual_bits
// ) {
//
// }

template <typename T, size_t N> void test_lvq_top() {
    // First, construct an online compression.
    auto lvq_dataset =
        T::compress(svs::data::SimpleData<float, N>::load(test_dataset::data_svs_file()));

    // Try saving and reloading.
    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    svs::lib::save_to_disk(lvq_dataset, temp_dir);
    auto reloaded_lvq_dataset = svs::lib::load_from_disk<T>(temp_dir);
    static_assert(std::is_same_v<decltype(lvq_dataset), decltype(reloaded_lvq_dataset)>);
}

} // namespace

CATCH_TEST_CASE("End-to-End Vector Quantization", "[quantization][lvq]") {
    CATCH_SECTION("OnlineCompression") {
        // Make sure we can construct an instance of "OnlineCompression" using one of the
        // "blessed" source types.
        auto x = lvq::OnlineCompression("a path!", svs::DataType::float32);
        CATCH_REQUIRE(x.path == "a path!");
        CATCH_REQUIRE(x.type == svs::DataType::float32);

        x = lvq::OnlineCompression("another path!", svs::DataType::float16);
        CATCH_REQUIRE(x.path == "another path!");
        CATCH_REQUIRE(x.type == svs::DataType::float16);

        // Incompatible type.
        CATCH_REQUIRE_THROWS_AS(
            lvq::OnlineCompression("another path!", svs::DataType::float64),
            svs::ANNException
        );
    }

    // One Level Compression.
    CATCH_SECTION("One Level Compression") {
        test_lvq_top<lvq::LVQDataset<8, 0, 128>, 128>();
        test_lvq_top<lvq::LVQDataset<4, 0, 128>, 128>();
    }

    CATCH_SECTION("Two Level Compression") {
        test_lvq_top<lvq::LVQDataset<4, 4, 128>, 128>();
        test_lvq_top<lvq::LVQDataset<4, 4, 128>, 128>();
    }
}
