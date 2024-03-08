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

template <typename T, size_t N> void test_lvq_top() {
    // First, construct an online compression.
    auto original = svs::data::SimpleData<float, N>::load(test_dataset::data_svs_file());
    auto lvq_dataset = T::compress(original);

    CATCH_REQUIRE(lvq_dataset.size() == original.size());
    CATCH_REQUIRE(lvq_dataset.dimensions() == original.dimensions());

    // Try saving and reloading.
    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    svs::lib::save_to_disk(lvq_dataset, temp_dir);
    auto reloaded_lvq_dataset = svs::lib::load_from_disk<T>(temp_dir);

    static_assert(std::is_same_v<decltype(lvq_dataset), decltype(reloaded_lvq_dataset)>);
    CATCH_REQUIRE(lvq_dataset.size() == reloaded_lvq_dataset.size());

    // Check for logical equivalence.
    for (size_t i = 0, imax = lvq_dataset.size(); i < imax; ++i) {
        CATCH_REQUIRE(lvq::logically_equal(
            lvq_dataset.get_datum(i), reloaded_lvq_dataset.get_datum(i)
        ));
    }

    // Check decompression against the original dataset.
    auto accessor = lvq::DecompressionAccessor(lvq_dataset);
    auto centroids = lvq_dataset.view_centroids();
    CATCH_REQUIRE(centroids);

    // Buffer for datasets elements with the centroid removed.
    auto buffer = std::vector<double>(original.dimensions());

    // The total bits of precision available to the dataset.
    constexpr size_t primary_bits = T::primary_bits;
    constexpr size_t residual_bits = T::residual_bits;

    for (size_t i = 0, imax = lvq_dataset.size(); i < imax; ++i) {
        auto datum = original.get_datum(i);
        auto lvq_datum = lvq_dataset.get_datum(i);
        auto selector = lvq_datum.get_selector();
        auto centroid = centroids->get_datum(selector);
        CATCH_REQUIRE(centroid.size() == buffer.size());

        // Subtract out the associated centroid and record the minimum and maximum values
        // for each coordinate.
        double max_val = std::numeric_limits<double>::min();
        double min_val = std::numeric_limits<double>::max();
        for (size_t j = 0; j < centroid.size(); ++j) {
            double r = datum[j] - centroid[j];
            buffer[j] = r;
            max_val = std::max(max_val, r);
            min_val = std::min(min_val, r);
        }

        // We have an actual delta (computed in high-precision floating point numbers),
        // and an expected delta computed from the accuracy loss due to float16.
        //
        // Ensure that the actual scaling parameter is close to the expected scaling
        // parameter - then use the actual scale for accuracy comparison.
        auto numerator = lvq::through_scaling_type(
            (max_val - min_val) / (std::pow(2.0, primary_bits) - 1)
        );
        auto denominator = residual_bits == 0 ? 1.0 : (std::pow(2.0, residual_bits) - 1);
        auto expected_scale = numerator / denominator;

        auto actual_scale = lvq_datum.get_scale();
        CATCH_REQUIRE(std::abs(actual_scale / expected_scale - 1.0) < 0.001);

        // The actual scale gives the size between steps in the LVQ encoding. Since we
        // round to the nearest integer when compressing, the actual expected delta is
        // half of the step size.
        double delta = actual_scale / 2 + 0.0001;

        // Reconstruct the original vector.
        auto reconstructed = accessor(lvq_dataset, i);
        CATCH_REQUIRE(reconstructed.size() == datum.size());

        double max_error = 0.0;
        for (size_t j = 0; j < buffer.size(); ++j) {
            // Compute the actual error between the reconstructed dimensions and the
            // original dimension.
            auto error = std::abs(
                static_cast<double>(reconstructed[j]) - static_cast<double>(datum[j])
            );

            // Error checking for one-level compression is straight-forward.
            CATCH_REQUIRE(error <= delta);
            max_error = std::max(max_error, error);
        }
        // Ensure our bound is tight.
        CATCH_REQUIRE(max_error / delta > 0.91);
    }

    // Check matchers.
    auto m = svs::lib::load_from_disk<lvq::Matcher>(temp_dir);
    CATCH_REQUIRE(m.primary == T::primary_bits);
    CATCH_REQUIRE(m.residual == T::residual_bits);
    CATCH_REQUIRE(m.dims == lvq_dataset.dimensions());

    // Try-load should yield the same result.
    auto ex = svs::lib::try_load_from_disk<lvq::Matcher>(temp_dir);
    CATCH_REQUIRE(ex);
    CATCH_REQUIRE(ex.value() == m);

    // Change the underlying schema to ensure schema mismatches are handled.
    auto src = temp_dir / svs::lib::config_file_name;
    auto dst = temp_dir / "modified.toml";
    svs_test::mutate_table(src, dst, {{{"object", "__schema__"}, "invalid_schema"}});
    ex = svs::lib::try_load_from_disk<lvq::Matcher>(dst);
    CATCH_REQUIRE(!ex);
    CATCH_REQUIRE(ex.error() == svs::lib::TryLoadFailureReason::InvalidSchema);
}

} // namespace

CATCH_TEST_CASE("End-to-End Vector Quantization", "[quantization][lvq][lvq_dataset]") {
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
        test_lvq_top<lvq::LVQDataset<4, 8, 128>, 128>();
    }
}
