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

// header under test.
#include "svs/core/kmeans.h"

// test utils
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

// Initialize the base dataset with known contents.
svs::data::SimpleData<float> gen_data(size_t count = 100) {
    auto data = svs::data::SimpleData<float>(count, 4);
    auto buffer = std::vector<float>(data.dimensions());
    for (auto i : data.eachindex()) {
        std::fill(buffer.begin(), buffer.end(), i);
        data.set_datum(i, svs::lib::as_const_span(buffer));
    }
    return data;
}

} // namespace

CATCH_TEST_CASE("KMeans Clustering", "[core][kmeans]") {
    CATCH_SECTION("Find Nearest") {
        auto data = gen_data();
        auto buffer = std::vector<float>(data.dimensions());

        // Make sure the correct constructor is called because C++ ...
        CATCH_REQUIRE(buffer.size() == data.dimensions());

        // Use a shift amount that is exactly representable as a floating point
        // number.
        const float shift = 0.125;
        for (size_t i = 0, imax = data.size(); i < imax; ++i) {
            std::fill(buffer.begin(), buffer.end(), i + shift);
            float expected_distance = (shift * shift) * buffer.size();
            auto nearest_neighbor =
                svs::find_nearest(svs::lib::as_const_span(buffer), data);
            CATCH_REQUIRE(nearest_neighbor.id() == i);
            CATCH_REQUIRE(nearest_neighbor.distance() == expected_distance);
        }
    }

    CATCH_SECTION("MSE Measurement") {
        // For the MSE, set up the data and centroids like so:
        //
        // DATA:            CENTROIDS
        //  0  0  0  0      1.5  1.5  1.5  1.5
        //  1  1  1  1      5.5  5.5  5.5  5.5
        //  2  2  2  2      9.5  9.5  9.5  9.5
        //  3  3  3  3
        //
        //  4  4  4  4
        //  5  5  5  5
        //  6  6  6  6
        //  7  7  7  7
        //
        //  8  8  8  8
        //  9  9  9  9
        // 10 10 10 10
        // 11 11 11 11
        //
        // Then, each set of four elements in data will be closest to one centroid.
        // Further, the MSE is easily computed.
        auto data = gen_data(12);
        auto centroids = svs::data::SimpleData<float>(4, data.dimensions());

        CATCH_REQUIRE(data.size() % centroids.size() == 0);
        for (size_t i = 0; i < centroids.size(); ++i) {
            float val = 1.5 + 4 * i;
            for (auto& v : centroids.get_datum(i)) {
                v = val;
            }
        }

        auto threadpool = svs::threads::NativeThreadPool(4);
        double mse = svs::mean_squared_error(data, centroids, threadpool);
        double expected_mse = centroids.dimensions() * ((0.5 * 0.5) + (1.5 * 1.5)) / 2;
        CATCH_REQUIRE(mse == expected_mse);
    }
}
