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

// Header under test
#include "svs/quantization/lvq/vectors.h"

// test utilities
#include "tests/utils/generators.h"
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

/////
///// Test Helpers
/////

namespace lvq = svs::quantization::lvq;

namespace {
template <typename Data> void apply_bias(Data& data, const std::vector<float>& bias) {
    CATCH_REQUIRE((bias.size() == data.dimensions()));
    for (size_t i = 0, imax = data.size(); i < imax; ++i) {
        auto datum = data.get_datum(i);
        std::transform(
            datum.begin(), datum.end(), bias.begin(), datum.begin(), std::plus()
        );
    }
}

template <typename Data, typename Queries, typename Distance>
svs::Matrix<float> compute_distances(
    const Data& data,
    const Queries& queries,
    Distance& distance,
    size_t num_queries,
    size_t num_data
) {
    svs::Matrix<float> distances{svs::make_dims(num_queries, num_data)};
    for (size_t i = 0; i < num_queries; ++i) {
        const auto& query = queries.get_datum(i);
        svs::distance::maybe_fix_argument(distance, query);
        for (size_t j = 0; j < num_data; ++j) {
            distances.at(i, j) = svs::distance::compute(distance, query, data.get_datum(j));
        }
    }
    return distances;
}

template <typename Distance, typename BiasDistance>
void test_biased_distance(double eps = 0.0001, double margin = 0.01, bool verbose = false) {
    const size_t num_queries = 10;
    const size_t num_data = 100;

    auto data = test_dataset::data_f32();
    auto biased_data = test_dataset::data_f32();

    // Get only the first 10-queries to keep run-time shorter.
    auto queries = test_dataset::queries();
    size_t ndims = data.dimensions();

    // Create a random per-entry bias.
    auto bias = std::vector<float>(ndims);
    svs_test::populate(bias, svs_test::make_generator<float>(-128, 128), ndims);
    apply_bias(biased_data, bias);

    auto distance = Distance{};
    auto biased_distance = BiasDistance{bias};
    CATCH_REQUIRE((biased_distance == biased_distance));

    auto a = compute_distances(biased_data, queries, distance, num_queries, num_data);
    auto b = compute_distances(data, queries, biased_distance, num_queries, num_data);

    CATCH_REQUIRE((a.size() == b.size()));
    for (size_t i = 0; i < svs::getsize<0>(a); ++i) {
        for (size_t j = 0; j < svs::getsize<1>(a); ++j) {
            if (verbose) {
                std::cout << "a[" << i << ", " << j << "] = " << a.at(i, j) << ", ";
                std::cout << "b[" << i << ", " << j << "] = " << b.at(i, j) << std::endl;
            }
            auto target = Catch::Approx(b.at(i, j)).epsilon(eps).margin(margin);
            CATCH_REQUIRE((a.at(i, j) == target));
        }
    }
}
} // namespace

/////
///// Tests
/////

CATCH_TEST_CASE("Global Vector Bias", "[quantizaiton][lvq]") {
    test_biased_distance<svs::distance::DistanceL2, lvq::EuclideanBiased>();
    test_biased_distance<svs::distance::DistanceIP, lvq::InnerProductBiased>();
}
