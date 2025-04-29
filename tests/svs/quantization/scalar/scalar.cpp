/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// svs
#include "svs/quantization/scalar/scalar.h"
#include "svs/core/data/simple.h"
#include "svs/lib/meta.h"

#include "tests/svs/core/data/data.h"
#include "tests/utils/generators.h"
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace scalar = svs::quantization::scalar;

template <typename T, size_t N> void test_sq_top() {
    // First, construct an online compression.
    auto original = svs::data::SimpleData<float, N>::load(test_dataset::data_svs_file());
    auto sq_dataset = scalar::SQDataset<T, N>::compress(original);

    // Assert compressed data has the same size and dimensions
    CATCH_REQUIRE(sq_dataset.size() == original.size());
    CATCH_REQUIRE(sq_dataset.dimensions() == original.dimensions());

    // Assert scale and bias are calculated correctly
    // Scale is calculated from (max_data - min_data) / (max_quant - min_quant)
    // The dataset features values [-127, 127], the quantization range is given by the MIN
    // and MAX elements of the provided type.
    constexpr float MIN = std::numeric_limits<T>::min();
    constexpr float MAX = std::numeric_limits<T>::max();
    constexpr float exp_scale = 254.0F / float(MAX - MIN);
    // Bias is calculated as min_data - min_quant * scale
    constexpr float exp_bias = -127.0F - MIN * exp_scale;
    // Calculations are performed in float everywhere and should therefore produce the exact
    // same results
    CATCH_REQUIRE(sq_dataset.get_scale() == exp_scale);
    CATCH_REQUIRE(sq_dataset.get_bias() == exp_bias);

    // Try saving and reloading.
    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    svs::lib::save_to_disk(sq_dataset, temp_dir);
    auto reloaded_sq_dataset = svs::lib::load_from_disk<decltype(sq_dataset)>(temp_dir);

    // Type is reconstructed correctly
    static_assert(std::is_same_v<decltype(sq_dataset), decltype(reloaded_sq_dataset)>);

    // Values don't change
    CATCH_REQUIRE(sq_dataset.size() == reloaded_sq_dataset.size());
    CATCH_REQUIRE(sq_dataset.get_scale() == reloaded_sq_dataset.get_scale());
    CATCH_REQUIRE(sq_dataset.get_bias() == reloaded_sq_dataset.get_bias());
    // Check for equality
    for (size_t i = 0, imax = sq_dataset.size(); i < imax; ++i) {
        // compare all elements in span at index i
        auto a = sq_dataset.get_datum(i);
        auto b = reloaded_sq_dataset.get_datum(i);
        CATCH_REQUIRE(a.size() == b.size());
        for (size_t j = 0; j < a.size(); ++j) {
            CATCH_REQUIRE(a[j] == b[j]);
        }
    }

    // Worst case is being off by one in the compression.
    // Because we are shifting, we should in fact never be off by more than (-0.5, 0.5)
    // in the compressed range.
    // A value "1" in the compressed range corresponds to the value of "scale" in the
    // uncompressed range.
    // We already verified the scale is calculated correctly above.
    float max_error = 0.5 * sq_dataset.get_scale();

    auto delta = std::vector<float>(original.dimensions());
    for (size_t i = 0; i < original.size(); ++i) {
        auto datum = original.get_datum(i);
        auto sq_datum = sq_dataset.decompress_datum(i);

        for (size_t j = 0; j < N; ++j) {
            float r = datum[j] - sq_datum[j];
            CATCH_REQUIRE(std::abs(r) < max_error);
        }
    }

    // Set a datum and check if it is the same.
    auto datum = original.get_datum(0);
    sq_dataset.set_datum(0, datum);
    auto sq_datum = sq_dataset.decompress_datum(0);
    for (size_t j = 0; j < N; ++j) {
        float r = datum[j] - sq_datum[j];
        CATCH_REQUIRE(std::abs(r) < max_error);
    }
}

template <typename T, typename Distance, size_t N> void test_distance_single(T lo, T hi) {
    constexpr size_t num_tests = 100;
    std::vector<T> a; // random query
    std::vector<T> b; // random compressed data
    auto generator = svs_test::make_generator<T>(lo, hi);
    auto float_generator = svs_test::make_generator<float>(-3, 3);

    // Generate random scale and bias
    float scale = svs_test::generate(float_generator);
    float bias = svs_test::generate(float_generator);
    // Decompression helper
    auto decompress = [scale, bias](T val) { return scale * float(val) + bias; };

    // Populate query
    svs_test::populate(a, generator, N);
    std::vector<float> buffer(N);
    std::transform(a.begin(), a.end(), buffer.begin(), decompress);
    auto query = svs::data::SimpleData<float>{1, N};
    query.set_datum(0, buffer);

    // Populate dataset
    auto bdata = svs::data::SimpleData<T, N>{num_tests, N};
    for (size_t i = 0; i < num_tests; ++i) {
        svs_test::populate(b, generator, N);
        bdata.set_datum(i, b);
    }
    auto compressed = scalar::SQDataset<T, N>(bdata, scale, bias);

    // Prepare distances
    Distance distance;
    auto compressed_distance = scalar::compressed_distance_t<Distance, T>{scale, bias, N};
    svs::distance::maybe_fix_argument(distance, query.get_datum(0));
    svs::distance::maybe_fix_argument(compressed_distance, query.get_datum(0));

    // A buffer into which we decompress the int8 values
    std::vector<float> rhs(N);

    for (size_t i = 0; i < num_tests; ++i) {
        auto datum = compressed.get_datum(i);
        std::transform(datum.begin(), datum.end(), rhs.begin(), decompress);
        auto rhs_span = std::span(rhs);
        float reference = svs::distance::compute(distance, query.get_datum(0), rhs_span);
        auto expected = Catch::Approx(reference).epsilon(0.01).margin(0.01);

        // Calculate compressed distance and compare with reference
        float result = compressed_distance.compute(compressed.get_datum(i));
        CATCH_REQUIRE(result == expected);
    }
}

template <typename T, typename Distance> void test_distance() {
    // Error accumulates proportional to number of dimensions, perform a low-dim test
    test_distance_single<std::int8_t, Distance, 2>(-127, 127);

    // More realistic, higher dimensionality tests for SIMD lanes with unrolling.
    // 16x4 = 64 unrolled, plus full epilogue (16), plus ragged epilogue (7)
    constexpr size_t N = 64 + 16 + 7;
    // a bunch of test cases from different ranges of the int8 spectrum
    test_distance_single<std::int8_t, Distance, N>(-127, 127);
    test_distance_single<std::int8_t, Distance, N>(80, 100);
    test_distance_single<std::int8_t, Distance, N>(-10, 10);
    test_distance_single<std::int8_t, Distance, N>(-100, -80);
    test_distance_single<std::int8_t, Distance, N>(-1, 1);
    test_distance_single<std::int8_t, Distance, N>(-10, 1);
}

CATCH_TEST_CASE("Testing SQDataset", "[quantization][scalar]") {
    CATCH_SECTION("Default SQDataset") {}

    CATCH_SECTION("SQDataset dynamic extent") {
        auto x = scalar::SQDataset<std::int8_t>(10, 100);

        CATCH_REQUIRE(x.size() == 10);
        CATCH_REQUIRE(x.dimensions() == 100);
        CATCH_REQUIRE(x.extent == svs::Dynamic);
    }

    CATCH_SECTION("SQDataset fixed extent") {
        constexpr size_t dims = 128;
        auto x = scalar::SQDataset<std::int8_t, dims>({}, 1.0F, 0.0F);

        CATCH_REQUIRE(x.size() == 0);
        CATCH_REQUIRE(x.dimensions() == dims);
        CATCH_REQUIRE(x.get_scale() == 1.0F);
        CATCH_REQUIRE(x.get_bias() == 0.0F);
        CATCH_REQUIRE(x.extent == dims);
    }

    CATCH_SECTION("SQDataset compression") {
        test_sq_top<std::int8_t, 128>();
        test_sq_top<std::int16_t, 128>();
    }

    CATCH_SECTION("SQDataset compress and resize") {
        // TODO
    }
}

CATCH_TEST_CASE(
    "Testing Distance computations with SQDataset", "[quantization][scalar][distance]"
) {
    CATCH_SECTION("Distance with SQDataset") {
        // IP and CS use the float32 query for computation
        using DistanceL2 = svs::distance::DistanceL2;
        using DistanceIP = svs::distance::DistanceIP;
        using DistanceCS = svs::distance::DistanceCosineSimilarity;

        test_distance<std::int8_t, DistanceL2>();
        test_distance<std::int8_t, DistanceIP>();
        test_distance<std::int8_t, DistanceCS>();

        // L2 computes with compressed query and data and the check works a bit differently
    }
}
