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

// stdlib
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

// svs
#include "svs/concepts/distance.h"
#include "svs/core/distance/cosine.h"
#include "svs/lib/array.h"
#include "svs/lib/float16.h"
#include "svs/lib/static.h"

// catch2
#include "catch2/benchmark/catch_benchmark_all.hpp"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// Testing Utilities
#include "tests/utils/benchmarks.h"
#include "tests/utils/generators.h"
#include "tests/utils/utils.h"

// The values for the floating point tolerance are chose somewhat heuristically based
// on looking at the values that failed tests with tighter tolerances.
namespace {
constexpr double COSINE_EPSILON = 0.01;
constexpr double COSINE_MARGIN = 0.015;
} // namespace

template <typename Ea, typename Eb>
double cosine_similarity_reference(const std::vector<Ea>& a, const std::vector<Eb>& b) {
    assert(a.size() == b.size());
    double accum = 0;
    double bnorm_accum = 0;
    double anorm_accum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        auto a_promoted = svs_test::promote(a[i]);
        auto b_promoted = svs_test::promote(b[i]);
        anorm_accum += a_promoted * a_promoted;
        bnorm_accum += b_promoted * b_promoted;
        accum += a_promoted * b_promoted;
    }
    return accum / (std::sqrt(anorm_accum) * std::sqrt(bnorm_accum));
}

// Testing utilities for common patterns
namespace {
template <typename Ea, typename Eb, size_t N, typename T>
void test_types(T lo, T hi, size_t num_tests) {
    // Vectors to hold populated random numbers.
    std::vector<Ea> a;
    std::vector<Eb> b;

    // Random number generators, one for each vector to support mixed types.
    auto generator_a = svs_test::make_generator<Ea>(lo, hi);
    auto generator_b = svs_test::make_generator<Eb>(lo, hi);

    for (size_t i = 0; i < num_tests; ++i) {
        svs_test::populate(a, generator_a, N);
        svs_test::populate(b, generator_b, N);
        auto expected = Catch::Approx(cosine_similarity_reference(a, b))
                            .epsilon(COSINE_EPSILON)
                            .margin(COSINE_MARGIN);

        // Statically Sized Computation
        auto a_norm = svs::distance::norm(std::span{a.data(), a.size()});
        CATCH_REQUIRE(
            (svs::distance::CosineSimilarity::compute<N>(a.data(), b.data(), a_norm) ==
             expected)
        );
        // Dynamically Sized Computation
        auto dist = svs::distance::CosineSimilarity::compute(a.data(), b.data(), a_norm, N);
        CATCH_REQUIRE((dist == expected));
    }
}
} // anonymous namespace

#if defined(NDEBUG)
const size_t NTESTS = 100000;
#else
const size_t NTESTS = 1000;
#endif

CATCH_TEST_CASE("Testing CosineSimilarity", "[distance][cosinesimilarity_distance]") {
    auto ntests = NTESTS;
    constexpr size_t ndims = 128;

    CATCH_SECTION("Float-Float") { test_types<float, float, ndims>(-1, 1, ntests); }
    CATCH_SECTION("Float-Float16") {
        test_types<float, svs::Float16, ndims>(-1, 1, ntests);
    }
    CATCH_SECTION("Float16-Float16") {
        test_types<svs::Float16, svs::Float16, ndims>(-1, 1, ntests);
    }
    CATCH_SECTION("Float-UInt8") { test_types<float, uint8_t, ndims>(0, 255, ntests); }
    CATCH_SECTION("UInt8-UInt8") { test_types<uint8_t, uint8_t, ndims>(0, 255, ntests); }
    CATCH_SECTION("Float-Int8") { test_types<float, int8_t, ndims>(-128, 127, ntests); }
    CATCH_SECTION("Int8-Int8") { test_types<int8_t, int8_t, ndims>(-128, 127, ntests); }

    // Saving and Loading
    CATCH_SECTION("Saving and Loading") {
        svs_test::cleanup_temp_directory();
        auto x = svs::distance::DistanceCosineSimilarity{};
        CATCH_REQUIRE(svs::lib::test_self_save_load(x, svs_test::temp_directory()));
    }
}

// This function is capable of running both the static and dynamically sized versions
// of the distance computation kernels.
//
// For the statically sized versions, pass the nonzero number of dimensions as the third
// template parameter.
//
// For the dynamically sized version, pass '0' for the template parameter and specify
// the number of dimensions as the trailing function argument.
namespace {

template <size_t N> auto constexpr forward_extent(size_t /*unused*/) {
    return svs::meta::Val<N>{};
}
template <> auto constexpr forward_extent<0>(size_t x) { return x; }

template <typename Ea, typename Eb, size_t N, typename T>
void run_benchmark(size_t num_elements, T lo, T hi, size_t ndims = N) {
    auto data = svs::make_dense_array<Eb>(num_elements, forward_extent<N>(ndims));
    CATCH_REQUIRE((svs::getsize<1>(data) == ndims));
    CATCH_REQUIRE((svs::getsize<0>(data) == num_elements));

    // Randomly populate the dataset.
    std::vector<Ea> a;
    std::vector<Eb> b;

    auto generator_a = svs_test::make_generator<Ea>(lo, hi);
    svs_test::populate(a, generator_a, ndims);

    auto generator_b = svs_test::make_generator<Eb>(lo, hi);
    std::vector<double> reference_distances{};
    for (size_t i = 0; i < svs::getsize<0>(data); ++i) {
        svs_test::populate(b, generator_b, ndims);
        // N.B.: Don't forget that within the code base, we return the negative of the
        // result for the inner product distance to keep comparisons uniform.
        reference_distances.emplace_back(cosine_similarity_reference(a, b));
        auto slice = data.slice(i);
        CATCH_REQUIRE((slice.size() == ndims));
        if (N != 0) {
            CATCH_REQUIRE((decltype(slice)::extent == N));
        }
        std::copy(b.begin(), b.end(), slice.begin());
    }

    std::vector<float> results(num_elements);
    auto distance = svs::distance::DistanceCosineSimilarity{};
    auto aspan = std::span(a);
    svs::distance::maybe_fix_argument(distance, aspan);
    NAMED_TEMPLATE_BENCHMARK("CosineSimilarity", (Ea, Eb, svs_test::Val<N>), ndims) {
        for (uint32_t i = 0; i < svs::getsize<0>(data); ++i) {
            results[i] = svs::distance::compute(distance, aspan, data.slice(i));
        }
    };

    // Sanity Check
    bool check = std::equal(
        results.begin(),
        results.end(),
        reference_distances.begin(),
        [](const float& x, const float& y) {
            return svs_test::isapprox_or_warn(x, y, COSINE_EPSILON, COSINE_MARGIN);
        }
    );
    CATCH_REQUIRE(check == true);
}
} // namespace

/////
///// Benchmarks
/////

CATCH_TEST_CASE(
    "Benchmark CosineSimilarity Distance",
    "[distance][cosinesimilarity_distance][benchmark_suite][!benchmark]"
) {
    auto num_elements = 1000000;
    // Types: `float` and `float`
    run_benchmark<float, float, 128>(num_elements, -1.0f, 1.0f);
    run_benchmark<float, float, 0>(num_elements, -1.0f, 1.0f, 128);
    run_benchmark<float, float, 100>(num_elements, -1.0f, 1.0f);
    run_benchmark<float, float, 0>(num_elements, -1.0f, 1.0f, 100);

    // Types: `float` and `svs::Float16`
    run_benchmark<float, svs::Float16, 128>(num_elements, -1.0f, 1.0f);
    run_benchmark<float, svs::Float16, 0>(num_elements, -1.0f, 1.0f, 128);
    run_benchmark<float, svs::Float16, 100>(num_elements, -1.0f, 1.0f);
    run_benchmark<float, svs::Float16, 0>(num_elements, -1.0f, 1.0f, 100);

    // Types: `svs::Float16` and `svs::Float16`
    run_benchmark<svs::Float16, svs::Float16, 128>(num_elements, -1.0f, 1.0f);
    run_benchmark<svs::Float16, svs::Float16, 0>(num_elements, -1.0f, 1.0f, 128);
    run_benchmark<svs::Float16, svs::Float16, 100>(num_elements, -1.0f, 1.0f);
    run_benchmark<svs::Float16, svs::Float16, 0>(num_elements, -1.0f, 1.0f, 100);

    // Types: `float` and `int8_t`
    run_benchmark<float, int8_t, 128>(num_elements, -128, 127);
    run_benchmark<float, int8_t, 0>(num_elements, -128, 127, 128);
    run_benchmark<float, int8_t, 100>(num_elements, -128, 127);
    run_benchmark<float, int8_t, 0>(num_elements, -128, 127, 100);

    // Types: `float` and `uint8_t`
    run_benchmark<float, uint8_t, 128>(num_elements, 0, 255);
    run_benchmark<float, uint8_t, 0>(num_elements, 0, 255, 128);
    run_benchmark<float, uint8_t, 100>(num_elements, 0, 255);
    run_benchmark<float, uint8_t, 0>(num_elements, 0, 255, 100);

    // Types: `uint8_t` and `uint8_t`
    run_benchmark<uint8_t, uint8_t, 128>(num_elements, 0, 255);
    run_benchmark<uint8_t, uint8_t, 0>(num_elements, 0, 255, 128);
    run_benchmark<uint8_t, uint8_t, 100>(num_elements, 0, 255);
    run_benchmark<uint8_t, uint8_t, 0>(num_elements, 0, 255, 100);

    // Types: `int8_t` and `int8_t`
    run_benchmark<int8_t, int8_t, 128>(num_elements, -128, 127);
    run_benchmark<int8_t, int8_t, 0>(num_elements, -128, 127, 128);
    run_benchmark<int8_t, int8_t, 100>(num_elements, -128, 127);
    run_benchmark<int8_t, int8_t, 0>(num_elements, -128, 127, 100);
}
