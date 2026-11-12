/*
 * Copyright 2026 Intel Corporation
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

// header under test
#include "svs/core/distance/conversion.h"

// svs
#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/lib/float16.h"
#include "svs/lib/threads/threadlocal.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/generators.h"
#include "tests/utils/preconversion.h"
#include "tests/utils/utils.h"

// stl
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

using svs::Float16;
namespace distance = svs::distance;

using svs_test::aligned_vector;

// Cover the full-width loops and ragged tails of both the 8-wide (AVX2) and 16-wide
// (AVX-512) converters, the empty input, and dimensionalities into the thousands.
constexpr auto test_lengths = std::to_array<size_t>(
    {0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 100, 128, 768, 1000, 4096, 4099}
);

// Elements after the converted range that every converter must leave untouched.
constexpr size_t num_sentinels = 32;
constexpr float sentinel = -12345.0F;

bool is_aligned(const void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % distance::conversion_alignment == 0;
}

std::vector<Float16> random_float16(size_t length) {
    auto values = std::vector<Float16>{};
    auto generator = svs_test::make_generator<Float16>(-10, 10);
    svs_test::populate(values, generator, length);
    svs_test::insert_subnormals(values);
    return values;
}

// IEEE 754 binary16 to binary32, exact for every finite input including subnormals. This
// is what the hardware conversion in the Intel(R) AVX kernels computes.
float exact_float16_to_float(Float16 x) {
    const uint16_t raw = x.raw();
    const int exponent = (raw >> 10) & 0x1F;
    const int mantissa = raw & 0x3FF;
    const float sign = (raw & 0x8000) != 0 ? -1.0F : 1.0F;
    if (exponent == 0) {
        return sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    return sign * std::ldexp(static_cast<float>(mantissa + 0x400), exponent - 25);
}

// The value `Convert` must produce at the dispatched level: the same conversion the
// distance kernels at that level apply, so that pre-converted and unconverted operands
// agree. The generic kernels use `Float16`'s software conversion, which flushes
// subnormals to zero.
float expected_conversion(std::string_view isa, Float16 x) {
    return isa == "generic" ? static_cast<float>(x) : exact_float16_to_float(x);
}

} // namespace

CATCH_TEST_CASE("Element Conversion", "[distance][conversion]") {
    CATCH_SECTION("Float16 to float") {
        svs_test::for_each_supported_isa([](std::string_view isa) {
            for (size_t length : test_lengths) {
                CATCH_INFO("ISA: " << isa << ", length: " << length);
                // One extra leading element lets the source start one element past an
                // allocation boundary: `Convert` must not require an aligned source.
                auto storage = random_float16(length + 1);
                const Float16* src = storage.data() + 1;

                auto dst = aligned_vector<float>(length + num_sentinels, sentinel);
                CATCH_REQUIRE(is_aligned(dst.data()));
                distance::Convert::apply(src, dst.data(), length);

                for (size_t i = 0; i < length; ++i) {
                    CATCH_REQUIRE(dst[i] == expected_conversion(isa, src[i]));
                }
                for (size_t i = length; i < dst.size(); ++i) {
                    CATCH_REQUIRE(dst[i] == sentinel);
                }
            }
        });
    }
}

CATCH_TEST_CASE("Distance Argument", "[distance][conversion]") {
    // `Float16` is converted for the distances whose kernels convert it on every step,
    // keeping the extent.
    static_assert(std::is_same_v<
                  distance::distance_argument_t<
                      distance::DistanceIP,
                      std::span<const Float16, 128>>,
                  std::span<const float, 128>>);
    static_assert(std::is_same_v<
                  distance::
                      distance_argument_t<distance::DistanceIP, std::span<const Float16>>,
                  std::span<const float>>);
    static_assert(std::is_same_v<
                  distance::distance_argument_t<
                      distance::DistanceCosineSimilarity,
                      std::span<const Float16>>,
                  std::span<const float>>);

    // Not converted: `float` already is the working type, and `int8_t` would lose the
    // VNNI kernel. L2 is not covered by this change.
    static_assert(std::is_same_v<
                  distance::
                      distance_argument_t<distance::DistanceIP, std::span<const float>>,
                  std::span<const float>>);
    static_assert(std::is_same_v<
                  distance::
                      distance_argument_t<distance::DistanceIP, std::span<const int8_t>>,
                  std::span<const int8_t>>);
    static_assert(std::is_same_v<
                  distance::
                      distance_argument_t<distance::DistanceL2, std::span<const Float16>>,
                  std::span<const Float16>>);
}

CATCH_TEST_CASE("Preconverted Argument", "[distance][conversion]") {
    CATCH_SECTION("Identity") {
        auto values = std::vector<float>(100, 1.0F);
        auto span = std::span<const float>(values);
        auto storage =
            distance::PreconvertedArgument<distance::DistanceIP, std::span<const float>>(
                values.size()
            );

        // An lvalue comes back as a reference to the same object, a prvalue by value.
        static_assert(std::is_same_v<decltype(storage(span)), std::span<const float>&>);
        static_assert(std::is_same_v<
                      decltype(storage(std::span<const float>(values))),
                      std::span<const float>>);
        CATCH_REQUIRE(storage(span).data() == values.data());

        auto int8_values = std::vector<int8_t>(100, 1);
        auto int8_storage =
            distance::PreconvertedArgument<distance::DistanceIP, std::span<const int8_t>>(
                int8_values.size()
            );
        auto int8_span = std::span<const int8_t>(int8_values);
        CATCH_REQUIRE(int8_storage(int8_span).data() == int8_values.data());
    }

    CATCH_SECTION("Converting, dynamic extent") {
        constexpr size_t dims = 100;
        using storage_type =
            distance::PreconvertedArgument<distance::DistanceIP, std::span<const Float16>>;
        static_assert(std::is_same_v<storage_type::converted_type, std::span<const float>>);

        auto storage = storage_type(dims);
        auto first = random_float16(dims);
        auto second = random_float16(dims);
        auto expected_first = svs_test::convert_like_kernels(first);
        auto expected_second = svs_test::convert_like_kernels(second);

        auto converted_first = storage(std::span<const Float16>(first));
        CATCH_REQUIRE(converted_first.size() == dims);
        CATCH_REQUIRE(is_aligned(converted_first.data()));
        for (size_t i = 0; i < dims; ++i) {
            CATCH_REQUIRE(converted_first[i] == expected_first[i]);
        }

        // The storage is reused, not reallocated, for the next vector.
        auto converted_second = storage(std::span<const Float16>(second));
        CATCH_REQUIRE(converted_second.data() == converted_first.data());
        for (size_t i = 0; i < dims; ++i) {
            CATCH_REQUIRE(converted_second[i] == expected_second[i]);
        }

        // Copies, as made by `SequentialTLS` for each thread, own separate storage.
        auto copy = storage;
        auto converted_copy = copy(std::span<const Float16>(first));
        CATCH_REQUIRE(converted_copy.data() != converted_second.data());
        CATCH_REQUIRE(is_aligned(converted_copy.data()));
        for (size_t i = 0; i < dims; ++i) {
            CATCH_REQUIRE(converted_copy[i] == expected_first[i]);
            CATCH_REQUIRE(converted_second[i] == expected_second[i]);
        }
    }

    CATCH_SECTION("Converting, static extent") {
        constexpr size_t dims = 128;
        using storage_type = distance::PreconvertedArgument<
            distance::DistanceCosineSimilarity,
            std::span<const Float16, dims>>;
        static_assert(std::is_same_v<
                      storage_type::converted_type,
                      std::span<const float, dims>>);

        auto storage = storage_type(dims);
        auto values = random_float16(dims);
        auto expected = svs_test::convert_like_kernels(values);
        auto converted = storage(std::span<const Float16, dims>(values.data(), dims));
        CATCH_REQUIRE(is_aligned(converted.data()));
        for (size_t i = 0; i < dims; ++i) {
            CATCH_REQUIRE(converted[i] == expected[i]);
        }
    }
}
