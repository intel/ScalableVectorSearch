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

// Comprehensive verification tests for the compute ops refactoring in PR #196
// These tests verify correctness across all type combinations

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/core/distance/cosine.h"
#include "svs/lib/float16.h"

#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace {

// Test configuration
constexpr size_t NUM_ITERATIONS = 100;

// Random number generator
std::mt19937 gen(42);

// Reference L2
template<typename T1, typename T2>
float reference_l2(const std::vector<T1>& a, const std::vector<T2>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += diff * diff;
    }
    return sum;
}

// Reference IP
template<typename T1, typename T2>
float reference_ip(const std::vector<T1>& a, const std::vector<T2>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return sum;
}

template<typename T>
std::vector<T> random_vec(size_t n, T lo, T hi) {
    std::vector<T> result(n);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<int> dist(static_cast<int>(lo), static_cast<int>(hi));
        for (auto& v : result) {
            v = static_cast<T>(dist(gen));
        }
    } else {
        std::uniform_real_distribution<float> dist(static_cast<float>(lo), static_cast<float>(hi));
        for (auto& v : result) {
            v = static_cast<T>(dist(gen));
        }
    }
    return result;
}

std::vector<svs::Float16> random_fp16(size_t n) {
    auto floats = random_vec<float>(n, -1.0f, 1.0f);
    std::vector<svs::Float16> result;
    result.reserve(n);
    for (float f : floats) {
        result.push_back(svs::Float16(f));
    }
    return result;
}

} // anonymous namespace

CATCH_TEST_CASE("L2 Distance Verification - PR #196", "[distance][l2][verification][pr196]") {
    std::vector<size_t> sizes = {7, 8, 15, 16, 17, 32, 33, 64, 65, 127, 128, 256};
    
    for (size_t n : sizes) {
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
            // Float x Float
            {
                auto a = random_vec<float>(n, -1.0f, 1.0f);
                auto b = random_vec<float>(n, -1.0f, 1.0f);
                float expected = reference_l2(a, b);
                float actual = svs::distance::L2::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // Int8 x Int8
            {
                auto a = random_vec<int8_t>(n, -128, 127);
                auto b = random_vec<int8_t>(n, -128, 127);
                float expected = reference_l2(a, b);
                float actual = svs::distance::L2::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // UInt8 x UInt8
            {
                auto a = random_vec<uint8_t>(n, 0, 255);
                auto b = random_vec<uint8_t>(n, 0, 255);
                float expected = reference_l2(a, b);
                float actual = svs::distance::L2::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // Float x Int8
            {
                auto a = random_vec<float>(n, -1.0f, 1.0f);
                auto b = random_vec<int8_t>(n, -128, 127);
                float expected = reference_l2(a, b);
                float actual = svs::distance::L2::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // Float16 x Float16
            {
                auto a = random_fp16(n);
                auto b = random_fp16(n);
                float expected = reference_l2(a, b);
                float actual = svs::distance::L2::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // Float x Float16
            {
                auto a = random_vec<float>(n, -1.0f, 1.0f);
                auto b = random_fp16(n);
                float expected = reference_l2(a, b);
                float actual = svs::distance::L2::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
        }
    }
}

CATCH_TEST_CASE("Inner Product Verification - PR #196", "[distance][ip][verification][pr196]") {
    std::vector<size_t> sizes = {7, 8, 15, 16, 32, 64, 128, 256};
    
    for (size_t n : sizes) {
        for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
            // Float x Float
            {
                auto a = random_vec<float>(n, -1.0f, 1.0f);
                auto b = random_vec<float>(n, -1.0f, 1.0f);
                float expected = reference_ip(a, b);
                float actual = svs::distance::IP::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // Int8 x Int8
            {
                auto a = random_vec<int8_t>(n, -128, 127);
                auto b = random_vec<int8_t>(n, -128, 127);
                float expected = reference_ip(a, b);
                float actual = svs::distance::IP::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // UInt8 x UInt8
            {
                auto a = random_vec<uint8_t>(n, 0, 255);
                auto b = random_vec<uint8_t>(n, 0, 255);
                float expected = reference_ip(a, b);
                float actual = svs::distance::IP::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // Float x Int8
            {
                auto a = random_vec<float>(n, -1.0f, 1.0f);
                auto b = random_vec<int8_t>(n, -128, 127);
                float expected = reference_ip(a, b);
                float actual = svs::distance::IP::compute(a.data(), b.data(), n);
                // Use slightly larger margin due to different accumulation order in generic_simd_op
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // Float16 x Float16
            {
                auto a = random_fp16(n);
                auto b = random_fp16(n);
                float expected = reference_ip(a, b);
                float actual = svs::distance::IP::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
            
            // Float x Float16
            {
                auto a = random_vec<float>(n, -1.0f, 1.0f);
                auto b = random_fp16(n);
                float expected = reference_ip(a, b);
                float actual = svs::distance::IP::compute(a.data(), b.data(), n);
                CATCH_REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
            }
        }
    }
}
