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

#pragma once

// svs
#include "svs/concepts/distance.h"
#include "svs/core/distance/conversion.h"
#include "svs/core/distance/distance_core.h"
#include "svs/lib/float16.h"
#include "svs/lib/threads/threadlocal.h"

// tests
#include "tests/utils/generators.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace svs_test {

template <typename T>
using aligned_vector = std::vector<T, svs::threads::CacheAlignedAllocator<T>>;

///
/// @brief Convert ``values`` to ``float`` the way the distance kernels at the currently
///     dispatched instruction-set level convert ``Float16`` operands.
///
/// The Intel(R) AVX kernels convert in hardware, which keeps subnormal halves exact. The
/// generic kernels use ``svs::Float16``'s software conversion, which flushes them to zero.
/// A ``float`` copy built any other way does not match the kernels it is compared against.
///
inline aligned_vector<float> convert_like_kernels(std::span<const svs::Float16> values) {
    auto converted = aligned_vector<float>(values.size());
    svs::distance::Convert::apply(values.data(), converted.data(), values.size());
    return converted;
}

///
/// @brief Overwrite a few elements of ``values`` with subnormal halves of both signs.
///
/// Uniformly drawn values are subnormal too rarely to exercise the one input class on
/// which the hardware and software conversions disagree.
///
inline void insert_subnormals(std::vector<svs::Float16>& values) {
    // Smallest, a middle, and the largest subnormal; positive and negative.
    constexpr auto subnormals =
        std::to_array<uint16_t>({0x0001, 0x8001, 0x0200, 0x8200, 0x03FF, 0x83FF});
    for (size_t i = 0; i < subnormals.size() && i < values.size(); ++i) {
        // Spread them out so they land in different SIMD lanes and loop iterations.
        values[(i * 37) % values.size()] = svs::Float16::from_raw(subnormals[i]);
    }
}

namespace detail {

template <typename Distance, size_t Extent>
void require_preconverted_operands_match(size_t dims, size_t num_trials) {
    CATCH_INFO("Extent: " << Extent << ", dimensions: " << dims);
    auto generator = make_generator<svs::Float16>(-1, 1);
    auto a = std::vector<svs::Float16>{};
    auto b = std::vector<svs::Float16>{};

    for (size_t trial = 0; trial < num_trials; ++trial) {
        populate(a, generator, dims);
        populate(b, generator, dims);
        insert_subnormals(a);
        insert_subnormals(b);
        auto a_float = convert_like_kernels(a);
        auto b_float = convert_like_kernels(b);

        auto a_half = std::span<const svs::Float16, Extent>(a.data(), dims);
        auto b_half = std::span<const svs::Float16, Extent>(b.data(), dims);
        auto a_full = std::span<const float, Extent>(a_float.data(), dims);
        auto b_full = std::span<const float, Extent>(b_float.data(), dims);

        // At one instruction-set level, every kernel below shares one SIMD operation and
        // `Convert` yields the register values the kernel's own conversion would, so the
        // results must be bit-identical.
        //
        // One functor, fixed once, serves all four calls. `fix_argument` is not a kernel:
        // for cosine, given `Float16` it computes the left-hand norm through `Float16`'s
        // software conversion, which flushes subnormals to zero; given converted `float`,
        // it uses the converted values, which keep subnormals exact at the Intel(R) AVX
        // levels. Fixing it per call would compare those two norms, not the kernels.
        auto f = Distance{};
        svs::distance::maybe_fix_argument(f, a_full);
        //
        // Compared as bit patterns: a vector whose every element flushes to zero has an
        // undefined cosine, and NaN does not compare equal to itself.
        auto bits = [&](auto left, auto right) {
            return std::bit_cast<uint32_t>(svs::distance::compute(f, left, right));
        };
        auto expected = bits(a_half, b_half);
        CATCH_REQUIRE(bits(a_full, b_half) == expected);
        CATCH_REQUIRE(bits(a_half, b_full) == expected);
        CATCH_REQUIRE(bits(a_full, b_full) == expected);
    }
}

// Each static extent in `supported_dim_list`, the list the kernels are specialized for.
template <typename Distance, size_t... Is>
void require_static_extents_match(size_t num_trials, std::index_sequence<Is...>) {
    (
        [&]() {
            constexpr size_t extent = svs::distance::supported_dim_list[Is];
            if constexpr (extent != svs::Dynamic) {
                require_preconverted_operands_match<Distance, extent>(extent, num_trials);
            }
        }(),
        ...
    );
}

} // namespace detail

///
/// @brief Require that converting either or both ``Float16`` arguments of
///     ``distance::compute`` to ``float`` ahead of time leaves the result bit-identical.
///
/// @tparam Distance The distance functor under test.
///
/// Runs at every supported instruction-set level, for each statically sized dimension the
/// kernels specialize and for dynamically sized arguments of several lengths, including
/// ragged tails and dimensionalities in the thousands.
///
template <typename Distance> void require_preconverted_operands_match() {
    constexpr size_t num_trials = 20;
    for_each_supported_isa([&](std::string_view isa) {
        CATCH_INFO("ISA: " << isa);
        detail::require_static_extents_match<Distance>(
            num_trials, std::make_index_sequence<svs::distance::supported_dim_list.size()>{}
        );
        for (size_t dims : {1, 17, 100, 128, 1000, 4096, 4099}) {
            detail::require_preconverted_operands_match<Distance, svs::Dynamic>(
                dims, num_trials
            );
        }
    });
}

} // namespace svs_test
