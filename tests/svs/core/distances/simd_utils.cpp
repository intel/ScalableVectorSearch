/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
#include <cstdint>
#include <type_traits>

#include "x86intrin.h"

#include "catch2/catch_test_macros.hpp"

#include "svs/core/distance/simd_utils.h"
#include "svs/lib/static.h"

CATCH_TEST_CASE("Masks", "[distance]") {
    // Make sure aliases are working
    CATCH_REQUIRE(std::is_same_v<svs::mask_repr_t<2>, uint8_t>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_repr_t<4>, uint8_t>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_repr_t<8>, uint8_t>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_repr_t<16>, uint16_t>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_repr_t<32>, uint32_t>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_repr_t<64>, uint64_t>);

    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_t<uint8_t>, __mmask8>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_t<uint16_t>, __mmask16>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_t<uint32_t>, __mmask32>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_t<uint64_t>, __mmask64>);

    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_from_length_t<2>, __mmask8>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_from_length_t<4>, __mmask8>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_from_length_t<8>, __mmask8>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_from_length_t<16>, __mmask16>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_from_length_t<32>, __mmask32>);
    CATCH_REQUIRE(std::is_same_v<svs::mask_intrinsic_from_length_t<64>, __mmask64>);

    // Test that mask generation is working correctly.
    CATCH_REQUIRE(svs::no_mask<2>() == 0xFF);
    CATCH_REQUIRE(svs::no_mask<4>() == 0xFF);
    CATCH_REQUIRE(svs::no_mask<8>() == 0xFF);
    CATCH_REQUIRE(svs::no_mask<16>() == 0xFFFF);
    CATCH_REQUIRE(svs::no_mask<32>() == 0xFFFFFFFF);
    CATCH_REQUIRE(svs::no_mask<64>() == 0xFFFFFFFFFFFFFFFF);

    // constexpr auto __temp = svs::create_mask<2>(svs::MaybeStatic<4>());
    // CATCH_REQUIRE(__temp == 0xFF);
    CATCH_REQUIRE(svs::create_mask<2>(svs::lib::MaybeStatic<5>()) == 0x1);
    CATCH_REQUIRE(svs::create_mask<32>(svs::lib::MaybeStatic<100>()) == 0xF);
    CATCH_REQUIRE(svs::create_mask<32>(svs::lib::MaybeStatic(100)) == 0xF);
    CATCH_REQUIRE(svs::create_mask<16>(svs::lib::MaybeStatic<100>()) == 0xF);
    CATCH_REQUIRE(svs::create_mask<16>(svs::lib::MaybeStatic<16>()) == 0xFFFF);
}
