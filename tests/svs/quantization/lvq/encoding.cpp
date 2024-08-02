/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// Header under test
#include "svs/quantization/lvq/compressed.h"

// Helper
#include "tests/svs/quantization/lvq/common.h"

// misc svs
#include "svs/lib/meta.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <cmath>
#include <vector>

namespace lvq = svs::quantization::lvq;

namespace {

// Test Signed/Unsigned endoding and decoding.
template <size_t Bits> void test_unsigned_encoder() {
    using Encoder = lvq::Encoding<lvq::Unsigned, Bits>;
    CATCH_REQUIRE((Encoder::min() == 0));
    CATCH_REQUIRE((Encoder::max() == (1 << Bits) - 1));
    CATCH_REQUIRE((Encoder::absmax() == (1 << Bits) - 1));
    for (auto i : {5, 20, 200, 1489}) {
        size_t bytes = Encoder::bytes(i);
        CATCH_REQUIRE(bytes == svs::lib::div_round_up(Bits * i, 8));
    }
    for (size_t i = 0; i <= Encoder::max(); ++i) {
        CATCH_REQUIRE((Encoder::encode(i) == i));
        CATCH_REQUIRE((Encoder::decode(i) == i));
        CATCH_REQUIRE((Encoder::check_bounds(i)));
    }

    size_t max_plus_one = static_cast<size_t>(Encoder::max()) + 1;
    CATCH_REQUIRE(!Encoder::check_bounds(max_plus_one));
}

template <size_t Bits> void test_signed_encoder() {
    using Encoder = lvq::Encoding<lvq::Signed, Bits>;
    CATCH_REQUIRE((Encoder::min() == -(1 << (Bits - 1))));
    CATCH_REQUIRE((Encoder::max() == (1 << (Bits - 1)) - 1));
    CATCH_REQUIRE((Encoder::absmax() == (1 << (Bits - 1))));

    for (auto i : {5, 20, 200, 1489}) {
        size_t bytes = Encoder::bytes(i);
        CATCH_REQUIRE(bytes == svs::lib::div_round_up(Bits * i, 8));
    }

    int64_t min = Encoder::min();
    int64_t max = Encoder::max();
    CATCH_REQUIRE(!Encoder::check_bounds(min - 1));
    CATCH_REQUIRE(!Encoder::check_bounds(max + 1));

    if (Bits == 8) {
        // No bias if storing 8-bits directly.
        for (int64_t i = Encoder::min(); i <= Encoder::max(); ++i) {
            auto encoded = Encoder::encode(i);
            CATCH_REQUIRE(std::is_same_v<uint8_t, decltype(encoded)>);
            CATCH_REQUIRE((std::bit_cast<int8_t>(encoded) == i));
            CATCH_REQUIRE((Encoder::decode(encoded) == i));
        }
    } else {
        // If the precision is fewer than 8-bits, then we need to apply a shift in order
        // to efficiently reapply the sign bits.
        int64_t shift = -Encoder::min();
        for (int64_t i = Encoder::min(); i <= Encoder::max(); ++i) {
            auto encoded = Encoder::encode(i);
            CATCH_REQUIRE(std::is_same_v<uint8_t, decltype(encoded)>);
            CATCH_REQUIRE((encoded == i + shift));
            CATCH_REQUIRE((Encoder::decode(encoded) == i));
        }
    }
}

// Test both the signed and unsigned encoders using `Bits` bits.
template <size_t Bits> void test_encode_decode() {
    test_unsigned_encoder<Bits>();
    test_signed_encoder<Bits>();
}

} // namespace

CATCH_TEST_CASE("LVQ Encoding", "[quantization][lvq][encoding]") {
    CATCH_SECTION("Compute Storage") {
        auto test_compute_storage = [](size_t nbits, size_t length, size_t expected) {
            CATCH_REQUIRE(lvq::compute_storage(nbits, length) == expected);
            CATCH_REQUIRE(lvq::compute_storage_extent(nbits, length) == expected);
            CATCH_REQUIRE(lvq::compute_storage_extent(nbits, svs::Dynamic) == svs::Dynamic);
        };

        test_compute_storage(2, 15, 4);
        test_compute_storage(2, 16, 4);
        test_compute_storage(2, 17, 5);

        test_compute_storage(3, 15, 6);
        test_compute_storage(3, 16, 6);
        test_compute_storage(3, 17, 7);

        test_compute_storage(4, 15, 8);
        test_compute_storage(4, 16, 8);
        test_compute_storage(4, 17, 9);

        test_compute_storage(5, 15, 10);
        test_compute_storage(5, 16, 10);
        test_compute_storage(5, 17, 11);

        test_compute_storage(6, 15, 12);
        test_compute_storage(6, 16, 12);
        test_compute_storage(6, 17, 13);

        test_compute_storage(7, 15, 14);
        test_compute_storage(7, 16, 14);
        test_compute_storage(7, 17, 15);

        test_compute_storage(8, 15, 15);
        test_compute_storage(8, 16, 16);
        test_compute_storage(8, 17, 17);
    }

    // Naming
    CATCH_SECTION("Naming") {
        CATCH_REQUIRE(lvq::Signed::name == "signed");
        CATCH_REQUIRE(lvq::Unsigned::name == "unsigned");
    }

    // Test encoders.
    CATCH_SECTION("Sub-byte encoding") {
        test_encode_decode<8>();
        test_encode_decode<7>();
        test_encode_decode<6>();
        test_encode_decode<5>();
        test_encode_decode<4>();
        test_encode_decode<3>();
    }
}
