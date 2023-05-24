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
        CATCH_REQUIRE(bytes >= svs::lib::div_round_up(Bits * i, 8));
        CATCH_REQUIRE(bytes % Bits == 0);
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
        CATCH_REQUIRE(bytes >= svs::lib::div_round_up(Bits * i, 8));
        CATCH_REQUIRE(bytes % Bits == 0);
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

///
/// Test both the signed and unsigned encoders using `Bits` bits.
///
template <size_t Bits> void test_encode_decode() {
    test_unsigned_encoder<Bits>();
    test_signed_encoder<Bits>();
}

///
/// Test
///
template <typename Sign, size_t Bits, size_t Extent>
void test_compressed(svs::lib::MaybeStatic<Extent> size = {}, size_t ntests = 5) {
    using compressed_type = lvq::MutableCompressedVector<Sign, Bits, Extent>;
    using const_compressed_type = lvq::CompressedVector<Sign, Bits, Extent>;

    using value_type = typename compressed_type::value_type;

    // Make sure the `value_type` is a suitable signed small integer.
    if constexpr (std::is_same_v<Sign, lvq::Signed>) {
        static_assert(std::is_same_v<value_type, int8_t>);
    } else {
        static_assert(std::is_same_v<Sign, lvq::Unsigned>);
        static_assert(std::is_same_v<value_type, uint8_t>);
    }

    constexpr size_t StorageExtent = compressed_type::storage_extent;
    size_t storage_bytes = lvq::compute_storage(Bits, size);

    // Allocate memory and construct a `CompressedVector` view over the data.
    std::vector<std::byte> v(storage_bytes);
    std::span<std::byte, StorageExtent> view{v.data(), v.size()};
    compressed_type cv{size, view};
    CATCH_REQUIRE((cv.size() == size));
    CATCH_REQUIRE(cv.extent == Extent);
    if constexpr (Extent != svs::Dynamic) {
        CATCH_REQUIRE(cv.size() == Extent);
        CATCH_REQUIRE(cv.storage_extent == storage_bytes);
        // Only need 8-bytes for the pointer
        CATCH_REQUIRE(sizeof(compressed_type) == 8);
    } else {
        CATCH_REQUIRE(cv.storage_extent == svs::Dynamic);
        // Only need 24 bytes - 8 for the actual length, 8 for the pointer, and 8 for the
        // span length.
        CATCH_REQUIRE(sizeof(compressed_type) == 24);
    }

    CATCH_REQUIRE((cv.data() == v.data()));

    // Reference stored values
    std::vector<value_type> reference(cv.size());

    // Random number generator for values that can live in the compressed dataset.
    auto g = test_q::create_generator<Sign, Bits>();

    // Populate the reference vector with random numbers that can fit in the compressed
    // vector view.
    //
    // Assign each value to the compressed vector and ensure that the correct values
    // come out the other end.
    for (size_t i = 0; i < ntests; ++i) {
        svs_test::populate(reference, g);
        for (size_t j = 0; j < cv.size(); ++j) {
            cv.set(reference.at(j), j);
            CATCH_REQUIRE((cv.get(j) == reference.at(j)));
        }

        // Implicit conversion to `const`
        const_compressed_type cv_const = cv;

        // Ensure that the `const` version works pretty much the same as the `non-const`
        // version.
        for (size_t j = 0; j < cv.size(); ++j) {
            CATCH_REQUIRE((cv.get(j) == reference.at(j)));
            CATCH_REQUIRE((cv_const.get(j) == reference.at(j)));
        }
    }
}

///
/// Test vectorized unpacking of compressed data.
/// @tparam VecWidth The vector width to use.
/// @tparam Sign Indicate whether the compression is signed or unsigned.
/// @tparam Bits The number of bits use per element in the compression.
/// @tparam Extent The compile-time dimensionality (Dynamiic if runtime).
/// @tparam IntType The intermediate integer type to store partiall unpacked results.
/// @tparam WideType The intermediate AVX register type to use for unpacing.
///
template <
    size_t VecWidth,
    typename Sign,
    size_t Bits,
    size_t Extent,
    typename IntType,
    typename WideType>
void test_unpacker(size_t mask, int64_t bias, svs::lib::MaybeStatic<Extent> size = {}) {
    using CV = lvq::MutableCompressedVector<Sign, Bits, Extent>;
    using U = lvq::detail::UnpackerBase<VecWidth, Sign, Bits>;
    using value_type = typename CV::value_type;
    using accum_type = eve::wide<float, eve::fixed<VecWidth>>;

    CATCH_REQUIRE((U::simd_width == VecWidth));
    CATCH_REQUIRE(std::is_same_v<typename U::int_type, IntType>);
    CATCH_REQUIRE(std::is_same_v<typename U::int_wide_type, WideType>);
    CATCH_REQUIRE(std::is_same_v<typename U::accum_type, accum_type>);
    CATCH_REQUIRE((U::bits == Bits));

    eve::wide<IntType, eve::fixed<8>> expected_shifts{
        0 * Bits,
        1 * Bits,
        2 * Bits,
        3 * Bits,
        4 * Bits,
        5 * Bits,
        6 * Bits,
        7 * Bits,
    };
    CATCH_REQUIRE(eve::all(U::shifts_x8 == expected_shifts));
    CATCH_REQUIRE((U::mask == svs::lib::narrow<int>(mask)));
    CATCH_REQUIRE((U::bias == svs::lib::narrow<int>(bias)));

    // Test the vectorized unpacking.
    constexpr size_t StorageExtent = CV::storage_extent;
    size_t bytes = lvq::compute_storage(Bits, size);
    std::vector<std::byte> v(bytes);
    auto cv = CV(size, std::span<std::byte, StorageExtent>{v.data(), v.size()});
    auto reference = std::vector<value_type>(cv.size());
    auto dst = std::vector<value_type>(cv.size());
    auto g = test_q::create_generator<Sign, Bits>();

    const size_t ntests = 10;
    for (size_t i = 0; i < ntests; ++i) {
        svs_test::populate(reference, g);
        for (size_t j = 0; j < cv.size(); ++j) {
            cv.set(reference.at(j), j);
        }

        lvq::unpack(dst, cv.as_const());
        CATCH_REQUIRE(std::equal(dst.begin(), dst.end(), reference.begin()));
    }
}

// Helper alias to cut down on visual clutter.
template <size_t N> using Val = svs::meta::Val<N>;
} // namespace

CATCH_TEST_CASE("Quantization Utilities", "[quantization][core_quantization]") {
    // Test bitmask generation.
    CATCH_SECTION("Mask") {
        // Setting single bits.
        for (uint8_t i = 0; i < 7; ++i) {
            auto m = lvq::detail::bitmask(uint8_t{i}, uint8_t{i});
            CATCH_REQUIRE(m == 1 << i);
        }

        // Several test patterns.
        CATCH_REQUIRE(lvq::detail::bitmask(uint8_t{0}, uint8_t{7}) == 0xff);
        CATCH_REQUIRE(lvq::detail::bitmask(uint8_t{0}, uint8_t{3}) == 0x0f);
        CATCH_REQUIRE(lvq::detail::bitmask(uint8_t{4}, uint8_t{7}) == 0xf0);
    }

    // Test the `IndexRange` utility struct.
    // See the documentation for that type to understand the logic.
    CATCH_SECTION("Index Range") {
        using IR = lvq::detail::IndexRange;
        CATCH_SECTION("8 Bits") {
            CATCH_REQUIRE(IR{Val<8>{}, 0} == IR{0, 0, 0, 7});
            CATCH_REQUIRE(IR{Val<8>{}, 1} == IR{1, 1, 0, 7});
            CATCH_REQUIRE(IR{Val<8>{}, 2} == IR{2, 2, 0, 7});
            CATCH_REQUIRE(IR{Val<8>{}, 3} == IR{3, 3, 0, 7});
            CATCH_REQUIRE(IR{Val<8>{}, 4} == IR{4, 4, 0, 7});
            CATCH_REQUIRE(IR{Val<8>{}, 5} == IR{5, 5, 0, 7});
            CATCH_REQUIRE(IR{Val<8>{}, 6} == IR{6, 6, 0, 7});
            CATCH_REQUIRE(IR{Val<8>{}, 7} == IR{7, 7, 0, 7});
        }

        CATCH_SECTION("7 Bits") {
            CATCH_REQUIRE(IR{Val<7>{}, 0} == IR{0, 0, 0, 6});
            CATCH_REQUIRE(IR{Val<7>{}, 1} == IR{0, 1, 7, 13});
            CATCH_REQUIRE(IR{Val<7>{}, 2} == IR{1, 2, 6, 12});
            CATCH_REQUIRE(IR{Val<7>{}, 3} == IR{2, 3, 5, 11});
            CATCH_REQUIRE(IR{Val<7>{}, 4} == IR{3, 4, 4, 10});
            CATCH_REQUIRE(IR{Val<7>{}, 5} == IR{4, 5, 3, 9});
            CATCH_REQUIRE(IR{Val<7>{}, 6} == IR{5, 6, 2, 8});
            CATCH_REQUIRE(IR{Val<7>{}, 7} == IR{6, 6, 1, 7});
        }

        CATCH_SECTION("6 Bits") {
            CATCH_REQUIRE(IR{Val<6>{}, 0} == IR{0, 0, 0, 5});
            CATCH_REQUIRE(IR{Val<6>{}, 1} == IR{0, 1, 6, 11});
            CATCH_REQUIRE(IR{Val<6>{}, 2} == IR{1, 2, 4, 9});
            CATCH_REQUIRE(IR{Val<6>{}, 3} == IR{2, 2, 2, 7});
            CATCH_REQUIRE(IR{Val<6>{}, 4} == IR{3, 3, 0, 5});
            CATCH_REQUIRE(IR{Val<6>{}, 5} == IR{3, 4, 6, 11});
            CATCH_REQUIRE(IR{Val<6>{}, 6} == IR{4, 5, 4, 9});
            CATCH_REQUIRE(IR{Val<6>{}, 7} == IR{5, 5, 2, 7});
        }
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

    // Compressed
    CATCH_SECTION("Basic Behavior") {
        const size_t N = 35;
        constexpr auto static_n = svs::lib::MaybeStatic(N);
        test_compressed<lvq::Signed, 8, N>();
        test_compressed<lvq::Signed, 8, svs::Dynamic>(static_n);
        test_compressed<lvq::Unsigned, 8, N>();
        test_compressed<lvq::Unsigned, 8, svs::Dynamic>(static_n);

        test_compressed<lvq::Signed, 7, N>();
        test_compressed<lvq::Signed, 7, svs::Dynamic>(static_n);
        test_compressed<lvq::Unsigned, 7, N>();
        test_compressed<lvq::Unsigned, 7, svs::Dynamic>(static_n);

        test_compressed<lvq::Signed, 6, N>();
        test_compressed<lvq::Signed, 6, svs::Dynamic>(static_n);
        test_compressed<lvq::Unsigned, 6, N>();
        test_compressed<lvq::Unsigned, 6, svs::Dynamic>(static_n);

        test_compressed<lvq::Signed, 5, N>();
        test_compressed<lvq::Signed, 5, svs::Dynamic>(static_n);
        test_compressed<lvq::Unsigned, 5, N>();
        test_compressed<lvq::Unsigned, 5, svs::Dynamic>(static_n);

        test_compressed<lvq::Signed, 4, N>();
        test_compressed<lvq::Signed, 4, svs::Dynamic>(static_n);
        test_compressed<lvq::Unsigned, 4, N>();
        test_compressed<lvq::Unsigned, 4, svs::Dynamic>(static_n);

        test_compressed<lvq::Signed, 3, N>();
        test_compressed<lvq::Signed, 3, svs::Dynamic>(static_n);
        test_compressed<lvq::Unsigned, 3, N>();
        test_compressed<lvq::Unsigned, 3, svs::Dynamic>(static_n);
    }

    CATCH_SECTION("AVX Unpacking Utilities") {
        CATCH_SECTION("Pick SIMD Width") {
            // TODO: Detection for non-AVX 512 machines.
            CATCH_REQUIRE(lvq::pick_simd_width<8>() == 16);
            CATCH_REQUIRE(lvq::pick_simd_width<7>() == 8);
            CATCH_REQUIRE(lvq::pick_simd_width<6>() == 8);
            CATCH_REQUIRE(lvq::pick_simd_width<5>() == 8);
            CATCH_REQUIRE(lvq::pick_simd_width<4>() == 16);
            CATCH_REQUIRE(lvq::pick_simd_width<3>() == 8);

            CATCH_REQUIRE(lvq::pick_simd_width<4, 4>() == 16);
            CATCH_REQUIRE(lvq::pick_simd_width<4, 5>() == 8);
            CATCH_REQUIRE(lvq::pick_simd_width<5, 4>() == 8);
            CATCH_REQUIRE(lvq::pick_simd_width<5, 5>() == 8);

            CATCH_REQUIRE(lvq::pick_simd_width<4, 8>() == 16);
            CATCH_REQUIRE(lvq::pick_simd_width<8, 4>() == 16);
        }
    }

    CATCH_SECTION("Unpacking") {
        constexpr size_t Dynamic = svs::Dynamic;
        using i32 = int32_t;
        using i64 = int64_t;

        using wide_64x8 = eve::wide<i64, eve::fixed<8>>;
        using wide_32x8 = eve::wide<i32, eve::fixed<8>>;
        using wide_32x16 = eve::wide<i32, eve::fixed<16>>;

        constexpr size_t N = 37;
        constexpr auto static_n = svs::lib::MaybeStatic(N);

        // 3-bit
        test_unpacker<8, lvq::Signed, 3, N, i32, wide_32x8>(0x7, -4);
        test_unpacker<8, lvq::Unsigned, 3, N, i32, wide_32x8>(0x7, 0);

        test_unpacker<8, lvq::Signed, 3, Dynamic, i32, wide_32x8>(0x7, -4, static_n);
        test_unpacker<8, lvq::Unsigned, 3, Dynamic, i32, wide_32x8>(0x7, 0, static_n);

        // 4-bit
        test_unpacker<8, lvq::Signed, 4, N, i32, wide_32x8>(0xf, -8);
        test_unpacker<8, lvq::Unsigned, 4, N, i32, wide_32x8>(0xf, 0);
        test_unpacker<16, lvq::Signed, 4, N, i32, wide_32x16>(0xf, -8);
        test_unpacker<16, lvq::Unsigned, 4, N, i32, wide_32x16>(0xf, 0);

        test_unpacker<8, lvq::Signed, 4, Dynamic, i32, wide_32x8>(0xf, -8, static_n);
        test_unpacker<8, lvq::Unsigned, 4, Dynamic, i32, wide_32x8>(0xf, 0, static_n);
        test_unpacker<16, lvq::Signed, 4, Dynamic, i32, wide_32x16>(0xf, -8, static_n);
        test_unpacker<16, lvq::Unsigned, 4, Dynamic, i32, wide_32x16>(0xf, 0, static_n);

        // 5-bit
        test_unpacker<8, lvq::Signed, 5, N, i64, wide_64x8>(0x1f, -16);
        test_unpacker<8, lvq::Unsigned, 5, N, i64, wide_64x8>(0x1f, 0);

        test_unpacker<8, lvq::Signed, 5, Dynamic, i64, wide_64x8>(0x1f, -16, static_n);
        test_unpacker<8, lvq::Unsigned, 5, Dynamic, i64, wide_64x8>(0x1f, 0, static_n);

        // 6-bit
        test_unpacker<8, lvq::Signed, 6, N, i64, wide_64x8>(0x3f, -32);
        test_unpacker<8, lvq::Unsigned, 6, N, i64, wide_64x8>(0x3f, 0);

        test_unpacker<8, lvq::Signed, 6, Dynamic, i64, wide_64x8>(0x3f, -32, static_n);
        test_unpacker<8, lvq::Unsigned, 6, Dynamic, i64, wide_64x8>(0x3f, 0, static_n);

        // 7-bit
        test_unpacker<8, lvq::Signed, 7, N, i64, wide_64x8>(0x7f, -64);
        test_unpacker<8, lvq::Unsigned, 7, N, i64, wide_64x8>(0x7f, 0);

        test_unpacker<8, lvq::Signed, 7, Dynamic, i64, wide_64x8>(0x7f, -64, static_n);
        test_unpacker<8, lvq::Unsigned, 7, Dynamic, i64, wide_64x8>(0x7f, 0, static_n);

        // 8-bit
        test_unpacker<8, lvq::Signed, 8, N, i32, wide_32x8>(0xff, -128);
        test_unpacker<8, lvq::Unsigned, 8, N, i32, wide_32x8>(0xff, 0);
        test_unpacker<16, lvq::Signed, 8, N, i32, wide_32x16>(0xff, -128);
        test_unpacker<16, lvq::Unsigned, 8, N, i32, wide_32x16>(0xff, 0);

        test_unpacker<8, lvq::Signed, 8, Dynamic, i32, wide_32x8>(0xff, -128, static_n);
        test_unpacker<8, lvq::Unsigned, 8, Dynamic, i32, wide_32x8>(0xff, 0, static_n);
        test_unpacker<16, lvq::Signed, 8, Dynamic, i32, wide_32x16>(0xff, -128, static_n);
        test_unpacker<16, lvq::Unsigned, 8, Dynamic, i32, wide_32x16>(0xff, 0, static_n);
    }
}