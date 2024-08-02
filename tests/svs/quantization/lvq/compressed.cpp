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

// Overloaded method to return multiple different strategy types for logical comparison.
template <typename Sign, size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy>
auto get_other_strategies(lvq::CompressedVector<Sign, Bits, Extent, Strategy> SVS_UNUSED(x)
) {
    return svs::lib::Types<Strategy>();
}

template <typename Sign, size_t Extent>
auto get_other_strategies(
    lvq::CompressedVector<Sign, 4, Extent, lvq::Sequential> SVS_UNUSED(x)
) {
    return svs::lib::Types<lvq::Sequential, lvq::Turbo<16, 8>>();
}

/// For equality testing, we want to ensure that
template <typename Sign, size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy>
void test_logical_equality(lvq::CompressedVector<Sign, Bits, Extent, Strategy> x) {
    // Get a collection of other strategies to use.
    auto strategies = get_other_strategies(x);
    // Storage for the other vector.
    auto storage = lvq::CVStorage();
    svs::lib::for_each_type(
        strategies,
        [&]<typename T>(svs::lib::Type<T> SVS_UNUSED(type)) {
            auto other = storage.template view<Sign, Bits, Extent, T>(
                svs::lib::MaybeStatic<Extent>(x.size())
            );

            // Copy each component to establish equality.
            for (size_t i = 0; i < x.size(); ++i) {
                other.set(x.get(i), i);
            }
            CATCH_REQUIRE(lvq::logically_equal(other, x));
            CATCH_REQUIRE(lvq::logically_equal(x, other));

            // Change each component to a different value.
            // Make sure that when we change that encoding, the vectors no longer compare as
            // equal.
            for (size_t i = 0; i < x.size(); ++i) {
                auto v = other.get(i);
                other.set(v == 0 ? 1 : 0, i);
                CATCH_REQUIRE(!lvq::logically_equal(other, x));
                CATCH_REQUIRE(!lvq::logically_equal(x, other));

                // Reset to its original value.
                other.set(v, i);
            }

            // Make sure that after all this modification, the vectors still evaluate equal.
            CATCH_REQUIRE(lvq::logically_equal(other, x));
            CATCH_REQUIRE(lvq::logically_equal(x, other));
        }
    );
}

template <typename Sign, size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy>
void test_compressed_constructors(svs::lib::MaybeStatic<Extent> size = {}) {
    using Compressed = lvq::CompressedVector<Sign, Bits, Extent, Strategy>;
    using MutCompressed = lvq::MutableCompressedVector<Sign, Bits, Extent, Strategy>;

    // Oversize the underlying storage to test shrinking logic.
    size_t storage_bytes = Strategy::compute_bytes(Bits, size);

    constexpr size_t StorageExtent =
        (Extent == svs::Dynamic) ? svs::Dynamic : Strategy::compute_bytes(Bits, Extent);

    constexpr size_t OversizedExtent =
        StorageExtent == svs::Dynamic ? svs::Dynamic : StorageExtent + 10;

    auto storage = std::vector<std::byte>(storage_bytes + 10);

    auto span = std::span<std::byte, StorageExtent>(storage.begin(), storage_bytes);
    auto const_span =
        std::span<const std::byte, StorageExtent>(storage.begin(), storage_bytes);

    auto test_cv = [&](const auto& cv) {
        using CVT = std::remove_cvref_t<decltype(cv)>;
        CATCH_REQUIRE(cv.size() == size);
        CATCH_REQUIRE(cv.size_bytes() == storage_bytes);
        CATCH_REQUIRE(cv.compute_bytes(size) == storage_bytes);
        CATCH_REQUIRE(CVT::storage_extent == StorageExtent);

        if constexpr (Extent != svs::Dynamic) {
            CATCH_REQUIRE(CVT::compute_bytes() == storage_bytes);
        }
    };

    // Construct from just a span is allowed if the length is known at compile time.
    if constexpr (Extent != svs::Dynamic) {
        auto cv = MutCompressed{span};
        test_cv(cv);
        test_cv(cv.as_const());
        auto cv_const = Compressed{span};
        test_cv(cv_const);
    }

    // Test the normal constructors.
    {
        auto cv = MutCompressed{size, span};
        test_cv(cv);
        test_cv(cv.as_const());
        auto cv_const = Compressed{size, span};
        test_cv(cv_const);
        cv_const = Compressed{size, const_span};

        if constexpr (Extent == svs::Dynamic) {
            auto span_short = span.subspan(0, 3);
            CATCH_REQUIRE_THROWS_AS(MutCompressed(size, span_short), svs::ANNException);
            CATCH_REQUIRE_THROWS_AS(Compressed(size, span_short), svs::ANNException);
        }
    }

    // Shrinking constructor.
    {
        auto span_oversized = std::span<std::byte, OversizedExtent>(storage);
        auto const_span_oversized = std::span<const std::byte, OversizedExtent>(storage);

        auto cv = MutCompressed{lvq::AllowShrinkingTag(), size, span_oversized};
        test_cv(cv);
        test_cv(cv.as_const());

        auto cv_const = Compressed{lvq::AllowShrinkingTag(), size, const_span_oversized};
        test_cv(cv_const);
        cv_const = Compressed{lvq::AllowShrinkingTag(), size, span_oversized};
        test_cv(cv_const);

        // If we're constructing a static-length CV, try building from a dynamic span.
        if constexpr (Extent == svs::Dynamic) {
            auto span_oversized_d = std::span<std::byte, svs::Dynamic>(storage);
            auto const_span_oversized_d = std::span<const std::byte, svs::Dynamic>(storage);
            cv = MutCompressed{lvq::AllowShrinkingTag(), size, span_oversized_d};
            test_cv(cv);
            cv_const = Compressed{lvq::AllowShrinkingTag(), size, const_span_oversized_d};
            test_cv(cv_const);
            cv_const = Compressed{lvq::AllowShrinkingTag(), size, span_oversized_d};
            test_cv(cv_const);
        }
    }
}

template <
    typename Sign,
    size_t Bits,
    size_t Extent,
    lvq::LVQPackingStrategy Strategy = lvq::Sequential>
void test_compressed(svs::lib::MaybeStatic<Extent> size = {}, size_t ntests = 5) {
    test_compressed_constructors<Sign, Bits, Extent, Strategy>(size);

    using MutCompressed = lvq::MutableCompressedVector<Sign, Bits, Extent, Strategy>;
    using Compressed = lvq::CompressedVector<Sign, Bits, Extent, Strategy>;
    using value_type = typename MutCompressed::value_type;

    // Make sure the `value_type` is a suitable signed small integer.
    if constexpr (std::is_same_v<Sign, lvq::Signed>) {
        static_assert(std::is_same_v<value_type, int8_t>);
    } else {
        static_assert(std::is_same_v<Sign, lvq::Unsigned>);
        static_assert(std::is_same_v<value_type, uint8_t>);
    }

    size_t storage_bytes = Strategy::compute_bytes(Bits, size);

    // Allocate memory and construct a `CompressedVector` view over the data.
    std::vector<std::byte> v(storage_bytes);
    MutCompressed cv{size, typename MutCompressed::span_type{v}};

    CATCH_REQUIRE((cv.size() == size));
    CATCH_REQUIRE(cv.extent == Extent);
    if constexpr (Extent != svs::Dynamic) {
        CATCH_REQUIRE(cv.size() == Extent);
        CATCH_REQUIRE(cv.storage_extent == storage_bytes);
        // Only need 8-bytes for the pointer
        CATCH_STATIC_REQUIRE(sizeof(MutCompressed) == 8);
    } else {
        CATCH_REQUIRE(cv.storage_extent == svs::Dynamic);
        // Need 16 bytes - 8 for the pointer and 8 for the length.
        CATCH_STATIC_REQUIRE(sizeof(MutCompressed) == 16);
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
    auto cv_size = cv.size();
    for (size_t i = 0; i < ntests; ++i) {
        svs_test::populate(reference, g);
        for (size_t j = 0; j < cv_size; ++j) {
            cv.set(reference.at(j), j);
            CATCH_REQUIRE((cv.get(j) == reference.at(j)));
        }

        // Implicit conversion to `const`
        Compressed cv_const = cv;

        // Ensure that the `const` version works pretty much the same as the `non-const`
        // version.
        for (size_t j = 0; j < cv_size; ++j) {
            CATCH_REQUIRE((cv.get(j) == reference.at(j)));
            CATCH_REQUIRE((cv_const.get(j) == reference.at(j)));
        }

        // Test copying.
        auto other_storage = lvq::CVStorage();
        auto other = other_storage.template view<Sign, Bits, Extent, Strategy>(size);
        CATCH_REQUIRE(other.size() == cv_size);

        // Rely on std::vector initializing to 0.
        for (size_t j = 0; j < cv_size; ++j) {
            CATCH_REQUIRE(other.get(j) == Compressed::decode(0));
        }
        CATCH_REQUIRE(!lvq::logically_equal(cv, other));
        CATCH_REQUIRE(!lvq::logically_equal(other, cv));

        other.copy_from(cv);
        for (size_t j = 0; j < cv_size; ++j) {
            CATCH_REQUIRE(other.get(j) == reference.at(j));
        }
        CATCH_REQUIRE(lvq::logically_equal(cv, other));
        CATCH_REQUIRE(lvq::logically_equal(other, cv));

        // Copy from a vector.
        svs_test::populate(reference, g);
        other.copy_from(reference);
        for (size_t j = 0; j < cv_size; ++j) {
            CATCH_REQUIRE(other.get(j) == reference.at(j));
        }

        CATCH_REQUIRE(!lvq::logically_equal(cv, other));

        // Logical Equality.
        test_logical_equality(cv_const);
    }
}

///
/// Test vectorized unpacking of compressed data.
/// @tparam VecWidth The vector width to use.
/// @tparam Sign Indicate whether the compression is signed or unsigned.
/// @tparam Bits The number of bits use per element in the compression.
/// @tparam Extent The compile-time dimensionality (Dynamiic if runtime).
///
template <typename Sign, size_t Bits, size_t Extent, typename Strategy>
void test_unpacker(svs::lib::MaybeStatic<Extent> size = {}) {
    using CV = lvq::MutableCompressedVector<Sign, Bits, Extent, Strategy>;
    using value_type = typename CV::value_type;
    CATCH_REQUIRE(size >= 16);

    // Test the vectorized unpacking.
    constexpr size_t StorageExtent = CV::storage_extent;
    size_t bytes = CV::compute_bytes(size);

    std::vector<std::byte> v(bytes);
    auto cv = CV(size, std::span<std::byte, StorageExtent>{v.data(), v.size()});
    auto reference = std::vector<value_type>(cv.size());
    auto dst = std::vector<value_type>(cv.size());
    auto g = test_q::create_generator<Sign, Bits>();

    auto populate_and_set = [&]() {
        svs_test::populate(reference, g);
        for (size_t j = 0; j < cv.size(); ++j) {
            cv.set(reference.at(j), j);
        }
    };

    const size_t ntests = 10;
    for (size_t i = 0; i < ntests; ++i) {
        populate_and_set();
        lvq::unpack(dst, cv.as_const());
        CATCH_REQUIRE(std::equal(dst.begin(), dst.end(), reference.begin()));
    }

    // Test handling of varying tail-element reads for the sequential strategy.
    if constexpr (lvq::UsesSequential<CV>) {
        populate_and_set();
        auto helper = lvq::prepare_unpack(cv.as_const());
        for (size_t i = 1; i <= 16; ++i) {
            auto w = lvq::unpack_as(
                cv.as_const(),
                0,
                eve::as<svs::wide_<int32_t, 16>>(),
                helper,
                eve::keep_first(i)
            );
            // Check that the lower reads are correct.
            for (size_t j = 0; j < i; ++j) {
                CATCH_REQUIRE(w.get(j) == reference.at(j));
            }
            // Check that all higher values are set to zero.
            for (size_t j = i; j < 16; ++j) {
                CATCH_REQUIRE(w.get(j) == 0);
            }
        }
    }
}

// Helper alias to cut down on visual clutter.
template <size_t N> using Val = svs::lib::Val<N>;

template <size_t Lanes, size_t ElementsPerLane>
using Turbo = lvq::Turbo<Lanes, ElementsPerLane>;

} // namespace

CATCH_TEST_CASE("Quantization Utilities", "[quantization][lvq][lvq_compressed]") {
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

        ///// Turbo
        using Turbo16x8 = lvq::Turbo<16, 8>;
        test_compressed<lvq::Signed, 4, N, Turbo16x8>();
        test_compressed<lvq::Signed, 4, svs::Dynamic, Turbo16x8>(static_n);
        test_compressed<lvq::Unsigned, 4, N, Turbo16x8>();
        test_compressed<lvq::Unsigned, 4, svs::Dynamic, Turbo16x8>(static_n);
    }

    CATCH_SECTION("Unpacking - Sequential") {
        constexpr size_t Dynamic = svs::Dynamic;
        using Sequential = lvq::Sequential;
        CATCH_REQUIRE(Sequential::name() == "sequential");
        // using i32 = int32_t;
        // using wide_32x16 = eve::wide<i32, eve::fixed<16>>;

        constexpr size_t N = 37;
        constexpr auto static_n = svs::lib::MaybeStatic(N);

        ///// Sequential

        auto test_sequential = [&]<size_t Bits>() {
            test_unpacker<lvq::Signed, Bits, N, Sequential>();
            test_unpacker<lvq::Unsigned, Bits, N, Sequential>();

            test_unpacker<lvq::Signed, Bits, Dynamic, Sequential>(static_n);
            test_unpacker<lvq::Unsigned, Bits, Dynamic, Sequential>(static_n);
        };

        test_sequential.template operator()<3>();
        test_sequential.template operator()<4>();
        test_sequential.template operator()<5>();
        test_sequential.template operator()<6>();
        test_sequential.template operator()<7>();
        test_sequential.template operator()<8>();
    }

    CATCH_SECTION("Unpacking - Turbo") {
        constexpr size_t Dynamic = svs::Dynamic;
        constexpr size_t N = 539;
        constexpr auto static_n = svs::lib::MaybeStatic(N);

        CATCH_SECTION("8-bit: Turbo<16, 4>") {
            test_unpacker<lvq::Unsigned, 8, N, Turbo<16, 4>>();
            test_unpacker<lvq::Unsigned, 8, Dynamic, Turbo<16, 4>>(static_n);
            CATCH_REQUIRE(Turbo<16, 4>::name() == "turbo<16x4>");
        }

        CATCH_SECTION("4-bit: Turbo<16, 8>") {
            test_unpacker<lvq::Unsigned, 4, N, Turbo<16, 8>>();
            test_unpacker<lvq::Unsigned, 4, Dynamic, Turbo<16, 8>>(static_n);
            CATCH_REQUIRE(Turbo<16, 8>::name() == "turbo<16x8>");
        }
    }
}
