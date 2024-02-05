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

// header under test
#include "svs/quantization/lvq/compressed.h"

// Helper
#include "tests/svs/quantization/lvq/common.h"

// misc svs
#include "svs/lib/meta.h"
#include "svs/lib/saveload.h"

// svs-test
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <vector>

namespace lvq = svs::quantization::lvq;

namespace {

/////
///// Turbo Permutations
/////

// Load permutations from a file.
struct TurboPermutation {
  public:
    size_t lanes_;
    size_t elements_per_lane_;
    std::vector<uint16_t> perm_;

  public:
    TurboPermutation(size_t lanes, size_t elements_per_lane, std::vector<uint16_t> perm)
        : lanes_{lanes}
        , elements_per_lane_{elements_per_lane}
        , perm_{std::move(perm)} {}

    static TurboPermutation load(const toml::table& table, const svs::lib::Version&) {
        return TurboPermutation{
            SVS_LOAD_MEMBER_AT_(table, lanes),
            SVS_LOAD_MEMBER_AT_(table, elements_per_lane),
            SVS_LOAD_MEMBER_AT_(table, perm),
        };
    }
};

std::vector<TurboPermutation> get_permutations() {
    auto permutations = svs_test::data_directory() / "lvq" / "turbo_permutations.toml";
    auto table = toml::parse_file(permutations.string());
    return svs::lib::load_at<std::vector<TurboPermutation>>(table, "turbo_permutations");
}

/////
///// Turbo Compressed Vector
/////

template <typename Sign, size_t Bits, size_t Extent, size_t Lanes, size_t ElementsPerLane>
struct TurboCompressedVectorTester {
    using turbo_type = lvq::Turbo<Lanes, ElementsPerLane>;
    using Compressed = lvq::CompressedVector<Sign, Bits, Extent, turbo_type>;
    using MutCompressed = lvq::MutableCompressedVector<Sign, Bits, Extent, turbo_type>;

    using value_type = typename MutCompressed::value_type;

    static_assert(Compressed::is_const);
    static_assert(!MutCompressed::is_const);

    static constexpr size_t storage_extent = Compressed::storage_extent;
    static constexpr size_t oversized_extent =
        storage_extent == svs::Dynamic ? svs::Dynamic : storage_extent + 10;

    // Methods
    std::vector<std::byte> create_storage(size_t bytes) const {
        return std::vector<std::byte>(bytes);
    }

    void test_constructors(svs::lib::MaybeStatic<Extent> size = {}) const {
        // Create the raw-storage for testing.
        auto storage_bytes = Compressed::compute_bytes(size);
        auto storage = create_storage(storage_bytes + 10);

        auto span = std::span<std::byte, storage_extent>(storage.begin(), storage_bytes);
        auto const_span = std::span<const std::byte, storage_extent>(span);

        auto oversized_span =
            std::span<std::byte, oversized_extent>(storage.begin(), storage_bytes + 10);
        auto oversized_const_span =
            std::span<const std::byte, oversized_extent>(oversized_span);

        // Testing routines.
        auto test_cv = [&](const auto& cv) {
            CATCH_REQUIRE(cv.size() == size);
            size_t expected_bytes = Compressed::compute_bytes(size);
            CATCH_REQUIRE(cv.size_bytes() == expected_bytes);
            CATCH_REQUIRE(cv.data() == storage.data());

            if constexpr (Extent != svs::Dynamic) {
                CATCH_STATIC_REQUIRE(
                    Compressed::compute_bytes() == Compressed::compute_bytes(size)
                );
            }
        };

        // Compile time lengths
        if constexpr (Extent != svs::Dynamic) {
            auto cv_mut = MutCompressed{span};
            test_cv(cv_mut);
            test_cv(cv_mut.as_const());
            auto cv_const = Compressed{span};
            test_cv(cv_const);
        }

        // Standard Constructors.
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
            auto cv = MutCompressed{lvq::AllowShrinkingTag(), size, oversized_span};
            test_cv(cv);
            test_cv(cv.as_const());

            auto cv_const =
                Compressed{lvq::AllowShrinkingTag(), size, oversized_const_span};
            test_cv(cv_const);
            cv_const = Compressed{lvq::AllowShrinkingTag(), size, oversized_span};
            test_cv(cv_const);

            // If we're constructing a static-length CV, try building from a dynamic span.
            if constexpr (Extent == svs::Dynamic) {
                auto oversized_span_d = std::span<std::byte, svs::Dynamic>(storage);
                auto oversized_const_span_d =
                    std::span<const std::byte, svs::Dynamic>(storage);
                cv = MutCompressed{lvq::AllowShrinkingTag(), size, oversized_span_d};
                test_cv(cv);
                cv_const =
                    Compressed{lvq::AllowShrinkingTag(), size, oversized_const_span_d};
                test_cv(cv_const);
                cv_const = Compressed{lvq::AllowShrinkingTag(), size, oversized_span_d};
                test_cv(cv_const);
            }
        }
    }

    void test(svs::lib::MaybeStatic<Extent> size = {}, size_t num_tests = 5) const {
        test_constructors(size);

        // Make sure the `value_type` is a suitable signed small integer.
        if constexpr (std::is_same_v<Sign, lvq::Signed>) {
            static_assert(std::is_same_v<value_type, int8_t>);
        } else {
            static_assert(std::is_same_v<Sign, lvq::Unsigned>);
            static_assert(std::is_same_v<value_type, uint8_t>);
        }

        size_t storage_bytes = Compressed::compute_bytes(size);

        // Allocate memory and construct a `CompressedVector` view over the data.
        std::vector<std::byte> v(storage_bytes);
        MutCompressed cv{size, typename MutCompressed::span_type{v}};

        CATCH_REQUIRE((cv.size() == size));
        CATCH_REQUIRE(cv.extent == Extent);
        if constexpr (Extent != svs::Dynamic) {
            CATCH_REQUIRE(cv.size() == Extent);
            CATCH_REQUIRE(cv.storage_extent == storage_bytes);
            // Only need 8-bytes for the pointer
            CATCH_REQUIRE(sizeof(MutCompressed) == 8);
        } else {
            CATCH_REQUIRE(cv.storage_extent == svs::Dynamic);
            // Need 24 bytes - 8 for the actual length, 8 for the pointer, and 8 for the
            // span length.
            CATCH_REQUIRE(sizeof(MutCompressed) == 24);
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
        for (size_t i = 0; i < num_tests; ++i) {
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
            auto other = other_storage.template view<Sign, Bits, Extent, turbo_type>(size);
            CATCH_REQUIRE(other.size() == cv_size);

            // Rely on std::vector initializing to 0.
            for (size_t j = 0; j < cv_size; ++j) {
                CATCH_REQUIRE(other.get(j) == Compressed::decode(0));
            }
            other.copy_from(cv);
            for (size_t j = 0; j < cv_size; ++j) {
                CATCH_REQUIRE(other.get(j) == reference.at(j));
            }

            // Copy from a vector.
            svs_test::populate(reference, g);
            other.copy_from(reference);
            for (size_t j = 0; j < cv_size; ++j) {
                CATCH_REQUIRE(other.get(j) == reference.at(j));
            }

            bool some_different = false;
            for (size_t j = 0; j < cv_size; ++j) {
                if (other.get(j) != cv.get(j)) {
                    some_different = true;
                }
            }
            CATCH_REQUIRE(some_different);
        }
    }
};

} // namespace

CATCH_TEST_CASE("Turbo Compressed Vector", "[quantization][lvq][turbo]") {
    CATCH_SECTION("Permutations") {
        auto permutations = get_permutations();
        auto check = [&permutations]<size_t Lanes, size_t ElementsPerLane>() {
            auto itr = std::find_if(
                permutations.begin(),
                permutations.end(),
                [](const auto& entry) {
                    return entry.lanes_ == Lanes &&
                           entry.elements_per_lane_ == ElementsPerLane;
                }
            );
            CATCH_REQUIRE(itr != permutations.end());
            using Turbo = lvq::Turbo<Lanes, ElementsPerLane>;

            auto& entry = *itr;
            auto& perm = entry.perm_;

            for (size_t i = 0, imax = perm.size(); i < imax; ++i) {
                CATCH_REQUIRE(Turbo::logical_to_linear(i) == perm.at(i));
                CATCH_REQUIRE(Turbo::linear_to_logical(Turbo::logical_to_linear(i)) == i);
            }
        };

        check.template operator()<8, 2>();
        check.template operator()<8, 4>();
        check.template operator()<8, 8>();

        check.template operator()<16, 2>();
        check.template operator()<16, 4>();
        check.template operator()<16, 8>();

        check.template operator()<32, 2>();
        check.template operator()<32, 4>();
        check.template operator()<32, 8>();
    }

    CATCH_SECTION("Turbo Compressed") {
        // Only test the AVX-512 ABI combinations for now.
        TurboCompressedVectorTester<lvq::Unsigned, 4, 157, 16, 8>().test();
        TurboCompressedVectorTester<lvq::Signed, 4, 157, 16, 8>().test();

        TurboCompressedVectorTester<lvq::Unsigned, 8, 157, 16, 4>().test();
        TurboCompressedVectorTester<lvq::Signed, 8, 157, 16, 4>().test();
    }
}
