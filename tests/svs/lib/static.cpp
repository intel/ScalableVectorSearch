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

#include <type_traits>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "svs/lib/static.h"

CATCH_TEST_CASE("MaybeStatic", "[core]") {
    CATCH_SECTION("Basic") {
        CATCH_REQUIRE(svs::lib::MaybeStatic<128>().size() == 128);
        CATCH_REQUIRE(svs::lib::MaybeStatic<10>().size() == 10);
        CATCH_REQUIRE(svs::lib::MaybeStatic<200>().size() == 200);

        CATCH_REQUIRE(svs::lib::MaybeStatic(128).size() == 128);
        CATCH_REQUIRE(svs::lib::MaybeStatic(10).size() == 10);
        CATCH_REQUIRE(svs::lib::MaybeStatic(200).size() == 200);

        // Test that the default constructor for dynamicly size `MaybeStatic` is
        // deleted.
        CATCH_REQUIRE(
            std::is_default_constructible_v<svs::lib::MaybeStatic<svs::Dynamic>> == false
        );

        // Test the constexpress of `MaybeStatic`
        constexpr size_t N = svs::lib::MaybeStatic<5>().size();
        CATCH_REQUIRE(N == 5);

        constexpr size_t M = svs::lib::MaybeStatic(5).size();
        CATCH_REQUIRE(M == 5);
    }

    CATCH_SECTION("Equality") {
        namespace lib = svs::lib;
        // Both Static
        CATCH_REQUIRE(lib::MaybeStatic<100>{} == lib::MaybeStatic<100>{});
        CATCH_REQUIRE(lib::MaybeStatic<100>{} != lib::MaybeStatic<101>{});
        CATCH_REQUIRE(lib::MaybeStatic<101>{} != lib::MaybeStatic<100>{});
        CATCH_REQUIRE(lib::MaybeStatic<101>{} == lib::MaybeStatic<101>{});

        // Mixed
        CATCH_REQUIRE(lib::MaybeStatic<10>{} != lib::MaybeStatic{10});
        CATCH_REQUIRE(lib::MaybeStatic<11>{} != lib::MaybeStatic{10});
        CATCH_REQUIRE(lib::MaybeStatic<10>{} != lib::MaybeStatic{11});
        CATCH_REQUIRE(lib::MaybeStatic<11>{} != lib::MaybeStatic{11});

        CATCH_REQUIRE(lib::MaybeStatic{10} != lib::MaybeStatic<10>{});
        CATCH_REQUIRE(lib::MaybeStatic{11} != lib::MaybeStatic<10>{});
        CATCH_REQUIRE(lib::MaybeStatic{10} != lib::MaybeStatic<11>{});
        CATCH_REQUIRE(lib::MaybeStatic{11} != lib::MaybeStatic<11>{});

        // Dynamic
        CATCH_REQUIRE(lib::MaybeStatic{100} == lib::MaybeStatic{100});
        CATCH_REQUIRE(lib::MaybeStatic{100} != lib::MaybeStatic{101});
        CATCH_REQUIRE(lib::MaybeStatic{101} != lib::MaybeStatic{100});
        CATCH_REQUIRE(lib::MaybeStatic{101} == lib::MaybeStatic{101});
    }

    CATCH_SECTION("Is Last") {
        constexpr bool temp1 = svs::lib::islast<16>(svs::lib::MaybeStatic<128>(), 112);
        CATCH_REQUIRE(temp1 == true);
        constexpr bool temp2 = svs::lib::islast<16>(svs::lib::MaybeStatic<128>(), 111);
        CATCH_REQUIRE(temp2 == false);

        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic<4>(), 0) == true);
        CATCH_REQUIRE(svs::lib::islast<4>(svs::lib::MaybeStatic<8>(), 0) == false);
        CATCH_REQUIRE(svs::lib::islast<4>(svs::lib::MaybeStatic<8>(), 4) == true);

        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic<128>(), 111) == false);
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic<128>(), 112) == true);
        // Indexing at 113 will yield an out-of-bounds access, so we should return "false"
        // for the laster iteration (since the true last iteration begins at index 112.
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic<128>(), 113) == false);

        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic<100>(), 95) == false);
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic<100>(), 96) == true);
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic<100>(), 97) == false);

        ///// Dynamic Lengths
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic(4), 0) == true);
        CATCH_REQUIRE(svs::lib::islast<4>(svs::lib::MaybeStatic(8), 0) == false);
        CATCH_REQUIRE(svs::lib::islast<4>(svs::lib::MaybeStatic(8), 4) == true);

        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic(128), 111) == false);
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic(128), 112) == true);
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic(128), 113) == false);

        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic(100), 95) == false);
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic(100), 96) == true);
        CATCH_REQUIRE(svs::lib::islast<16>(svs::lib::MaybeStatic(100), 97) == false);
    }

    CATCH_SECTION("Upper and Rest") {
        namespace lib = svs::lib;
        // Evenly divides
        CATCH_REQUIRE(lib::upper<8>(lib::MaybeStatic<16>{}) == 16);
        CATCH_REQUIRE(lib::rest<8>(lib::MaybeStatic<16>{}) == lib::MaybeStatic<0>{});
        CATCH_REQUIRE(lib::upper<8>(lib::MaybeStatic{16}) == 16);
        CATCH_REQUIRE(lib::rest<8>(lib::MaybeStatic{16}) == lib::MaybeStatic{0});

        // Doesn't evenly divide
        CATCH_REQUIRE(lib::upper<8>(lib::MaybeStatic<17>{}) == 16);
        CATCH_REQUIRE(lib::rest<8>(lib::MaybeStatic<17>{}) == lib::MaybeStatic<1>{});
        CATCH_REQUIRE(lib::upper<8>(lib::MaybeStatic{17}) == 16);
        CATCH_REQUIRE(lib::rest<8>(lib::MaybeStatic{17}) == lib::MaybeStatic{1});

        CATCH_REQUIRE(lib::upper<8>(lib::MaybeStatic<15>{}) == 8);
        CATCH_REQUIRE(lib::rest<8>(lib::MaybeStatic<15>{}) == lib::MaybeStatic<7>{});
        CATCH_REQUIRE(lib::upper<8>(lib::MaybeStatic{15}) == 8);
        CATCH_REQUIRE(lib::rest<8>(lib::MaybeStatic{15}) == lib::MaybeStatic{7});
    }
}

CATCH_TEST_CASE("Extract Extent", "[core]") {
    constexpr size_t a = svs::lib::extract_extent(10, 10);
    CATCH_REQUIRE(a == 10);

    constexpr size_t b = svs::lib::extract_extent(100, svs::Dynamic);
    CATCH_REQUIRE(b == 100);

    constexpr size_t c = svs::lib::extract_extent(svs::Dynamic, 25);
    CATCH_REQUIRE(c == 25);

    CATCH_REQUIRE_THROWS_AS(svs::lib::extract_extent(10, 20), std::logic_error);
}
