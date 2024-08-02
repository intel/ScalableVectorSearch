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

// Header under test.
#include "svs/lib/expected.h"

// Test utils
#include "tests/utils/require_error.h"

// Catch2
#include "catch2/catch_test_macros.hpp"

// STL
#include <vector>

CATCH_TEST_CASE("Expected", "[lib][expected]") {
    CATCH_SECTION("Unexpected") {
        auto v = std::vector<int>{{1, 2, 3}};

        CATCH_SECTION("Construction by const-ref") {
            auto u = svs::lib::Unexpected(v);
            CATCH_REQUIRE(std::as_const(u).value() == v);
            CATCH_REQUIRE(std::as_const(u).value().data() != v.data()); // copy was made
        }

        CATCH_SECTION("Construction by rvalue-ref") {
            auto* data = v.data();
            auto u = svs::lib::Unexpected(std::move(v));
            CATCH_REQUIRE(u.value().data() == data);
        }

        CATCH_SECTION("Accessors") {
            auto u = svs::lib::Unexpected(v);
            const auto& u_const = std::as_const(u).value();
            auto& u_mut = u.value();
            u_mut.push_back(4);
            CATCH_REQUIRE(u_const.back() == 4);

            const auto* data = u_const.data();
            auto moved = std::move(u).value();
            CATCH_REQUIRE(moved.data() == data);
        }

        CATCH_SECTION("Comparison") {
            auto a = svs::lib::Unexpected(1);
            auto b = svs::lib::Unexpected(2);
            auto c = svs::lib::Unexpected(1);

            CATCH_REQUIRE(a == c);
            CATCH_REQUIRE(a <= c);
            CATCH_REQUIRE(a >= c);
            CATCH_REQUIRE(!(a < c));
            CATCH_REQUIRE(!(a > c));

            CATCH_REQUIRE(a != b);
            CATCH_REQUIRE(a <= b);
            CATCH_REQUIRE(a < b);
            CATCH_REQUIRE(b >= a);
            CATCH_REQUIRE(b > a);
        }
    }

    CATCH_SECTION("Expected") {
        CATCH_SECTION("Aliases") {
            using E = svs::lib::Expected<float, int>;
            CATCH_STATIC_REQUIRE(std::is_same_v<typename E::value_type, float>);
            CATCH_STATIC_REQUIRE(std::is_same_v<typename E::error_type, int>);
            CATCH_STATIC_REQUIRE(std::is_same_v<
                                 typename E::unexpected_type,
                                 svs::lib::Unexpected<int>>);
        }

        CATCH_SECTION("Constructors") {
            using T = svs::lib::Expected<std::vector<int>, std::vector<double>>;
            auto v = std::vector<int>{{1, 2, 3}};
            auto e = std::vector<double>{{-1.0, -2.0, -3.0}};
            CATCH_SECTION("Expected - const-ref") {
                auto u = T{v};
                CATCH_REQUIRE(u); // contextual conversion to bool
                CATCH_REQUIRE(u.has_value());
                CATCH_REQUIRE(std::as_const(u).value() == v);
                CATCH_REQUIRE(std::as_const(u).value().data() != v.data());
            }

            CATCH_SECTION("Expected - rvalue-ref") {
                const auto* data = v.data();
                auto u = T{std::move(v)};
                CATCH_REQUIRE(u);
                CATCH_REQUIRE(u.has_value());
                CATCH_REQUIRE(u.value().data() == data);
            }

            CATCH_SECTION("Unexpected - const-ref") {
                auto unexpected = svs::lib::Unexpected{e};
                auto u = T{unexpected};
                CATCH_REQUIRE(!u);
                CATCH_REQUIRE(!u.has_value());
                CATCH_REQUIRE(std::as_const(u).error() == e);
                CATCH_REQUIRE(std::as_const(u).error().data() != e.data());
            }

            CATCH_SECTION("Unexpected - const-ref") {
                const auto* data = e.data();
                auto u = T{svs::lib::Unexpected{std::move(e)}};
                CATCH_REQUIRE(!u);
                CATCH_REQUIRE(!u.has_value());
                CATCH_REQUIRE(std::as_const(u).error().data() == data);
            }
        }

        CATCH_SECTION("Observers") {
            using T = svs::lib::Expected<std::vector<int>, std::vector<double>>;
            auto v = std::vector<int>{{1, 2, 3}};
            auto e = std::vector<double>{{-1, -2, -3}};

            // Make sure we can't directly construct an Expected from an unwrapped
            // unexpected type.
            SVS_REQUIRE_DOES_NOT_COMPILE(T, TestType{e});
            SVS_REQUIRE_DOES_NOT_COMPILE(T, TestType{std::move(e)});

            // Cannot compile construction from the wrong Unexpected type.
            SVS_REQUIRE_DOES_NOT_COMPILE(T, TestType{svs::lib::Unexpected{v}});

            CATCH_SECTION("Expected") {
                auto as_expected = T{v};
                CATCH_REQUIRE(as_expected);

                // Unsafe dereference operators.
                CATCH_REQUIRE(*as_expected == v);
                CATCH_REQUIRE(as_expected->data() != v.data());

                // Ensure errors are thrown when trying to access as an error.
                CATCH_REQUIRE_THROWS_AS(
                    std::as_const(as_expected).error(), svs::ANNException
                );
                CATCH_REQUIRE_THROWS_AS(std::move(as_expected).error(), svs::ANNException);

                // After throwing an exception, the container should be unmodified.
                CATCH_REQUIRE(std::as_const(as_expected).value() == v);
                const auto* data = as_expected->data();
                auto moved_from = std::move(as_expected).value();
                CATCH_REQUIRE(moved_from == v);
                CATCH_REQUIRE(moved_from.data() == data);
            }

            CATCH_SECTION("Unexpected") {
                auto as_unexpected = T{svs::lib::Unexpected{e}};
                CATCH_REQUIRE(!as_unexpected);

                // Ensure errors are thrown why trying to access a value.
                CATCH_REQUIRE_THROWS_AS(
                    std::as_const(as_unexpected).value(), svs::ANNException
                );
                CATCH_REQUIRE_THROWS_AS(
                    std::move(as_unexpected).value(), svs::ANNException
                );

                // After throwing an exception, the container should be unmodified.
                CATCH_REQUIRE(std::as_const(as_unexpected).error() == e);
                const auto* data = as_unexpected.error().data();
                auto moved_from = std::move(as_unexpected).error();
                CATCH_REQUIRE(moved_from == e);
                CATCH_REQUIRE(moved_from.data() == data);
            }
        }
    }
}
