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

// header under test
#include "svs/lib/saveload.h"

// svs
#include "svs/lib/file.h"

// test helpers
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

// stl
#include <filesystem>

namespace {

inline constexpr std::string_view all_toml_types = R"(
__schema__ = "a_random_schema"
__version__ = "v1.2.3"
a = 1
b = 2.0
c = true
d = "hello"
e = [1, 2, 3]

    [f]
    __schema__ = "subschema"
    __version__ = "v10.20.30"
    a = 10
    b = 20.0
)";

template <bool HasContext, typename T>
void test_node_view(const T& x, const toml::table& base) {
    CATCH_REQUIRE(&x.unwrap() == &base);
    CATCH_REQUIRE(x.schema() == "a_random_schema");
    CATCH_REQUIRE(x.version() == svs::lib::Version(1, 2, 3));
    CATCH_REQUIRE(x.contains("a"));
    CATCH_REQUIRE(x.contains("b"));
    CATCH_REQUIRE(x.contains("c"));
    CATCH_REQUIRE(x.contains("d"));
    CATCH_REQUIRE(x.contains("e"));
    CATCH_REQUIRE(x.contains("f"));
    CATCH_REQUIRE(!x.contains("not_a_key"));

    auto check_context = [&](const auto& derived) {
        if constexpr (HasContext) {
            CATCH_REQUIRE(&x.context() == &derived.context());
        }
    };

    auto bad_node_cast =
        svs_test::ExceptionMatcher(Catch::Matchers::ContainsSubstring("Bad node cast"));

    constexpr std::string_view not_a_key = "not_a_key";

    // Invalid accesses
    CATCH_REQUIRE_THROWS_MATCHES(
        x.at(not_a_key),
        svs::ANNException,
        svs_test::ExceptionMatcher(Catch::Matchers::ContainsSubstring("Bad access to key"))
    );

    CATCH_REQUIRE(!x.try_at(not_a_key));

    // Testing for ``toml::value`` types.
    auto check_value = [&]<typename As, typename Not>(std::string_view key, As expected) {
        auto yn = x.at(key);
        check_context(yn);

        // Refine to the expected type.
        auto y = yn.template cast<toml::value<As>>();
        check_context(y);

        // Implicit conversion to the target TOML type should work.
        toml::value<As> c = y;
        CATCH_REQUIRE(c.get() == expected);

        // An invalid cast should throw.
        CATCH_REQUIRE_THROWS_MATCHES(
            yn.template cast<toml::value<Not>>(), svs::ANNException, bad_node_cast
        );

        // An invalid cast using `try_cast` should return an empty optional.
        CATCH_REQUIRE(!yn.template try_cast<toml::value<Not>>());

        // However, a correct `try_cast` should yield the same result as `cast`.
        auto oy = yn.template try_cast<toml::value<As>>();
        CATCH_REQUIRE(oy);
        check_context(*oy);
        CATCH_REQUIRE(&(oy->unwrap()) == &y.unwrap());

        // Finally, using `try_at` on the parent should yield the same result.
        auto oyn = x.try_at(key);
        CATCH_REQUIRE(oyn);
        CATCH_REQUIRE(&(oyn->unwrap()) == &yn.unwrap());
    };

    // Values
    check_value.template operator()<int64_t, std::string>("a", 1);
    check_value.template operator()<double, bool>("b", 2.0);
    check_value.template operator()<bool, int64_t>("c", true);
    check_value.template operator()<std::string, double>("d", "hello");

    // Array
    {
        auto yn = x.at("e").template cast<toml::array>();
        check_context(yn);

        // A local accumulator to match against the extracted values.
        int64_t i = 1;
        yn.visit([&i, &check_context](auto v) {
            // v should propagate the context (if needed);
            check_context(v);
            toml::value<int64_t> val = v.template cast<toml::value<int64_t>>();
            CATCH_REQUIRE(val.get() == i);
            ++i;
        });
        // Ensure we had the correct number of iterations.
        CATCH_REQUIRE(i == 4);

        // Ensure that try-cast to an array works.
        auto oyn = x.at("e").template try_cast<toml::array>();
        CATCH_REQUIRE(oyn);
        CATCH_REQUIRE(&(oyn->unwrap()) == &yn.unwrap());
    }

    // Table
    {
        auto yn = x.at("f").template cast<toml::table>();
        CATCH_REQUIRE(yn.schema() == "subschema");
        CATCH_REQUIRE(yn.version() == svs::lib::Version(10, 20, 30));
        check_context(yn);

        toml::value<int64_t> a = yn.at("a").template cast<toml::value<int64_t>>();
        CATCH_REQUIRE(a.get() == 10);

        toml::value<double> b = yn.at("b").template cast<toml::value<double>>();
        CATCH_REQUIRE(b.get() == 20.0);

        auto oyn = x.at("f").template try_cast<toml::table>();
        CATCH_REQUIRE(oyn);
        CATCH_REQUIRE(&(oyn->unwrap()) == &yn.unwrap());
    }
}

} // namespace

CATCH_TEST_CASE("Loader V2", "[lib][saveload][loader_v2]") {
    auto table = toml::parse(all_toml_types);
    CATCH_SECTION("ContextFreeNodeView") {
        auto view = svs::lib::ContextFreeNodeView<toml::table>(table);
        test_node_view<false>(view, table);
    }

    CATCH_SECTION("NodeView") {
        auto ctx = svs::lib::LoadContext(".", svs::lib::Version(1, 2, 3));
        auto view = svs::lib::NodeView<toml::table>(table, ctx);
        test_node_view<true>(view, table);
    }
}
