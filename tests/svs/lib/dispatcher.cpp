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
#include "svs/lib/dispatcher.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

enum class Kind { Add, Subtract, Multiply };

struct Arithmetic {
    using key_type = std::tuple<Kind, size_t>;
    using mapped_type = std::function<int64_t(int64_t, int64_t)>;

    template <typename F> static void fill(F&& f) {
        f({{Kind::Add, svs::Dynamic}, [](int64_t x, int64_t y) { return x + y; }});
        f({{Kind::Subtract, svs::Dynamic}, [](int64_t x, int64_t y) { return x - y; }});
        f({{Kind::Multiply, svs::Dynamic}, [](int64_t x, int64_t y) { return x * y; }});
        f({{Kind::Multiply, 0}, [](int64_t, int64_t) { return 0; }});
    }
};

using Dispatcher = svs::lib::Dispatcher<Arithmetic>;

} // namespace

CATCH_TEST_CASE("Dispatcher", "[lib][dispatcher]") {
    CATCH_SECTION("Correct dispatching") {
        CATCH_REQUIRE(Dispatcher::contains(false, svs::Dynamic, Kind::Add));
        CATCH_REQUIRE(!Dispatcher::contains(false, 0, Kind::Add));
        auto f = Dispatcher::lookup(false, svs::Dynamic, Kind::Add);
        CATCH_REQUIRE(f(10, 20) == 30);

        CATCH_REQUIRE(Dispatcher::contains(false, svs::Dynamic, Kind::Subtract));
        CATCH_REQUIRE(!Dispatcher::contains(false, 0, Kind::Subtract));
        f = Dispatcher::lookup(false, svs::Dynamic, Kind::Subtract);
        CATCH_REQUIRE(f(10, 20) == -10);

        CATCH_REQUIRE(Dispatcher::contains(false, svs::Dynamic, Kind::Multiply));
        f = Dispatcher::lookup(false, svs::Dynamic, Kind::Multiply);
        CATCH_REQUIRE(f(10, 20) == 200);

        CATCH_REQUIRE(Dispatcher::contains(false, 0, Kind::Multiply));
        f = Dispatcher::lookup(false, 0, Kind::Multiply);
        CATCH_REQUIRE(f(10, 20) == 0);
    }

    CATCH_SECTION("Fallback") {
        CATCH_REQUIRE(Dispatcher::contains(true, 0, Kind::Add));
        auto f = Dispatcher::lookup(true, 0, Kind::Add);
        CATCH_REQUIRE(f(10, 20) == 30);

        CATCH_REQUIRE(Dispatcher::contains(true, 0, Kind::Subtract));
        f = Dispatcher::lookup(true, 0, Kind::Subtract);
        CATCH_REQUIRE(f(10, 20) == -10);

        CATCH_REQUIRE(Dispatcher::contains(true, svs::Dynamic, Kind::Multiply));
        f = Dispatcher::lookup(true, svs::Dynamic, Kind::Multiply);
        CATCH_REQUIRE(f(10, 20) == 200);

        CATCH_REQUIRE(Dispatcher::contains(true, 0, Kind::Multiply));
        f = Dispatcher::lookup(true, 0, Kind::Multiply);
        CATCH_REQUIRE(f(10, 20) == 0);
    }

    CATCH_SECTION("Exception path") {
        CATCH_REQUIRE_THROWS_AS(Dispatcher::lookup(false, 0, Kind::Add), svs::ANNException);
    }
}
