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

// Header under test.
#include "svs/lib/timing.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <cstdlib>
#include <thread>

namespace {
double getrand() {
    return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}

void stress(svs::lib::Timer& timer, size_t max_depth = 3, size_t max_flat = 3) {
    for (size_t i = 0; i < max_flat; ++i) {
        auto handle = timer.push_back(fmt::format("Hello {}", i));
        if (max_depth == 0 && getrand() < 0.8) {
            return;
        }
        if (max_depth > 0 && getrand() < 0.8) {
            // Call twice to ensure we hit repeat measurements.
            stress(timer, max_depth - 1, max_flat);
            stress(timer, max_depth - 1, max_flat);
        }
        if (getrand() < 0.2) {
            handle.finish();
        }
    }
}
} // namespace

CATCH_TEST_CASE("Timing", "[lib][timing]") {
    CATCH_SECTION("TimeData") {
        auto x = svs::lib::TimeData();
        CATCH_REQUIRE(x.num_calls == 0);
        CATCH_REQUIRE(x.total_time == std::chrono::nanoseconds(0));

        x = svs::lib::TimeData(10, std::chrono::nanoseconds(100));
        auto y = svs::lib::TimeData(20, std::chrono::nanoseconds(210));
        x += y;
        CATCH_REQUIRE(x.num_calls == 10 + 20);
        CATCH_REQUIRE(x.total_time == std::chrono::nanoseconds(310));
        CATCH_REQUIRE(x.min_time == std::chrono::nanoseconds(100));
        CATCH_REQUIRE(x.max_time == std::chrono::nanoseconds(210));
    }

    CATCH_SECTION("Basic") {
        auto timer = svs::lib::Timer();
        {
            auto x = timer.push_back("a");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        {
            auto x = timer.push_back("b");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        {
            auto x = timer.push_back("b");
            auto y = timer.push_back("c");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Number of elapsed time should be pretty close to the sleep time.
        {
            const auto& t = timer.get_timer("a");
            CATCH_REQUIRE(t.get_num_calls() == 1);
            CATCH_REQUIRE(t.get_time() >= std::chrono::milliseconds(10));
            CATCH_REQUIRE(t.get_time() < std::chrono::milliseconds(11));
        }
        {
            const auto& t = timer.get_timer("b");
            CATCH_REQUIRE(t.get_num_calls() == 2);
            CATCH_REQUIRE(t.get_time() >= std::chrono::milliseconds(20));
            CATCH_REQUIRE(t.get_time() < std::chrono::milliseconds(21));

            const auto& u = t.get_timer("c");
            CATCH_REQUIRE(u.get_num_calls() == 1);
            CATCH_REQUIRE(u.get_time() >= std::chrono::milliseconds(10));
            CATCH_REQUIRE(u.get_time() < std::chrono::milliseconds(11));
        }

        // CATCH_REQUIRE_THROWS(timer.get_timer("not a timer"));
        timer.print();

        // The expected layout of the timer labels should look something like this:
        // ---
        // a
        // b
        //   c
        // ---
        // Therefore, the longest name should be 3 (including the indent spaces)
        CATCH_REQUIRE(timer.longest_name() == 3);
    }

    CATCH_SECTION("Stress Test") {
        auto timer = svs::lib::Timer();
        stress(timer);
        timer.print();
    }
}
