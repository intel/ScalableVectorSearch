/*
 * Copyright 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// header under test
#include "svs/core/logging.h"
#include "svs/lib/narrow.h"

// svs
#include "svs/third-party/fmt.h"

// spdlog
#include "spdlog/sinks/callback_sink.h"
#include "spdlog/spdlog.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Logging", "[core][logging]") {
    CATCH_SECTION("Enum Conversion") {
        using enum svs::logging::Level;
        // Log level integer comparisons.
        CATCH_STATIC_REQUIRE(static_cast<int>(Trace) == 0);
        CATCH_STATIC_REQUIRE(static_cast<int>(Debug) == 1);
        CATCH_STATIC_REQUIRE(static_cast<int>(Info) == 2);
        CATCH_STATIC_REQUIRE(static_cast<int>(Warn) == 3);
        CATCH_STATIC_REQUIRE(static_cast<int>(Error) == 4);
        CATCH_STATIC_REQUIRE(static_cast<int>(Critical) == 5);
        CATCH_STATIC_REQUIRE(static_cast<int>(Off) == 6);

        CATCH_STATIC_REQUIRE(
            svs::logging::all_levels ==
            std::array<svs::logging::Level, 7>{
                Trace, Debug, Info, Warn, Error, Critical, Off}
        );

        // SVS to spdlog
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::to_spdlog(Trace) == spdlog::level::trace
        );
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::to_spdlog(Debug) == spdlog::level::debug
        );
        CATCH_STATIC_REQUIRE(svs::logging::detail::to_spdlog(Info) == spdlog::level::info);
        CATCH_STATIC_REQUIRE(svs::logging::detail::to_spdlog(Warn) == spdlog::level::warn);
        CATCH_STATIC_REQUIRE(svs::logging::detail::to_spdlog(Error) == spdlog::level::err);
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::to_spdlog(Critical) == spdlog::level::critical
        );
        CATCH_STATIC_REQUIRE(svs::logging::detail::to_spdlog(Off) == spdlog::level::off);
        // spdlog to SVS
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::from_spdlog(spdlog::level::trace) == Trace
        );
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::from_spdlog(spdlog::level::debug) == Debug
        );
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::from_spdlog(spdlog::level::info) == Info
        );
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::from_spdlog(spdlog::level::warn) == Warn
        );
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::from_spdlog(spdlog::level::err) == Error
        );
        CATCH_STATIC_REQUIRE(
            svs::logging::detail::from_spdlog(spdlog::level::critical) == Critical
        );
        CATCH_STATIC_REQUIRE(svs::logging::detail::from_spdlog(spdlog::level::off) == Off);
    }

    CATCH_SECTION("tolower") {
        auto conversions =
            std::string_view("AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz");

        constexpr auto tolower = svs::logging::detail::to_lower;

        // Use a signed 64-bit integer to access all possible values of `char`.
        auto lo = svs::lib::narrow<int64_t>(std::numeric_limits<char>::min());
        auto hi = svs::lib::narrow<int64_t>(std::numeric_limits<char>::max());
        for (int64_t i = lo, imax = hi; i < imax; ++i) {
            // Ensure lossless conversion.
            auto ch = svs::lib::narrow<char>(i);
            if (ch >= 'A' && ch <= 'Z') {
                auto pos = conversions.find(ch);
                CATCH_REQUIRE(pos != decltype(conversions)::npos);
                auto replacement = conversions.at(pos + 1);
                CATCH_REQUIRE(tolower(ch) == replacement);
            } else {
                CATCH_REQUIRE(tolower(ch) == ch);
            }
        }
    }

    CATCH_SECTION("Level Parsing") {
        using enum svs::logging::Level;
        // common names
        CATCH_REQUIRE(svs::logging::detail::level_from_string("trace") == Trace);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("TRACE") == Trace);

        CATCH_REQUIRE(svs::logging::detail::level_from_string("debug") == Debug);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("DEBUG") == Debug);

        CATCH_REQUIRE(svs::logging::detail::level_from_string("info") == Info);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("INFO") == Info);

        CATCH_REQUIRE(svs::logging::detail::level_from_string("warning") == Warn);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("WARNING") == Warn);

        CATCH_REQUIRE(svs::logging::detail::level_from_string("error") == Error);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("ERROR") == Error);

        CATCH_REQUIRE(svs::logging::detail::level_from_string("critical") == Critical);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("CRITICAL") == Critical);

        CATCH_REQUIRE(svs::logging::detail::level_from_string("off") == Off);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("OFF") == Off);

        // Aliases
        CATCH_REQUIRE(svs::logging::detail::level_from_string("warn") == Warn);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("WARN") == Warn);

        CATCH_REQUIRE(svs::logging::detail::level_from_string("err") == Error);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("ERR") == Error);

        // Reject mal-formed strings.
        CATCH_REQUIRE(svs::logging::detail::level_from_string("") == Off);
        CATCH_REQUIRE(svs::logging::detail::level_from_string("not a value") == Off);
    }

    CATCH_SECTION("Environment Variable Names") {
        // The full testing of environment variable based loading is done using an external
        // tool as part of the CI pipeline.
        //
        // Here, we test some basic functionality related to environment variable handling.
        //
        // If either of the two `static_asserts` fails, then documentation needs to be
        // updated.
        CATCH_STATIC_REQUIRE(
            std::string_view(svs::logging::detail::log_level_var_name()) == "SVS_LOG_LEVEL"
        );
        CATCH_STATIC_REQUIRE(
            std::string_view(svs::logging::detail::log_sink_var_name()) == "SVS_LOG_SINK"
        );
    }

    // Test logging functionality.
    CATCH_SECTION("Test Global Logger") {
        // We want to test that logging messages are generated at the right time.
        // To do this, we can register a callback logger and in the callback, inspect the
        // returned string for expected values.
        bool called;
        std::function<void(std::string_view)> checker = {};

        auto callback_logger = std::make_shared<spdlog::logger>(
            "callback_test_logger",
            std::make_shared<spdlog::sinks::callback_sink_mt>([&checker](const auto& message
                                                              ) {
                checker(svs::make_string_view(message.payload));
            })
        );

        // Set the custom logger.
        svs::logging::set(callback_logger);

        constexpr std::string_view fmt_string = "A = {}, B = {}, C = {}";
        auto expected = fmt::format(fmt_string, 1, 2, 3);

        // Check that the provided callback string is equal to the expected string.
        checker = [&called, &expected](std::string_view str) {
            CATCH_REQUIRE(!called);
            CATCH_REQUIRE(str == expected);
            called = true;
        };

        auto assert_callback_called = [&](auto&& f) {
            called = false;
            f();
            CATCH_REQUIRE(called);
        };

        auto assert_callback_not_called = [&](auto&& f) {
            called = false;
            f();
            CATCH_REQUIRE(!called);
        };

        // Set logging level to trace.
        for (auto level : svs::logging::all_levels) {
            // Set the logging to this level.
            svs::logging::set_level(level);
            // Make sure the level was set properly.
            CATCH_REQUIRE(svs::logging::get_level() == level);

            // Actually retrieve the logger.
            // Make sure that it has the same level as the reported earlier.
            auto logger = svs::logging::get();
            CATCH_REQUIRE(svs::logging::get_level(logger) == level);

            for (auto other_level : svs::logging::all_levels) {
                // Invoke the logger in multiple ways.
                // First, use our local pointer.
                // Second, use the implicit global logger.
                auto do_log_with_logger = [&]() {
                    svs::logging::log(logger, other_level, fmt_string, 1, 2, 3);
                };

                auto do_log_with_global_logger = [&]() {
                    svs::logging::log(other_level, fmt_string, 1, 2, 3);
                };

                // Determine whether or not a log should be made, then run tests to ensure
                // that the requested operation happens.
                if (other_level >= level) {
                    assert_callback_called(do_log_with_logger);
                    assert_callback_called(do_log_with_global_logger);
                } else {
                    assert_callback_not_called(do_log_with_logger);
                    assert_callback_not_called(do_log_with_global_logger);
                }
            }
        }

        // Restore defaults.
        svs::logging::reset_to_default();
    }
}
