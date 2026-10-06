/*
 * Copyright 2026 Intel Corporation
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

// C API
#include "svs/c/svs_c.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// Test utilities
#include "c_api_test_utils.h"

// Standard library
#include <algorithm>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

// Records every message received by the custom logger callback.
struct LogRecorder {
    std::mutex mutex;
    std::vector<std::pair<svs_log_level_t, std::string>> messages;

    bool contains(svs_log_level_t level, const std::string& text) {
        std::lock_guard lock{mutex};
        return std::any_of(messages.begin(), messages.end(), [&](const auto& m) {
            return m.first == level && m.second.find(text) != std::string::npos;
        });
    }

    bool any_below(svs_log_level_t level) {
        std::lock_guard lock{mutex};
        return std::any_of(messages.begin(), messages.end(), [&](const auto& m) {
            return m.first < level;
        });
    }
};

void record_log(void* self, enum svs_log_level level, const char* message) {
    auto* recorder = static_cast<LogRecorder*>(self);
    std::lock_guard lock{recorder->mutex};
    recorder->messages.emplace_back(level, message);
}

void noop_log(void* /*self*/, enum svs_log_level /*level*/, const char* /*message*/) {}

// Restores the SVS built-in default logger when a test that calls
// svs_set_default_logger ends, so later tests never log through a callback whose `self`
// has been destroyed.
struct DefaultLoggerGuard {
    DefaultLoggerGuard() = default;
    DefaultLoggerGuard(const DefaultLoggerGuard&) = delete;
    DefaultLoggerGuard& operator=(const DefaultLoggerGuard&) = delete;
    ~DefaultLoggerGuard() { svs_set_default_logger(nullptr, nullptr); }
};

// Builds (and frees) a small static index. Vamana build logs at TRACE level through the
// global default logger, e.g. "Number of syncs: ..." and "Completed pass ...".
void build_small_index() {
    const size_t num_vectors = 100;
    const size_t dimension = 16;
    std::vector<float> data;
    generate_test_data(data, num_vectors, dimension);

    svs_error_h error = svs_error_create();
    svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
    CATCH_REQUIRE(algorithm != nullptr);
    svs_index_builder_h builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, dimension, algorithm, error
    );
    CATCH_REQUIRE(builder != nullptr);
    CATCH_REQUIRE(
        svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_NATIVE, 2, error)
    );
    svs_index_h index = svs_index_build(builder, data.data(), num_vectors, error);
    CATCH_REQUIRE(index != nullptr);
    CATCH_REQUIRE(svs_error_ok(error));

    svs_index_free(index);
    svs_index_builder_free(builder);
    svs_algorithm_free(algorithm);
    svs_error_free(error);
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    return std::string(
        std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()
    );
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::trunc);
    out << content;
}

} // namespace

CATCH_TEST_CASE("C API Logger Handle", "[c_api][logging]") {
    CATCH_SECTION("Create With Each Kind") {
        svs_error_h error = svs_error_create();
        TempDir tmp;
        const std::string path = (tmp.path() / "svs.log").string();

        for (auto kind :
             {SVS_LOGGING_KIND_NONE, SVS_LOGGING_KIND_STDOUT, SVS_LOGGING_KIND_STDERR}) {
            // The path is ignored by these kinds.
            svs_logger_h logger = svs_logger_create(kind, nullptr, error);
            CATCH_REQUIRE(logger != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));
            svs_logger_free(logger);

            logger = svs_logger_create(kind, "ignored", error);
            CATCH_REQUIRE(logger != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));
            svs_logger_free(logger);
        }

        for (auto kind : {SVS_LOGGING_KIND_FILE_APPEND, SVS_LOGGING_KIND_FILE_TRUNCATE}) {
            svs_logger_h logger = svs_logger_create(kind, path.c_str(), error);
            CATCH_REQUIRE(logger != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));
            svs_logger_free(logger);
        }

        // NULL error handle and freeing NULL are allowed.
        svs_logger_h logger = svs_logger_create(SVS_LOGGING_KIND_NONE, nullptr, nullptr);
        CATCH_REQUIRE(logger != nullptr);
        svs_logger_free(logger);
        svs_logger_free(nullptr);
        svs_error_free(error);
    }

    CATCH_SECTION("Create Invalid") {
        svs_error_h error = svs_error_create();

        // Custom output must go through svs_logger_create_custom.
        CATCH_REQUIRE(
            svs_logger_create(SVS_LOGGING_KIND_CUSTOM, nullptr, error) == nullptr
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE(
            std::string(svs_error_get_message(error)).find("svs_logger_create_custom") !=
            std::string::npos
        );

        CATCH_REQUIRE(
            svs_logger_create(static_cast<svs_logging_kind_t>(6), nullptr, error) == nullptr
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        // File kinds need a non-empty path.
        for (auto kind : {SVS_LOGGING_KIND_FILE_APPEND, SVS_LOGGING_KIND_FILE_TRUNCATE}) {
            CATCH_REQUIRE(svs_logger_create(kind, nullptr, error) == nullptr);
            CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
            CATCH_REQUIRE(svs_logger_create(kind, "", error) == nullptr);
            CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        }

        // A path that cannot be opened as a file (an existing directory).
        TempDir tmp;
        CATCH_REQUIRE(
            svs_logger_create(
                SVS_LOGGING_KIND_FILE_TRUNCATE, tmp.string().c_str(), error
            ) == nullptr
        );
        CATCH_REQUIRE(svs_error_get_code(error) != SVS_OK);

        // NULL error handle: failure is still reported by the return value.
        CATCH_REQUIRE(
            svs_logger_create(SVS_LOGGING_KIND_FILE_APPEND, nullptr, nullptr) == nullptr
        );

        svs_error_free(error);
    }

    CATCH_SECTION("Create Custom") {
        svs_error_h error = svs_error_create();
        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(noop_log);
        svs_logging_t user_logger = {&ops, nullptr};

        svs_logger_h logger = svs_logger_create_custom(&user_logger, error);
        CATCH_REQUIRE(logger != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        svs_logger_free(logger);

        logger = svs_logger_create_custom(&user_logger, nullptr);
        CATCH_REQUIRE(logger != nullptr);
        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("Create Custom Invalid") {
        svs_error_h error = svs_error_create();

        auto expect_invalid = [&](svs_logging_i user_logger) {
            CATCH_REQUIRE(svs_logger_create_custom(user_logger, error) == nullptr);
            CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        };

        expect_invalid(nullptr);

        svs_logging_t no_ops = {nullptr, nullptr};
        expect_invalid(&no_ops);

        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(noop_log);
        ops.log = nullptr;
        svs_logging_t no_log = {&ops, nullptr};
        expect_invalid(&no_log);

        svs_logging_ops_t bad_version = SVS_INIT_LOGGING_OPS(noop_log);
        bad_version.version = svs_get_version() + 1;
        svs_logging_t bad_version_logger = {&bad_version, nullptr};
        expect_invalid(&bad_version_logger);

        svs_logging_ops_t bad_size = SVS_INIT_LOGGING_OPS(noop_log);
        bad_size.struct_size = sizeof(svs_logging_ops_t) - 1;
        svs_logging_t bad_size_logger = {&bad_size, nullptr};
        expect_invalid(&bad_size_logger);

        // NULL error handle: failure is still reported by the return value.
        CATCH_REQUIRE(svs_logger_create_custom(nullptr, nullptr) == nullptr);

        svs_error_free(error);
    }

    CATCH_SECTION("NULL Arguments") {
        svs_error_h error = svs_error_create();
        svs_logger_h logger = svs_logger_create(SVS_LOGGING_KIND_NONE, nullptr, error);
        CATCH_REQUIRE(logger != nullptr);

        auto expect_invalid = [&](bool result) {
            CATCH_REQUIRE(result == false);
            CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        };

        svs_log_level_t level;
        const char* pattern = nullptr;

        expect_invalid(svs_logger_set_level(nullptr, SVS_LOG_LEVEL_INFO, error));
        expect_invalid(svs_logger_get_level(nullptr, &level, error));
        expect_invalid(svs_logger_get_level(logger, nullptr, error));
        expect_invalid(svs_logger_set_pattern(nullptr, "%v", error));
        expect_invalid(svs_logger_set_pattern(logger, nullptr, error));
        expect_invalid(svs_logger_get_pattern(nullptr, &pattern, error));
        expect_invalid(svs_logger_get_pattern(logger, nullptr, error));

        // NULL error handle: failure is still reported by the return value.
        CATCH_REQUIRE(svs_logger_set_level(nullptr, SVS_LOG_LEVEL_INFO, nullptr) == false);

        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("Defaults After Create") {
        svs_error_h error = svs_error_create();
        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(noop_log);
        svs_logging_t user_logger = {&ops, nullptr};
        TempDir tmp;
        const std::string path = (tmp.path() / "svs.log").string();

        auto check_defaults = [&](svs_logger_h logger) {
            CATCH_REQUIRE(logger != nullptr);
            svs_log_level_t level = SVS_LOG_LEVEL_OFF;
            const char* pattern = nullptr;
            CATCH_REQUIRE(svs_logger_get_level(logger, &level, error));
            CATCH_REQUIRE(level == SVS_LOG_LEVEL_WARN);
            CATCH_REQUIRE(svs_logger_get_pattern(logger, &pattern, error));
            CATCH_REQUIRE(pattern != nullptr);
            CATCH_REQUIRE(std::string(pattern) == "%v");
            svs_logger_free(logger);
        };

        check_defaults(svs_logger_create(SVS_LOGGING_KIND_NONE, nullptr, error));
        check_defaults(svs_logger_create(SVS_LOGGING_KIND_STDOUT, nullptr, error));
        check_defaults(svs_logger_create(SVS_LOGGING_KIND_STDERR, nullptr, error));
        check_defaults(
            svs_logger_create(SVS_LOGGING_KIND_FILE_TRUNCATE, path.c_str(), error)
        );
        check_defaults(svs_logger_create_custom(&user_logger, error));

        svs_error_free(error);
    }

    CATCH_SECTION("Level Round Trip") {
        svs_error_h error = svs_error_create();
        svs_logger_h logger = svs_logger_create(SVS_LOGGING_KIND_NONE, nullptr, error);
        CATCH_REQUIRE(logger != nullptr);

        for (auto level :
             {SVS_LOG_LEVEL_TRACE,
              SVS_LOG_LEVEL_DEBUG,
              SVS_LOG_LEVEL_INFO,
              SVS_LOG_LEVEL_WARN,
              SVS_LOG_LEVEL_ERROR,
              SVS_LOG_LEVEL_CRITICAL,
              SVS_LOG_LEVEL_OFF}) {
            CATCH_REQUIRE(svs_logger_set_level(logger, level, error));
            svs_log_level_t out_level = SVS_LOG_LEVEL_OFF;
            CATCH_REQUIRE(svs_logger_get_level(logger, &out_level, error));
            CATCH_REQUIRE(out_level == level);
        }

        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("Pattern Round Trip") {
        svs_error_h error = svs_error_create();
        svs_logger_h logger = svs_logger_create(SVS_LOGGING_KIND_NONE, nullptr, error);
        CATCH_REQUIRE(logger != nullptr);

        const char* pattern = nullptr;
        CATCH_REQUIRE(svs_logger_get_pattern(logger, &pattern, error));
        CATCH_REQUIRE(pattern != nullptr);
        CATCH_REQUIRE(std::string(pattern) == "%v");

        CATCH_REQUIRE(svs_logger_set_pattern(logger, "[%l] %v", error));
        CATCH_REQUIRE(svs_logger_get_pattern(logger, &pattern, error));
        CATCH_REQUIRE(std::string(pattern) == "[%l] %v");

        svs_logger_free(logger);
        svs_error_free(error);
    }
}

CATCH_TEST_CASE("C API Logger Output", "[c_api][logging]") {
    CATCH_SECTION("Custom Callback Receives Messages") {
        LogRecorder recorder; // Declared first: must outlive the guard.
        DefaultLoggerGuard guard;
        svs_error_h error = svs_error_create();

        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(record_log);
        svs_logging_t user_logger = {&ops, &recorder};
        svs_logger_h logger = svs_logger_create_custom(&user_logger, error);
        CATCH_REQUIRE(logger != nullptr);
        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
        CATCH_REQUIRE(svs_set_default_logger(logger, error));
        CATCH_REQUIRE(svs_error_ok(error));

        build_small_index();

        CATCH_REQUIRE(recorder.contains(SVS_LOG_LEVEL_TRACE, "Number of syncs"));
        // Default pattern "%v": bare message, no timestamp/level prefix, no newline.
        {
            std::lock_guard lock{recorder.mutex};
            for (const auto& [level, text] : recorder.messages) {
                CATCH_REQUIRE(!text.empty());
                CATCH_REQUIRE(text.front() != '[');
                CATCH_REQUIRE(text.back() != '\n');
            }
        }

        CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("None Output Is Silent") {
        DefaultLoggerGuard guard;
        svs_error_h error = svs_error_create();
        svs_logger_h logger = svs_logger_create(SVS_LOGGING_KIND_NONE, nullptr, error);
        CATCH_REQUIRE(logger != nullptr);

        // No output: using it as default (even at TRACE) writes nothing and must not
        // crash.
        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
        CATCH_REQUIRE(svs_set_default_logger(logger, error));
        build_small_index();
        CATCH_REQUIRE(svs_error_ok(error));

        CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("Level Changes Reach Existing Users") {
        LogRecorder recorder; // Declared first: must outlive the guard.
        DefaultLoggerGuard guard;
        svs_error_h error = svs_error_create();

        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(record_log);
        svs_logging_t user_logger = {&ops, &recorder};
        svs_logger_h logger = svs_logger_create_custom(&user_logger, error);
        CATCH_REQUIRE(logger != nullptr);
        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
        CATCH_REQUIRE(svs_set_default_logger(logger, error));

        build_small_index();
        CATCH_REQUIRE(recorder.contains(SVS_LOG_LEVEL_TRACE, "Number of syncs"));

        // Level changes apply without calling svs_set_default_logger again.
        {
            std::lock_guard lock{recorder.mutex};
            recorder.messages.clear();
        }
        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_OFF, error));
        build_small_index();
        {
            std::lock_guard lock{recorder.mutex};
            CATCH_REQUIRE(recorder.messages.empty());
        }

        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
        build_small_index();
        CATCH_REQUIRE(recorder.contains(SVS_LOG_LEVEL_TRACE, "Number of syncs"));

        CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("Pattern Not Applied To Custom Callback") {
        LogRecorder recorder; // Declared first: must outlive the guard.
        DefaultLoggerGuard guard;
        svs_error_h error = svs_error_create();

        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(record_log);
        svs_logging_t user_logger = {&ops, &recorder};
        svs_logger_h logger = svs_logger_create_custom(&user_logger, error);
        CATCH_REQUIRE(logger != nullptr);
        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
        CATCH_REQUIRE(svs_set_default_logger(logger, error));
        CATCH_REQUIRE(svs_logger_set_pattern(logger, "[x] %v", error));

        build_small_index();

        {
            std::lock_guard lock{recorder.mutex};
            CATCH_REQUIRE(!recorder.messages.empty());
            for (const auto& [level, text] : recorder.messages) {
                CATCH_REQUIRE(text.rfind("[x] ", 0) == std::string::npos);
            }
            bool found = std::any_of(
                recorder.messages.begin(),
                recorder.messages.end(),
                [](const auto& m) { return m.second.rfind("Number of syncs: ", 0) == 0; }
            );
            CATCH_REQUIRE(found);
        }

        CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("Level Filtering") {
        LogRecorder recorder; // Declared first: must outlive the guard.
        DefaultLoggerGuard guard;
        svs_error_h error = svs_error_create();

        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(record_log);
        svs_logging_t user_logger = {&ops, &recorder};
        svs_logger_h logger = svs_logger_create_custom(&user_logger, error);
        CATCH_REQUIRE(logger != nullptr);
        // Default level is WARN: the TRACE/DEBUG build messages must be filtered.
        CATCH_REQUIRE(svs_set_default_logger(logger, error));
        build_small_index();
        CATCH_REQUIRE(!recorder.any_below(SVS_LOG_LEVEL_WARN));

        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_INFO, error));
        build_small_index();
        CATCH_REQUIRE(!recorder.any_below(SVS_LOG_LEVEL_INFO));

        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
        build_small_index();
        CATCH_REQUIRE(recorder.contains(SVS_LOG_LEVEL_TRACE, "Number of syncs"));

        CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("Pattern Applied To File Output") {
        TempDir tmp;
        const std::string path = (tmp.path() / "svs.log").string();
        {
            DefaultLoggerGuard guard;
            svs_error_h error = svs_error_create();
            svs_logger_h logger =
                svs_logger_create(SVS_LOGGING_KIND_FILE_TRUNCATE, path.c_str(), error);
            CATCH_REQUIRE(logger != nullptr);
            CATCH_REQUIRE(svs_logger_set_pattern(logger, "[x] %v", error));
            CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
            CATCH_REQUIRE(svs_set_default_logger(logger, error));
            svs_logger_free(logger);

            build_small_index();
            CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
            svs_error_free(error);
        }
        // Resetting the default logger released the last reference, which closes (and
        // flushes) the file.
        auto content = read_file(path);
        CATCH_REQUIRE(content.find("[x] Number of syncs") != std::string::npos);
    }

    CATCH_SECTION("Default Logger Outlives Handle") {
        LogRecorder recorder; // Declared first: must outlive the guard.
        DefaultLoggerGuard guard;
        svs_error_h error = svs_error_create();

        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(record_log);
        svs_logging_t user_logger = {&ops, &recorder};
        svs_logger_h logger = svs_logger_create_custom(&user_logger, error);
        CATCH_REQUIRE(logger != nullptr);
        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
        CATCH_REQUIRE(svs_set_default_logger(logger, error));
        svs_logger_free(logger);

        build_small_index();

        CATCH_REQUIRE(recorder.contains(SVS_LOG_LEVEL_TRACE, "Number of syncs"));
        CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
        svs_error_free(error);
    }

    CATCH_SECTION("Reset Default Logger With NULL") {
        LogRecorder recorder; // Declared first: must outlive the guard.
        DefaultLoggerGuard guard;
        svs_error_h error = svs_error_create();

        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(record_log);
        svs_logging_t user_logger = {&ops, &recorder};
        svs_logger_h logger = svs_logger_create_custom(&user_logger, error);
        CATCH_REQUIRE(logger != nullptr);
        CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
        CATCH_REQUIRE(svs_set_default_logger(logger, error));

        build_small_index();
        CATCH_REQUIRE(recorder.contains(SVS_LOG_LEVEL_TRACE, "Number of syncs"));

        // NULL restores the built-in default; the callback must no longer be called.
        CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
        CATCH_REQUIRE(svs_error_ok(error));
        {
            std::lock_guard lock{recorder.mutex};
            recorder.messages.clear();
        }
        build_small_index();
        {
            std::lock_guard lock{recorder.mutex};
            CATCH_REQUIRE(recorder.messages.empty());
        }

        svs_logger_free(logger);
        svs_error_free(error);
    }

    CATCH_SECTION("Switch Default Logger To Another Logger") {
        LogRecorder first;  // Declared first: must outlive the guard.
        LogRecorder second; // Declared first: must outlive the guard.
        DefaultLoggerGuard guard;
        svs_error_h error = svs_error_create();

        svs_logging_ops_t ops = SVS_INIT_LOGGING_OPS(record_log);
        svs_logging_t first_logger = {&ops, &first};
        svs_logging_t second_logger = {&ops, &second};
        svs_logger_h logger_a = svs_logger_create_custom(&first_logger, error);
        svs_logger_h logger_b = svs_logger_create_custom(&second_logger, error);
        CATCH_REQUIRE(logger_a != nullptr);
        CATCH_REQUIRE(logger_b != nullptr);
        CATCH_REQUIRE(svs_logger_set_level(logger_a, SVS_LOG_LEVEL_TRACE, error));
        CATCH_REQUIRE(svs_logger_set_level(logger_b, SVS_LOG_LEVEL_TRACE, error));

        CATCH_REQUIRE(svs_set_default_logger(logger_a, error));
        build_small_index();
        CATCH_REQUIRE(first.contains(SVS_LOG_LEVEL_TRACE, "Number of syncs"));
        {
            std::lock_guard lock{second.mutex};
            CATCH_REQUIRE(second.messages.empty());
        }

        // The output of a logger cannot change; to log elsewhere, install another logger.
        {
            std::lock_guard lock{first.mutex};
            first.messages.clear();
        }
        CATCH_REQUIRE(svs_set_default_logger(logger_b, error));
        build_small_index();
        CATCH_REQUIRE(second.contains(SVS_LOG_LEVEL_TRACE, "Number of syncs"));
        {
            std::lock_guard lock{first.mutex};
            CATCH_REQUIRE(first.messages.empty());
        }

        CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
        svs_logger_free(logger_a);
        svs_logger_free(logger_b);
        svs_error_free(error);
    }

    CATCH_SECTION("File Truncate") {
        TempDir tmp;
        const std::string path = (tmp.path() / "svs.log").string();
        write_file(path, "OLD CONTENT\n");
        {
            DefaultLoggerGuard guard;
            svs_error_h error = svs_error_create();
            svs_logger_h logger =
                svs_logger_create(SVS_LOGGING_KIND_FILE_TRUNCATE, path.c_str(), error);
            CATCH_REQUIRE(logger != nullptr);
            CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
            CATCH_REQUIRE(svs_set_default_logger(logger, error));
            svs_logger_free(logger);

            build_small_index();
            CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
            svs_error_free(error);
        }
        auto content = read_file(path);
        CATCH_REQUIRE(content.find("OLD CONTENT") == std::string::npos);
        CATCH_REQUIRE(content.find("Number of syncs") != std::string::npos);
    }

    CATCH_SECTION("File Append") {
        TempDir tmp;
        const std::string path = (tmp.path() / "svs.log").string();
        write_file(path, "OLD CONTENT\n");
        {
            DefaultLoggerGuard guard;
            svs_error_h error = svs_error_create();
            svs_logger_h logger =
                svs_logger_create(SVS_LOGGING_KIND_FILE_APPEND, path.c_str(), error);
            CATCH_REQUIRE(logger != nullptr);
            CATCH_REQUIRE(svs_logger_set_level(logger, SVS_LOG_LEVEL_TRACE, error));
            CATCH_REQUIRE(svs_set_default_logger(logger, error));
            svs_logger_free(logger);

            build_small_index();
            CATCH_REQUIRE(svs_set_default_logger(nullptr, error));
            svs_error_free(error);
        }
        auto content = read_file(path);
        CATCH_REQUIRE(content.rfind("OLD CONTENT\n", 0) == 0);
        CATCH_REQUIRE(content.find("Number of syncs") != std::string::npos);
    }
}
