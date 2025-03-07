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

#pragma once

// svs
#include "svs/lib/concurrency/readwrite_protected.h"
#include "svs/lib/exception.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/timing.h"
#include "svs/third-party/fmt.h"

// third-party
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/null_sink.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/spdlog.h"

// stl
#include <bit>
#include <cstdlib>
#include <memory>
#include <type_traits>

namespace svs::logging {

/// @brief Enum controlling verbosity.
///
/// These definitions mirror those defined in "spdlog"
enum class Level {
    Trace = SPDLOG_LEVEL_TRACE,
    Debug = SPDLOG_LEVEL_DEBUG,
    Info = SPDLOG_LEVEL_INFO,
    Warn = SPDLOG_LEVEL_WARN,
    Error = SPDLOG_LEVEL_ERROR,
    Critical = SPDLOG_LEVEL_CRITICAL,
    Off = SPDLOG_LEVEL_OFF
};
static_assert(std::is_same_v<
              std::underlying_type_t<Level>,
              std::underlying_type_t<::spdlog::level::level_enum>>);

inline constexpr std::array<Level, 7> all_levels = {
    Level::Trace,
    Level::Debug,
    Level::Info,
    Level::Warn,
    Level::Error,
    Level::Critical,
    Level::Off};

/// @brief The type of the global logger.
using logger_ptr = std::shared_ptr<::spdlog::logger>;

/// @brief The type for sinks registered with loggers.
using sink_ptr = std::shared_ptr<::spdlog::sinks::sink>;

/// @brief A sink going nowhere. Used to disable logging entirely.
inline sink_ptr null_sink() { return std::make_shared<::spdlog::sinks::null_sink_mt>(); }

/// @brief A sink printing to `stdout`.
inline sink_ptr stdout_sink() {
    return std::make_shared<::spdlog::sinks::stdout_sink_mt>();
}

/// @brief A sink printing to `stderr`.
inline sink_ptr stderr_sink() {
    return std::make_shared<::spdlog::sinks::stderr_sink_mt>();
}

static_assert(
    std::is_same_v<::spdlog::filename_t, std::string>,
    "SVS and spdlog disagree on the type of `::spdlog::filename_t`"
);

/// @brief A sink writing logging message to a file.
///
/// This function uses `spdlog` to create and open the log file.
/// As such, `spdlog` will make the full path to the log file, creating intermediate
/// directories as needed. If the process lacks sufficient permissions to create the path,
/// then an exception is thrown at creation time.
inline sink_ptr file_sink(const std::string& filename, bool truncate = true) {
    return std::make_shared<::spdlog::sinks::basic_file_sink_mt>(filename, truncate);
}

namespace detail {

// Map SVS levels to SPDLOG levels.
constexpr spdlog::level::level_enum to_spdlog(logging::Level level) {
    return std::bit_cast<spdlog::level::level_enum>(level);
}

constexpr Level from_spdlog(spdlog::level::level_enum level) {
    return std::bit_cast<Level>(level);
}

// Defaults
inline constexpr logging::Level default_logging_level = Level::Warn;
inline sink_ptr default_sink() { return svs::logging::stdout_sink(); }

// Utils
struct ToLower {
    // `std::tolower` is locale-dependent and undefined for `chars`.
    // This implementation mirrors that in `spdlog`.
    constexpr char operator()(char c) const {
        return static_cast<char>((c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c);
    }
};

inline constexpr ToLower to_lower{};

inline void make_lower_case(std::string& str) {
    std::transform(str.begin(), str.end(), str.begin(), to_lower);
}

// Check for a `SVS_LOG_LEVEL` environment variable.
// If it exists, try to parse it as one of our pre-defined levels.
inline Level level_from_string(std::string str) {
    // Use the spdlog API to parse the level.
    make_lower_case(str);
    return from_spdlog(spdlog::level::from_str(str));
}

// The maximum size of environment variable strings we will accept.
constexpr size_t max_environment_variable_length() { return 1000; }

// Environmental variable name to set the log level.
constexpr const char* log_level_var_name() { return "SVS_LOG_LEVEL"; }

// Environmental variable name to set the log sink.
constexpr const char* log_sink_var_name() { return "SVS_LOG_SINK"; }

// Don't process excessively long environment variable values.
// Prefer to throw an exception rather than fail silently.
inline void check_max_length(const char* varname, size_t length) {
    constexpr size_t max_length = max_environment_variable_length();
    if (length > max_length) {
        throw ANNEXCEPTION(
            "Provided value for {} exceeds maximum length of {}.", varname, max_length
        );
    }
}

// Get the logging level from the SVS_LOG_LEVEL
inline Level get_level_from_environment() {
    constexpr const char* varname = log_level_var_name();
    const char* str = ::std::getenv(varname);
    if (str == nullptr) {
        return default_logging_level;
    }

    const size_t strlen = std::strlen(str);
    check_max_length(varname, strlen);
    return level_from_string(std::string(str, strlen));
}

inline sink_ptr get_sink_from_environment() {
    constexpr const char* varname = log_sink_var_name();
    const char* c_str = std::getenv(varname);
    if (c_str == nullptr) {
        return default_sink();
    }

    const size_t strlen = std::strlen(c_str);
    check_max_length(varname, strlen);

    auto str = std::string(c_str, strlen);
    make_lower_case(str);

    // Handle recognized pre-configured sinks.
    if (str == "null" || str == "none") {
        return null_sink();
    }
    if (str == "stdout" || str.empty()) {
        return stdout_sink();
    }
    if (str == "stderr") {
        return stderr_sink();
    }

    // Handle cases like "file:/the/file/path"
    if (str.starts_with("file:")) {
        // The originally created string was canonicalized to lowercase.
        // We need to re-create it to restore the original formatting.
        // TODO: Since we know ahead-of-time the maximum length of the expected strings,
        // we could use a smaller string.
        str = std::string(c_str, strlen);

        // Since we control `to_lower`, lower casing ':' should be the identity.
        assert(str.at(4) == ':');
        return logging::file_sink(str.substr(5));
    }

    // No special cases handled.
    // Use the default sink.
    return default_sink();
}

// Initialization for the default logger.
// By default, logging goes to `stdout`.
inline logger_ptr default_logger_(sink_ptr sink, logging::Level level) {
    auto logger = std::make_shared<::spdlog::logger>("default", std::move(sink));
    logger->set_level(to_spdlog(level));
    return logger;
}

// Initial logger configuration.
SVS_VALIDATE_BOOL_ENV(SVS_INITIALIZE_LOGGER);
#if SVS_INITIALIZE_LOGGER
inline logger_ptr default_logger() {
    return default_logger_(get_sink_from_environment(), get_level_from_environment());
}
#else
inline logger_ptr default_logger() { return default_logger_(null_sink(), Level::Off); }
#endif

// Opt-in to slightly less code repitition for function definition by providing compile-time
// transformation from string literals to the level enum.
consteval svs::logging::Level parselevel(std::string_view str) {
    if (str == "trace") {
        return Level::Trace;
    }
    if (str == "debug") {
        return Level::Debug;
    }
    if (str == "info") {
        return Level::Info;
    }
    if (str == "warn") {
        return Level::Warn;
    }
    if (str == "error") {
        return Level::Error;
    }
    if (str == "critical") {
        return Level::Critical;
    }
    // Custom mapping of "must_log" to "Off".
    if (str == "must_log") {
        return Level::Off;
    }
    throw ANNEXCEPTION("Unhandled string {}\n", str);
}

} // namespace detail

// Thread-safe container for the global logger.
inline ::svs::lib::ReadWriteProtected<logger_ptr>& global_logger() {
    static ::svs::lib::ReadWriteProtected<logger_ptr> logger = detail::default_logger();
    return logger;
}

/// @brief Return a shared pointer to the current global logger.
///
/// This function is safe to call in a multi-threaded context.
inline logger_ptr get() { return global_logger().get(); }

/// @brief Override the currently configured logger.
///
/// @param logger A shared pointer to any ``spdlog::logger``.
///
/// This function is safe to call in a multi-threaded contex but it is the user's
/// responsibility to ensure that all sinks registered with the logger are multi-thread
/// safe.
inline void set(const logger_ptr& logger) { global_logger().set(logger); }

/// @copydoc set(const logger_ptr&)
inline void set(logger_ptr&& logger) { global_logger().set(std::move(logger)); }

/// @brief Reset the logger to its default.
inline void reset_to_default() { set(detail::default_logger()); }

/// @brief Retrieve the logging level for the provided logger.
inline Level get_level(const logger_ptr& logger) {
    return detail::from_spdlog(logger->level());
}

/// @brief Retrieve the logging level for the global logger.
inline Level get_level() { return get_level(get()); }

/// @brief Set the logging level for the provided logger.
inline void set_level(logger_ptr& logger, Level level) {
    logger->set_level(detail::to_spdlog(level));
}

/// @brief Set the logging level for the global logger.
inline void set_level(Level level) {
    auto logger = logging::get();
    set_level(logger, level);
}

/// @brief Return whether a message should be created for the logger at the given level.
inline bool should_log(const logger_ptr& logger, logging::Level level) {
    return logger->should_log(detail::to_spdlog(level));
}

/// @brief Send a message to the logger at the given logging level.
///
/// Materialization of the logging message will be deferred until the level has been
/// checked.
///
/// Convenience aliases at the corresponding log level include: ``trace``, ``debug``,
/// ``info``, ``warn``, ``error``, ``critical``.
template <typename... Args>
void log(logger_ptr& logger, Level level, fmt::format_string<Args...> fmt, Args&&... args) {
    logger->log(detail::to_spdlog(level), fmt, SVS_FWD(args)...);
}

/// @brief Send a message to the global logger at the given logging level.
///
/// @copydetails log
template <typename... Args>
void log(Level level, fmt::format_string<Args...> fmt, Args&&... args) {
    auto global_logger = logging::get();
    logging::log(global_logger, level, fmt, SVS_FWD(args)...);
}

// Convenience methods
#define SVS_DEFINE_LOG_FUNCTION(name)                                      \
    template <typename... Args>                                            \
    void name(                                                             \
        ::svs::logging::logger_ptr& logger,                                \
        fmt::format_string<Args...> fmt,                                   \
        Args&&... args                                                     \
    ) {                                                                    \
        constexpr ::svs::logging::Level level = detail::parselevel(#name); \
        return ::svs::logging::log(logger, level, fmt, SVS_FWD(args)...);  \
    }                                                                      \
                                                                           \
    template <typename... Args>                                            \
    void name(fmt::format_string<Args...> fmt, Args&&... args) {           \
        constexpr ::svs::logging::Level level = detail::parselevel(#name); \
        return ::svs::logging::log(level, fmt, SVS_FWD(args)...);          \
    }

SVS_DEFINE_LOG_FUNCTION(trace);
SVS_DEFINE_LOG_FUNCTION(debug);
SVS_DEFINE_LOG_FUNCTION(info);
SVS_DEFINE_LOG_FUNCTION(warn);
SVS_DEFINE_LOG_FUNCTION(error);
SVS_DEFINE_LOG_FUNCTION(critical);
SVS_DEFINE_LOG_FUNCTION(must_log);

#undef SVS_DEFINE_LOG_FUNCTION

} // namespace svs::logging
