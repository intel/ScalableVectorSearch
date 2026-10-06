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
#pragma once

#include "svs/c/svs_c.h"

#include "error.hpp"

#include <svs/core/logging.h>

#include "spdlog/sinks/callback_sink.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace svs {
namespace c_runtime {

// spdlog level values are passed to the user callback by a plain cast.
static_assert(static_cast<int>(SVS_LOG_LEVEL_TRACE) == SPDLOG_LEVEL_TRACE);
static_assert(static_cast<int>(SVS_LOG_LEVEL_DEBUG) == SPDLOG_LEVEL_DEBUG);
static_assert(static_cast<int>(SVS_LOG_LEVEL_INFO) == SPDLOG_LEVEL_INFO);
static_assert(static_cast<int>(SVS_LOG_LEVEL_WARN) == SPDLOG_LEVEL_WARN);
static_assert(static_cast<int>(SVS_LOG_LEVEL_ERROR) == SPDLOG_LEVEL_ERROR);
static_assert(static_cast<int>(SVS_LOG_LEVEL_CRITICAL) == SPDLOG_LEVEL_CRITICAL);
static_assert(static_cast<int>(SVS_LOG_LEVEL_OFF) == SPDLOG_LEVEL_OFF);

/// Name given to every spdlog logger created by the C API.
inline constexpr const char* logger_name = "svs_c";

/// Default pattern of every logger handle: the bare message.
inline constexpr const char* default_log_pattern = "%v";

/// Checks the version, struct size and NULL pointers of a user-provided custom logger.
inline void validate_custom_logger(const svs_logging_i user_logger) {
    if (user_logger == nullptr) {
        throw std::invalid_argument("Custom logger pointer cannot be null.");
    }
    if (user_logger->ops == nullptr) {
        throw std::invalid_argument("Custom logger interface is not initialized.");
    }
    if (user_logger->ops->version > svs_get_version()) {
        throw std::invalid_argument("Custom logger interface version is not supported.");
    }
    if (user_logger->ops->struct_size < sizeof(svs_logging_ops_t)) {
        throw std::invalid_argument("Incompatible custom logger interface struct size.");
    }
    if (user_logger->ops->log == nullptr) {
        throw std::invalid_argument("Custom logger interface has null log function.");
    }
}

/// Maps a C API log level to the SVS logging level. Unknown values map to Info.
inline svs::logging::Level to_logging_level(svs_log_level_t level) {
    switch (level) {
        case SVS_LOG_LEVEL_TRACE:
            return svs::logging::Level::Trace;
        case SVS_LOG_LEVEL_DEBUG:
            return svs::logging::Level::Debug;
        case SVS_LOG_LEVEL_INFO:
            return svs::logging::Level::Info;
        case SVS_LOG_LEVEL_WARN:
            return svs::logging::Level::Warn;
        case SVS_LOG_LEVEL_ERROR:
            return svs::logging::Level::Error;
        case SVS_LOG_LEVEL_CRITICAL:
            return svs::logging::Level::Critical;
        case SVS_LOG_LEVEL_OFF:
            return svs::logging::Level::Off;
        default:
            return svs::logging::Level::Info;
    }
}

/// Creates the spdlog sink for a built-in output kind.
inline svs::logging::sink_ptr make_sink(svs_logging_kind_t kind, const char* path) {
    switch (kind) {
        case SVS_LOGGING_KIND_NONE:
            return svs::logging::null_sink();
        case SVS_LOGGING_KIND_STDOUT:
            return svs::logging::stdout_sink();
        case SVS_LOGGING_KIND_STDERR:
            return svs::logging::stderr_sink();
        case SVS_LOGGING_KIND_FILE_APPEND:
        case SVS_LOGGING_KIND_FILE_TRUNCATE:
            if (path == nullptr || *path == '\0') {
                throw std::invalid_argument(
                    "File path must be provided for file logging kind"
                );
            }
            return svs::logging::file_sink(path, kind == SVS_LOGGING_KIND_FILE_TRUNCATE);
        default:
            throw std::invalid_argument("Invalid logging kind");
    }
}

/// Creates the spdlog sink that forwards every message to a user callback.
inline svs::logging::sink_ptr make_custom_sink(const svs_logging_i user_logger) {
    validate_custom_logger(user_logger);
    auto log = user_logger->ops->log;
    auto self = user_logger->self;
    return std::make_shared<spdlog::sinks::callback_sink_mt>(
        [log, self](const spdlog::details::log_msg& msg) {
            std::string text(msg.payload.data(), msg.payload.size());
            log(self, static_cast<svs_log_level_t>(msg.level), text.c_str());
        }
    );
}

} // namespace c_runtime
} // namespace svs

/// The logger handle of the C API (svs_logger_h).
struct svs_logger {
    svs::logging::logger_ptr impl;
    std::string pattern = svs::c_runtime::default_log_pattern;

    explicit svs_logger(svs::logging::sink_ptr sink)
        : impl{std::make_shared<spdlog::logger>(
              svs::c_runtime::logger_name, std::move(sink)
          )} {
        impl->set_level(spdlog::level::warn);
        impl->set_pattern(pattern);
    }
};
