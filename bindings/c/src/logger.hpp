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
#include "spdlog/sinks/dist_sink.h"

#include <stdexcept>

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

/// Default pattern of every logger handle: the bare message. Patterns apply to the
/// stdout, stderr and file outputs; custom callbacks always receive the bare message.
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

} // namespace c_runtime
} // namespace svs
