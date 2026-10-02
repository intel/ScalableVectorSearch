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

#include "spdlog/sinks/base_sink.h"
#include "spdlog/sinks/dist_sink.h"

#include <mutex>
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

/// Default pattern of every logger handle, for all output kinds: the bare message.
/// Hosts that forward messages to their own logger (custom kind) add their own timestamp
/// and level, so SVS does not add them by default.
inline constexpr const char* default_log_pattern = "%v";

/// A spdlog sink forwarding formatted messages to a user-provided C callback.
///
/// Messages are formatted with the sink's own formatter, so the logger pattern
/// (svs_logger_set_pattern) applies. The trailing end-of-line added by the formatter is
/// stripped before calling the user callback.
/// The base_sink mutex serializes calls to the user callback.
class CallbackSink final : public spdlog::sinks::base_sink<std::mutex> {
  public:
    using log_func_t = void (*)(void*, enum svs_log_level, const char*);

    static void validate(const svs_logging_i user_logger) {
        if (user_logger == nullptr) {
            throw std::invalid_argument("Custom logger pointer cannot be null.");
        }
        if (user_logger->ops == nullptr) {
            throw std::invalid_argument("Custom logger interface is not initialized.");
        }
        if (user_logger->ops->version > svs_get_version()) {
            throw std::invalid_argument("Custom logger interface version is not supported."
            );
        }
        if (user_logger->ops->struct_size < sizeof(svs_logging_ops_t)) {
            throw std::invalid_argument("Incompatible custom logger interface struct size."
            );
        }
        if (user_logger->ops->log == nullptr) {
            throw std::invalid_argument("Custom logger interface has null log function.");
        }
    }

    /// Copies the log function pointer and the user's self pointer.
    /// @p user_logger must have been checked with validate().
    explicit CallbackSink(const svs_logging_i user_logger)
        : log_{user_logger->ops->log}
        , self_{user_logger->self} {}

  protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        this->formatter_->format(msg, formatted);
        auto text = std::string(formatted.data(), formatted.size());
        // Strip the end-of-line appended by the formatter.
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
        log_(self_, static_cast<svs_log_level_t>(msg.level), text.c_str());
    }

    void flush_() override {}

  private:
    log_func_t log_;
    void* self_;
};

} // namespace c_runtime
} // namespace svs
