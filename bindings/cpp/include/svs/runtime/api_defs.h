/*
 * Copyright 2025 Intel Corporation
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

#include <svs/runtime/version.h>

#include <cstdint>
#include <span>

#ifdef svs_runtime_EXPORTS
#define SVS_RUNTIME_API __attribute__((visibility("default")))
#define SVS_RUNTIME_API_INTERFACE // reserved for future use
#else
#define SVS_RUNTIME_API
#define SVS_RUNTIME_API_INTERFACE // reserved for future use
#endif

namespace svs {
namespace runtime {
namespace v0 {

enum class MetricType { L2, INNER_PRODUCT };

enum class StorageKind {
    FP32,
    FP16,
    SQI8,
    LVQ4x0,
    LVQ4x4,
    LVQ4x8,
    LeanVec4x4,
    LeanVec4x8,
    LeanVec8x8,
};

enum class ErrorCode {
    SUCCESS = 0,
    UNKNOWN_ERROR = 1,
    INVALID_ARGUMENT = 2,
    NOT_IMPLEMENTED = 3,
    NOT_INITIALIZED = 4,
    RUNTIME_ERROR = 5
};

struct SVS_RUNTIME_API Status {
    constexpr Status(ErrorCode c = ErrorCode::SUCCESS, const char* msg = nullptr)
        : code(c)
        , message_storage_(nullptr) {
        if (msg != nullptr) {
            store_message(msg);
        }
    }

    constexpr ~Status() noexcept {
        if (message_storage_ != nullptr) {
            destroy_message();
        }
    }

    ErrorCode code = ErrorCode::SUCCESS;
    const char* message() const { return message_storage_ ? message_storage_ : ""; };
    constexpr bool ok() const { return code == ErrorCode::SUCCESS; }

  private:
    void store_message(const char* msg) noexcept;
    void destroy_message() noexcept;
    char* message_storage_ = nullptr;
};

constexpr Status Status_Ok{};

struct SVS_RUNTIME_API_INTERFACE IDFilter {
    virtual bool is_member(size_t id) const = 0;
    virtual ~IDFilter() = default;

    // Helper method to allow using IDFilter instances as callable objects
    bool operator()(size_t id) const { return this->is_member(id); }
};

struct SearchResultsStorage {
    std::span<int64_t> labels; // faiss::idx_t is int64_t
    std::span<float> distances;
};

struct SVS_RUNTIME_API_INTERFACE ResultsAllocator {
    virtual SearchResultsStorage allocate(std::span<size_t> result_counts) const = 0;
    virtual ~ResultsAllocator() = default;

    // Helper method to allow using ResultsAllocator instances as callable objects
    SearchResultsStorage operator()(std::span<size_t> result_counts) const {
        return this->allocate(result_counts);
    }
};

} // namespace v0
} // namespace runtime
} // namespace svs