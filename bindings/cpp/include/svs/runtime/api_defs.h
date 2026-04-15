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
#include <limits>
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

class OptionalBool {
    enum class Value : int8_t { Undef = -1, True = 1, False = 0 };
    Value value_;

  public:
    constexpr OptionalBool()
        : value_(Value::Undef) {}
    constexpr OptionalBool(bool b)
        : value_(b ? Value::True : Value::False) {}

    constexpr bool is_enabled() const { return value_ == Value::True; }
    constexpr bool is_disabled() const { return value_ == Value::False; }
    constexpr bool is_default() const { return value_ == Value::Undef; }

    friend constexpr bool operator==(const OptionalBool& lhs, const OptionalBool& rhs) {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr bool operator!=(const OptionalBool& lhs, const OptionalBool& rhs) {
        return lhs.value_ != rhs.value_;
    }
};

template <typename T> struct Unspecified;
template <> struct Unspecified<size_t> {
    static constexpr size_t value = std::numeric_limits<size_t>::max();
};
template <> struct Unspecified<float> {
    static constexpr float value = std::numeric_limits<float>::infinity();
};
template <> struct Unspecified<int> {
    static constexpr int value = std::numeric_limits<int>::max();
};
template <> struct Unspecified<bool> {
    static constexpr OptionalBool value = {};
};
template <> struct Unspecified<OptionalBool> {
    static constexpr OptionalBool value = {};
};

template <typename T> constexpr auto Unspecify() { return Unspecified<T>::value; }

inline bool is_specified(const OptionalBool& value) { return !value.is_default(); }

template <typename T> bool is_specified(const T& value) {
    return value != Unspecified<T>::value;
}

inline void set_if_specified(bool& target, const OptionalBool& value) {
    if (is_specified(value)) {
        target = value.is_enabled();
    }
}

template <typename T> void set_if_specified(T& target, const T& value) {
    if (is_specified(value)) {
        target = value;
    }
}

enum class MetricType { L2, INNER_PRODUCT };

enum class StorageKind {
    FP32,
    FP16,
    SQI8,
    LVQ4x0,
    LVQ8x0,
    LVQ4x4,
    LVQ4x8,
    LeanVec4x4,
    LeanVec4x8,
    LeanVec8x8,
};

/// Configuration for SSD-backed (memory-mapped) data storage.
///
/// When provided to index assembly methods, data is loaded via memory-mapped
/// files from the specified SSD path instead of being copied into heap memory.
/// This enables zero-copy loading and dramatically reduces memory usage for
/// large datasets.
///
/// For dual-component storage (LVQ4x4, LVQ4x8, LeanVec), primary and
/// secondary/residual data placement can be controlled independently.
/// When either component is on SSD, *both* use the same MMapAllocator
/// (preserving codegen) but with different eviction policies:
///   - on_ssd=true:  pages are evicted after mmap, lazily faulted on access.
///   - on_ssd=false: pages stay resident (pre-populated by kernel).
struct SSDConfig {
    /// Path to the SSD mount point (e.g., "/mnt/nvme").
    /// The runtime uses this as the base directory for memory-mapped files.
    /// Must be non-null when primary_on_ssd or secondary_on_ssd is true.
    const char* ssd_path = nullptr;

    /// Place primary (or only) data component on SSD.
    /// For LeanVec this is the reduced-dimension primary data.
    /// For dual-level LVQ (LVQ4x4, LVQ4x8) this is the primary quantized data.
    /// For single-level LVQ (LVQ4x0, LVQ8x0) this is the only data component.
    bool primary_on_ssd = false;

    /// Place secondary/residual data component on SSD.
    /// For LeanVec this is the full-dimension secondary data.
    /// For dual-level LVQ this is the residual correction data.
    /// Ignored for single-level LVQ and non-quantized types.
    bool secondary_on_ssd = false;

    /// Enable primary-only mode for LeanVec storage.
    /// When true, only the reduced-dimension primary data is used for both
    /// graph traversal and scoring (no reranking with secondary data).
    /// This reduces memory usage at the cost of recall accuracy.
    /// Ignored for non-LeanVec storage kinds.
    bool primary_only = false;

    /// Helper: true when any component should use mmap.
    bool use_mmap() const {
        return (primary_on_ssd || secondary_on_ssd) && ssd_path != nullptr;
    }
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

    Status(const Status& other)
        : code(other.code)
        , message_storage_(nullptr) {
        if (other.message_storage_ != nullptr) {
            store_message(other.message_storage_);
        }
    }

    Status& operator=(const Status& other) {
        if (this != &other) {
            code = other.code;
            if (message_storage_ != nullptr) {
                destroy_message();
            }
            message_storage_ = nullptr;
            if (other.message_storage_ != nullptr) {
                store_message(other.message_storage_);
            }
        }
        return *this;
    }

    Status(Status&& other) noexcept
        : code(other.code)
        , message_storage_(other.message_storage_) {
        other.message_storage_ = nullptr;
    }

    Status& operator=(Status&& other) noexcept {
        if (this != &other) {
            code = other.code;
            if (message_storage_ != nullptr) {
                destroy_message();
            }
            message_storage_ = other.message_storage_;
            other.message_storage_ = nullptr;
        }
        return *this;
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
    std::span<size_t> labels;
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
