// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// C API
#include "svs/c_api/svs_c.h"

// Standard library
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// RAII wrapper that creates a unique temporary directory on construction
/// and removes it (recursively) on destruction.
class TempDir {
  public:
    TempDir() {
        auto tmp = std::filesystem::temp_directory_path();
        // Create a unique directory using a template-style approach
        std::string tmpl = (tmp / "svs_test_XXXXXX").string();
        if (::mkdtemp(tmpl.data()) == nullptr) {
            throw std::runtime_error("Failed to create temporary directory");
        }
        path_ = tmpl;
    }

    ~TempDir() {
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(path_, ec);
        }
    }

    // Non-copyable
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    // Movable
    TempDir(TempDir&& other) noexcept
        : path_(std::move(other.path_)) {
        other.path_.clear();
    }
    TempDir& operator=(TempDir&& other) noexcept {
        if (this != &other) {
            if (!path_.empty()) {
                std::error_code ec;
                std::filesystem::remove_all(path_, ec);
            }
            path_ = std::move(other.path_);
            other.path_.clear();
        }
        return *this;
    }

    const std::filesystem::path& path() const { return path_; }
    std::string string() const { return path_.string(); }

    operator const std::filesystem::path&() const { return path_; }

  private:
    std::filesystem::path path_;
};

// Helper function to generate test data
inline void
generate_test_data(std::vector<float>& data, size_t num_vectors, size_t dimension) {
    data.resize(num_vectors * dimension);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((i * 7) % 100) / 100.0f;
    }
}

// Sequential threadpool for testing
inline size_t sequential_tp_size(void* /*self*/) { return 1; }

inline void sequential_tp_parallel_for(
    void* /*self*/, void (*func)(void*, size_t), void* svs_param, size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        func(svs_param, i);
    }
}

// Helper to calculate Euclidean distance
inline float euclidean_distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Helper to calculate Inner Product distance
inline float inner_product_distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Helper to calculate Cosine distance
inline float cosine_distance(const float* a, const float* b, size_t dim) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 1.0f; // Define cosine distance as 1 if either vector is zero
    }
    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

inline bool check_storage_support(svs_storage_h storage, svs_error_h error) {
    if (storage == nullptr) {
        auto code = svs_error_get_code(error);
        return code == SVS_ERROR_NOT_IMPLEMENTED || code == SVS_ERROR_UNSUPPORTED_HW;
    } else {
        return svs_error_ok(error) == true;
    }
}
