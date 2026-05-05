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

#include "svs/runtime/api_defs.h"

#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace svs_test {

/////
///// File System
/////

namespace detail {
struct TempDirectory {
    std::filesystem::path path;

    explicit TempDirectory(const std::string& prefix)
        : path{create_unique_temp_directory(prefix)} {}

    ~TempDirectory() noexcept {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        // Ignore errors in cleanup.
    }

    std::filesystem::path get() const { return path; }
    operator const std::filesystem::path&() const { return path; }

    std::filesystem::path operator/(const std::string& subpath) const {
        return path / subpath;
    }

    static std::filesystem::path create_unique_temp_directory(const std::string& prefix) {
        namespace fs = std::filesystem;
        auto temp_dir = fs::temp_directory_path();
        constexpr int hex_mask = 0xFFFFFF; // 6 hex digits is enough.
        // Try up to 10 times to create a unique directory.
        for (int i = 0; i < 10; ++i) {
            auto random_hex = std::to_string(std::rand() & hex_mask);
            auto dir = temp_dir / (prefix + "-" + random_hex);
            if (std::filesystem::create_directories(dir)) {
                return dir;
            }
        }
        throw std::runtime_error("Could not create a unique temporary directory!");
    }
};
} // namespace detail

inline detail::TempDirectory temp_directory() {
    // Use /tmp for runtime binding tests
    return detail::TempDirectory("svs_runtime_test");
}

inline bool prepare_temp_directory() { return true; }

inline detail::TempDirectory prepare_temp_directory_v2() { return temp_directory(); }

} // namespace svs_test

// Test utility functions
namespace test_utils {

// Simple ID filter implementation for testing
class IDFilterRange : public svs::runtime::v0::IDFilter {
  private:
    size_t min_id_;
    size_t max_id_;

  public:
    IDFilterRange(size_t min_id, size_t max_id)
        : min_id_(min_id)
        , max_id_(max_id) {}

    bool is_member(size_t id) const override { return id >= min_id_ && id < max_id_; }
};

// Custom results allocator for testing
class TestResultsAllocator : public svs::runtime::v0::ResultsAllocator {
  private:
    mutable std::vector<size_t> labels_;
    mutable std::vector<float> distances_;

  public:
    svs::runtime::v0::SearchResultsStorage allocate(std::span<size_t> result_counts
    ) const override {
        size_t total_results = 0;
        for (size_t count : result_counts) {
            total_results += count;
        }

        // Resize storage
        labels_.resize(total_results);
        distances_.resize(total_results);

        return {
            std::span<size_t>(labels_.data(), total_results),
            std::span<float>(distances_.data(), total_results)};
    }

    const std::vector<size_t>& labels() const { return labels_; }
    const std::vector<float>& distances() const { return distances_; }
};

} // namespace test_utils
