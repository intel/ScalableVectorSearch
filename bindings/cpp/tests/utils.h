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

inline std::filesystem::path temp_directory() {
    // Use /tmp for runtime binding tests
    return std::filesystem::path("/tmp/svs_runtime_test");
}

inline bool cleanup_temp_directory() {
    return std::filesystem::remove_all(temp_directory());
}

inline bool make_temp_directory() {
    return std::filesystem::create_directories(temp_directory());
}

inline bool prepare_temp_directory() {
    cleanup_temp_directory();
    return make_temp_directory();
}

inline std::filesystem::path prepare_temp_directory_v2() {
    cleanup_temp_directory();
    make_temp_directory();
    return temp_directory();
}

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
