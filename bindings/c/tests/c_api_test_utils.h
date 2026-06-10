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

#include <filesystem>
#include <stdexcept>
#include <string>

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
        if (!path_.empty() && std::filesystem::exists(path_)) {
            std::filesystem::remove_all(path_);
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
            if (!path_.empty() && std::filesystem::exists(path_)) {
                std::filesystem::remove_all(path_);
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
