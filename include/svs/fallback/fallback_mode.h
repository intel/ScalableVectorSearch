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

namespace svs {
namespace fallback {

enum class FallbackMode { Silent, Warning, Error };

// Warn by default
inline FallbackMode mode = FallbackMode::Warning;

inline void set_mode(FallbackMode new_mode) { mode = new_mode; }
inline FallbackMode get_mode() { return mode; }

class UnsupportedHardwareError : public std::runtime_error {
  public:
    explicit UnsupportedHardwareError()
        : std::runtime_error{"LVQ and Leanvec functionality of SVS is not supported on "
                             "non-Intel hardware."} {}
};

constexpr const char* fallback_warning =
    "LVQ and Leanvec functionality of SVS is not supported on non-Intel hardware. "
    "Using uncompressed data.\n";

} // namespace fallback
} // namespace svs
