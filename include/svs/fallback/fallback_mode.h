/*
 * Copyright 2023 Intel Corporation
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

#ifdef USE_PROPRIETARY

#include "../../../../include/svs/cpuid.h"

#endif // USE_PROPRIETARY

namespace svs {
namespace fallback {

enum class FallbackBool { True, False, Dispatcher };

enum class FallbackReason { NoFallback, CpuId, MissingProprietary };

inline FallbackReason get_fallback_reason() {
#ifndef USE_PROPRIETARY
  return FallbackReason::MissingProprietary;
#else // USE_PROPRIETARY
  if (!svs::detail::allow_proprietary()) {
    return FallbackReason::CpuId;
  }
#endif // USE_PROPRIETARY
  return FallbackReason::NoFallback;
}

inline bool use_fallback() {
  if (get_fallback_reason() == FallbackReason::NoFallback) {
    return false;
  }
  return true;
}

enum class FallbackMode { Silent, Warning, Error };

// Warn by default
inline FallbackMode mode = FallbackMode::Warning;

inline void set_mode(FallbackMode new_mode) { mode = new_mode; }
inline FallbackMode get_mode() { return mode; }

class FallbackError : public std::runtime_error { 
public:
    explicit FallbackError(const std::string& message)
        : std::runtime_error{message} {}
};

inline void handle_fallback(FallbackMode fallback_mode, FallbackReason fallback_reason) {
    if (fallback_mode == FallbackMode::Error) {
      if (fallback_reason == FallbackReason::CpuId) {
        throw FallbackError{"LVQ and Leanvec functionality of SVS is not supported on non-Intel hardware."};
      }
      else if (fallback_reason == FallbackReason::MissingProprietary) {
        throw FallbackError{"Library was not compiled with proprietary (LVQ/LeanVec) interface support."};
    }
  }
  else if (fallback_mode == FallbackMode::Warning) {
      if (fallback_reason == FallbackReason::CpuId) {
        fmt::print("LVQ and Leanvec functionality of SVS is not supported on non-Intel hardware. Using uncompressed data.\n");
      }
      else if (fallback_reason == FallbackReason::MissingProprietary) {
        fmt::print("Library was not compiled with proprietary (LVQ/LeanVec) interface support. Using uncompressed data.\n");
    }
  }
}

} // namespace fallback
} // namespace svs
