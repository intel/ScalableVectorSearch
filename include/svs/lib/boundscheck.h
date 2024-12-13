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

#include "svs/lib/preprocessor.h"
#include "tsl/robin_map.h"

#include <unordered_map>
#include <vector>

namespace svs {

SVS_VALIDATE_BOOL_ENV(SVS_CHECK_BOUNDS);
#if SVS_CHECK_BOUNDS
static constexpr bool checkbounds_v = true;
#else
static constexpr bool checkbounds_v = false;
#endif

// Type must explicitly opt-in to optional bounds checking.
template <typename T> inline constexpr bool enable_boundschecking = false;

template <typename T, typename Alloc>
inline constexpr bool enable_boundschecking<std::vector<T, Alloc>> = true;

template <typename... Args>
inline constexpr bool enable_boundschecking<std::unordered_map<Args...>> = true;

///
/// Templates
///

template <typename T, typename K>
    requires enable_boundschecking<T>
auto getindex(T& v, K i) -> typename T::reference {
    if constexpr (checkbounds_v) {
        return v.at(i);
    } else {
        return v[i];
    }
}

template <typename T, typename K>
    requires enable_boundschecking<T>
auto getindex(const T& v, K i) -> typename T::const_reference {
    if constexpr (checkbounds_v) {
        return v.at(i);
    } else {
        return v[i];
    }
}

} // namespace svs
