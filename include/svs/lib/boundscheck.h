/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
