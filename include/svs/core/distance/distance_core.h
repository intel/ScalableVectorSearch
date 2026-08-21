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

// lib
#include "svs/lib/meta.h"
#include "svs/lib/saveload.h"
#include "svs/lib/type_traits.h"

// The extent list and the ISA levels, generated from cmake/dispatch-surface.cmake.
#include "svs/core/distance/dispatch_surface.h"

#include <cmath>
#include <span>

namespace svs::distance {

enum class AVX_AVAILABILITY { NONE, AVX2, AVX512 };

/// The extents that have a fixed-extent kernel, including svs::Dynamic.
#define SVS_DIM_LIST_ENTRY(N) N,
constexpr std::array<size_t, SVS_SUPPORTED_DIM_COUNT> supported_dim_list{
    SVS_FOR_EACH_SUPPORTED_DIM(SVS_DIM_LIST_ENTRY)};
#undef SVS_DIM_LIST_ENTRY

/// Whether N has a fixed-extent kernel.
///
/// This is not a capability test: every dimensionality is supported. An extent
/// that answers `false` here dispatches to the svs::Dynamic kernel instead of a
/// fully unrolled one.
template <size_t N> constexpr bool is_dim_supported() {
    for (auto i : supported_dim_list) {
        if (i == N) {
            return true;
        }
    }
    return false;
}

using default_accum_type = float;

template <Arithmetic Accum, typename T, size_t Extent>
Accum norm_square(lib::Type<Accum> /*unused*/, std::span<T, Extent> data) {
    Accum accum{0};
    for (const auto& i : data) {
        accum += i * i;
    }
    return accum;
}

template <Arithmetic Accum, typename T, size_t Extent>
Accum norm(lib::Type<Accum> type, std::span<T, Extent> data) {
    return std::sqrt(norm_square(type, data));
}

template <typename T, size_t Extent>
default_accum_type norm_square(std::span<T, Extent> data) {
    return norm_square(lib::Type<default_accum_type>(), data);
}

template <typename T, size_t Extent> default_accum_type norm(std::span<T, Extent> data) {
    return norm(lib::Type<default_accum_type>(), data);
}

struct DistanceSerialization {
    static constexpr lib::Version save_version = lib::Version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "builtin_distance_function";

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == serialization_schema && version == save_version;
    }

    static lib::SaveTable save(std::string_view name) {
        return lib::SaveTable(serialization_schema, save_version, {SVS_LIST_SAVE(name)});
    }

    static void
    check_load(const lib::ContextFreeLoadTable& table, std::string_view expected) {
        auto retrieved = lib::load_at<std::string>(table, "name");
        if (retrieved != expected) {
            throw ANNEXCEPTION(
                "Loading error. Expected name {}. Instead, got {}.", expected, retrieved
            );
        }
    }
};

} // namespace svs::distance
