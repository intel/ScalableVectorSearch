/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
#pragma once

// lib
#include "svs/lib/meta.h"
#include "svs/lib/saveload.h"
#include "svs/lib/type_traits.h"

#include <cmath>
#include <span>

namespace svs::distance {

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
