/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
