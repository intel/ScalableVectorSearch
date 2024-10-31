/**
 *    Copyright (C) 2024, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

// svs python bindings
#include "svs/python/core.h"

// svs
#include "svs/core/data.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/saveload.h"

// Dispatch rule for serialized objects to a VectorDataLoader.
template <typename T, size_t N>
struct svs::lib::DispatchConverter<
    svs::lib::SerializedObject,
    svs::VectorDataLoader<T, N, svs::python::RebindAllocator<T>>> {
    using To = svs::VectorDataLoader<T, N, svs::python::RebindAllocator<T>>;

    static int64_t match(const svs::lib::SerializedObject& object) {
        auto ex = svs::lib::try_load<svs::data::Matcher>(object);
        if (!ex) {
            // Could not load for some reason.
            // Invalid match.
            return lib::invalid_match;
        }

        // Check suitability.
        const auto& matcher = ex.value();
        auto code = svs::data::detail::check_match<T, N>(matcher.eltype, matcher.dims);
        return code;
    }

    static To convert(const svs::lib::SerializedObject& object) {
        return To{object.context().get_directory()};
    }
};
