/*
 * Copyright 2024 Intel Corporation
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
