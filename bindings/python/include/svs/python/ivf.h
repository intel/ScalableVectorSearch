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

// svs python bindings
#include "svs/python/common.h"
#include "svs/python/core.h"

#include "svs/core/distance.h"
#include "svs/lib/bfloat16.h"
#include "svs/lib/datatype.h"
#include "svs/lib/float16.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

namespace svs::python {
namespace ivf_specializations {
///
/// Flag to selectively enable index building.
///
enum class EnableBuild { None, FromFile, FromFileAndArray };

template <EnableBuild B>
inline constexpr bool enable_build_from_file =
    (B == EnableBuild::FromFile || B == EnableBuild::FromFileAndArray);

template <EnableBuild B>
inline constexpr bool enable_build_from_array = (B == EnableBuild::FromFileAndArray);

// Define all desired specializations for searching and building.
template <typename F> void for_standard_specializations(F&& f) {
#define X(Q, T, N, B) f.template operator()<Q, T, N, B>();
#define XN(Q, T, N) X(Q, T, N, EnableBuild::None)
    // Pattern:
    // QueryType, DataType, Dimensionality, Enable Building
    // clang-format off
    X(float,  svs::BFloat16, Dynamic, EnableBuild::FromFileAndArray);
    X(float,  float,         Dynamic, EnableBuild::FromFileAndArray);
    X(float,  svs::Float16,  Dynamic, EnableBuild::FromFileAndArray);
    // clang-format on
#undef XN
#undef X
}
} // namespace ivf_specializations

namespace ivf {
template <typename Manager> void add_interface(pybind11::class_<Manager>& manager) {
    manager.def_property_readonly(
        "experimental_backend_string",
        &Manager::experimental_backend_string,
        R"(
            Read Only (str): Get a string identifying the full-type of the backend implementation.

            This property is experimental and subject to change without a deprecation warning.)"
    );

    manager.def_property(
        "search_parameters",
        &Manager::get_search_parameters,
        &Manager::set_search_parameters,
        R"(
            "Read/Write (svs.IVFSearchParameters): Get/set the current search parameters for the
            index. These parameters modify both the algorithmic properties of search (affecting recall)
            and non-algorthmic properties of search (affecting queries-per-second).

            See also: `svs.IVFSearchParameters`.)"
    );
}

void wrap(pybind11::module& m);
} // namespace ivf
} // namespace svs::python
