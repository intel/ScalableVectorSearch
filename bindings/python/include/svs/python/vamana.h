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

// svs python bindings
#include "svs/python/common.h"
#include "svs/python/core.h"

// svs
#include "svs/index/vamana/calibrate.h"
#include "svs/lib/datatype.h"
#include "svs/lib/float16.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"

#include "svs/core/distance.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

namespace svs::python {
namespace vamana_specializations {

///
/// Specializations
///

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
    // XN(float,   float,        960);        // Gist - F32
    // XN(float,   float,        768);        // DPR - F32
    // XN(float,   float,        200);        // Text2Image - F32
    // XN(float,   float,        128);        // Sift - F32
    // XN(float,   float,        100);        // MSTuring 1B - F32
    // XN(float,   float,        96);         // Deep 1B - F32
    // XN(float,   float,        50);         // Glove50 - F32
    // XN(float,   float,        25);         // Glove25 - F32
    X (float,   float,        Dynamic, EnableBuild::FromFileAndArray);
    // X (float,   float,        512, EnableBuild::FromFileAndArray);

    // XN(float,   svs::Float16, 960); // Gist - F16
    // XN(float,   svs::Float16, 768); // DPR - F16
    // XN(float,   svs::Float16, 200); // Text2Image - F16
    // XN(float,   svs::Float16, 128); // Sift - F16
    // XN(float,   svs::Float16, 100); // MSTuring 1B - F16
    // X(float,   svs::Float16, 96, EnableBuild::FromFile);  // Deep 1B - F16
    // XN(float,   svs::Float16, 50);  // Glove50 - F16
    // XN(float,   svs::Float16, 25);  // Glove25 - F16
    using float16xfloat = svs::lib::Types<svs::Float16, float>;
    X (float16xfloat, svs::Float16, Dynamic, EnableBuild::FromFileAndArray);

    // XN(uint8_t, uint8_t,      128); // BigANN 1B
    X (uint8_t, uint8_t,      Dynamic, EnableBuild::FromFileAndArray);

    // XN(int8_t,  int8_t,       100); // MSSpace 1B
    X (int8_t,  int8_t,       Dynamic, EnableBuild::FromFileAndArray);
    // clang-format on
#undef XN
#undef X
}
} // namespace vamana_specializations

namespace vamana {

template <typename QueryType, typename Manager>
void add_experimental_calibration(pybind11::class_<Manager>& py_manager) {
    py_manager.def(
        "experimental_calibrate",
        [](Manager& self,
           py_contiguous_array_t<QueryType> queries,
           py_contiguous_array_t<uint32_t> groundtruth,
           size_t num_neighbors,
           double target_recall,
           const svs::index::vamana::CalibrationParameters& calibration_parameters) {
            return self.experimental_calibrate(
                data_view(queries),
                data_view(groundtruth),
                num_neighbors,
                target_recall,
                calibration_parameters
            );
        },
        pybind11::arg("queries"),
        pybind11::arg("groundtruth"),
        pybind11::arg("num_neighbors"),
        pybind11::arg("target_recall"),
        pybind11::arg("calibration_parameters") =
            svs::index::vamana::CalibrationParameters(),
        R"(
NOTE: This method is experimental and subject to change or removal without notice.

Run an experimental calibration routine to select the best search parameters.

Args:
    queries: Queries used to drive the calibration process.
    groundtruth: The groundtruth for the given query set.
    num_neighbors: The number of nearest neighbors to calibrate for.
    target_recall: The target `num_neighbors`-recall-at-`num_neighbors`. If such a recall is
        possible, then calibration will find parameters that optimize performance at this
        recall level.
    calibration_parameters: The hyper-parameters to use during calibration.

Returns:
    The best `svs.VamanaSearchParameters` found.

The calibration routine will also configure the index with the best found parameters.
Note that calibration will use the number of threads already assigned to the index and can
therefore be used to tune the algorithm to different threading amounts.

See also: `svs.VamanaCalibrationParameters`)"
    );
}

template <typename Manager> void add_interface(pybind11::class_<Manager>& manager) {
    manager.def_property_readonly(
        "experimental_backend_string",
        &Manager::experimental_backend_string,
        R"(
Read Only (str): Get a string identifying the full-type of the backend implementation.

This property is experimental and subject to change without a deprecation warning.)"
    );

    manager.def_property(
        "search_window_size",
        &Manager::get_search_window_size,
        &Manager::set_search_window_size,
        R"(
Read/Write (int): Get/set the size of the internal search buffer.
A larger value will likely yield more accurate results at the cost of speed.)"
    );

    manager.def_property(
        "search_parameters",
        &Manager::get_search_parameters,
        &Manager::set_search_parameters,
        R"(
"Read/Write (svs.VamanaSearchParameters): Get/set the current search parameters for the
index. These parameters modify both the algorithmic properties of search (affecting recall)
and non-algorthmic properties of search (affecting queries-per-second).

See also: `svs.VamanaSearchParameters`.)"
    );

    manager.def(
        "experimental_reset_performance_parameters",
        &Manager::experimental_reset_performance_parameters,
        R"(
Reset the internal performance-only parameters to built-in heuristics. This can be useful
if experimenting with different dataset implementations which may need different values
for performance-only parameters (such as prefetchers).

Calling this method should not affect recall.)"
    );

    manager.def_property(
        "visited_set_enabled",
        [](const Manager& self) {
            return self.get_search_parameters().search_buffer_visited_set_;
        },
        [](Manager& self, bool enable) {
            PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "Direct calls to to \"visited_set_enabled\" are deprecated. Instead, "
                "please use the \"svs.Vamana.search_parameters\" method to get and set "
                "the search parameters used by the index.",
                1
            );

            auto p = self.get_search_parameters();
            p.search_buffer_visited_set_ = enable;
            self.set_search_parameters(p);
        },
        R"(
    **Deprecated**

    Read/Write (bool): Get/set whether the visited set is used.
    Enabling the visited set can be helpful if the distance computations required are
    relatively expensive as it can reduce redundant computations.

    In general, through, it's probably faster to leave this disabled.
    )"
    );

    ///// Experiemntal Interfaces
    add_experimental_calibration<svs::Float16>(manager);
    add_experimental_calibration<float>(manager);
}

void wrap(pybind11::module& m);
} // namespace vamana
} // namespace svs::python
