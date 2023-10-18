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

#include "common.h"
#include "core.h"

#include "svs/lib/datatype.h"
#include "svs/lib/float16.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"

#include "svs/core/distance.h"

#include <pybind11/pybind11.h>

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
#define X(Q, T, N, B) f(Type<Q>(), Type<T>(), Val<N>(), Const<B>())
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

    // XN(float,   svs::Float16, 960); // Gist - F16
    // XN(float,   svs::Float16, 768); // DPR - F16
    // XN(float,   svs::Float16, 200); // Text2Image - F16
    // XN(float,   svs::Float16, 128); // Sift - F16
    // XN(float,   svs::Float16, 100); // MSTuring 1B - F16
    // X(float,   svs::Float16, 96, EnableBuild::FromFile);  // Deep 1B - F16
    // XN(float,   svs::Float16, 50);  // Glove50 - F16
    // XN(float,   svs::Float16, 25);  // Glove25 - F16
    X (float,   svs::Float16, Dynamic, EnableBuild::FromFileAndArray);

    XN(uint8_t, uint8_t,      128); // BigANN 1B
    X (uint8_t, uint8_t,      Dynamic, EnableBuild::FromFileAndArray);

    // XN(int8_t,  int8_t,       100); // MSSpace 1B
    X (int8_t,  int8_t,       Dynamic, EnableBuild::FromFileAndArray);
    // clang-format on
#undef XN
#undef X
}

// Compressed search specializations.
template <typename F> void compressed_specializations(F&& f) {
#define X(Dist, N, B) f(Dist(), svs::meta::Val<N>(), Const<B>())
    // Pattern:
    // DistanceType, Dimensionality, EnableBuild
    // X(DistanceL2, 25, true);
    // X(DistanceIP, 25, true);
    // X(DistanceL2, 50, true);
    // X(DistanceIP, 50, true);
    X(DistanceL2, 96, true);
    X(DistanceIP, 96, true);
    // X(DistanceL2, 100, true);
    // X(DistanceIP, 100, true);
    X(DistanceL2, 128, true);
    X(DistanceIP, 128, true);
    // X(DistanceL2, 200, true);
    // X(DistanceIP, 200, true);
    // X(DistanceL2, 768, true);
    // X(DistanceIP, 768, true);
    // X(DistanceL2, 960, true);
    // X(DistanceIP, 960, true);
    X(DistanceL2, Dynamic, true);
    X(DistanceIP, Dynamic, true);
#undef X
}

} // namespace vamana_specializations

namespace vamana {

template <typename Manager> void add_interface(pybind11::class_<Manager>& manager) {
    manager.def_property(
        "search_window_size",
        &Manager::get_search_window_size,
        &Manager::set_search_window_size,
        R"(
Read/Write (int): Get/set the size of the internal search buffer.
A larger value will likely yield more accurate results at the cost of speed.
        )"
    );

    manager.def_property(
        "visited_set_enabled",
        &Manager::visited_set_enabled,
        [](Manager& self, bool enable) {
            enable ? self.enable_visited_set() : self.disable_visited_set();
        },
        R"(
Read/Write (bool): Get/set whether the visited set is used.
Enabling the visited set can be helpful if the distance computations required are
relatively expensive as it can reduce redundant computations.

In general, through, it's probably faster to leave this disabled.
        )"
    );
}

void wrap(pybind11::module& m);
} // namespace vamana
