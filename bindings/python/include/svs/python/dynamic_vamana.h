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

// svs python bindings
#include "svs/python/core.h"

#include <pybind11/pybind11.h>

namespace svs::python::dynamic_vamana {

// Specializations
template <typename F> void for_standard_specializations(F&& f) {
#define X(Q, T, Dist, N) f.template operator()<Q, T, Dist, N>()
    X(float, float, DistanceL2, Dynamic);
    X(float, float, DistanceIP, Dynamic);
    X(float, svs::Float16, DistanceL2, Dynamic);
    X(float, svs::Float16, DistanceIP, Dynamic);
#undef X
}

void wrap(pybind11::module& m);
} // namespace svs::python::dynamic_vamana
