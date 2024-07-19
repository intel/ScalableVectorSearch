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

#include "svs/core.h"

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

template <typename F> void for_compressed_specializations(F&& f) {
    using Sequential = svs::quantization::lvq::Sequential;
#define X(Dist, Primary, Residual, Strategy, N) \
    f.template operator()<Dist, Primary, Residual, Strategy, N>()
    // Sequential
    X(DistanceL2, 4, 8, Sequential, Dynamic);
    X(DistanceIP, 4, 8, Sequential, Dynamic);
    X(DistanceL2, 8, 0, Sequential, Dynamic);
    X(DistanceIP, 8, 0, Sequential, Dynamic);

    // Turbo
    using Turbo16x8 = svs::quantization::lvq::Turbo<16, 8>;
    X(DistanceIP, 4, 8, Turbo16x8, Dynamic);
#undef X
}

void wrap(pybind11::module& m);
} // namespace dynamic_vamana
