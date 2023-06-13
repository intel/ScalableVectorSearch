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

#include "core.h"

#include <pybind11/pybind11.h>

namespace dynamic_vamana {

// Specializations
template <typename F> void for_standard_specializations(F&& f) {
#define X(Q, T, Dist, N) f(Type<Q>(), Type<T>(), Dist(), Val<N>())
    X(float, float, DistanceL2, svs::Dynamic);
    X(float, float, DistanceIP, svs::Dynamic);
    X(float, svs::Float16, DistanceL2, svs::Dynamic);
    X(float, svs::Float16, DistanceIP, svs::Dynamic);
#undef X
}

template <typename F> void for_compressed_specializations(F&& f) {
#define X(Dist, N) f(Dist(), Val<N>())
    X(DistanceL2, svs::Dynamic);
    X(DistanceIP, svs::Dynamic);
#undef X
}

void wrap(pybind11::module& m);
} // namespace dynamic_vamana
