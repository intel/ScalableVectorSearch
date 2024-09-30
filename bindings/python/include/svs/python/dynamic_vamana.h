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
} // namespace svs::python::dynamic_vamana
