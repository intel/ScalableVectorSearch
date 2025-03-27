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
    X(DistanceL2, 4, 0, Sequential, Dynamic);
    X(DistanceIP, 4, 0, Sequential, Dynamic);
    X(DistanceL2, 4, 4, Sequential, Dynamic);
    X(DistanceIP, 4, 4, Sequential, Dynamic);
    X(DistanceL2, 4, 8, Sequential, Dynamic);
    X(DistanceIP, 4, 8, Sequential, Dynamic);
    X(DistanceL2, 8, 0, Sequential, Dynamic);
    X(DistanceIP, 8, 0, Sequential, Dynamic);

    // Turbo
    using Turbo16x8 = svs::quantization::lvq::Turbo<16, 8>;
    X(DistanceL2, 4, 0, Turbo16x8, Dynamic);
    X(DistanceIP, 4, 0, Turbo16x8, Dynamic);
    X(DistanceL2, 4, 4, Turbo16x8, Dynamic);
    X(DistanceIP, 4, 4, Turbo16x8, Dynamic);
    X(DistanceL2, 4, 8, Turbo16x8, Dynamic);
    X(DistanceIP, 4, 8, Turbo16x8, Dynamic);
#undef X
}

template <typename F> void for_leanvec_specializations(F&& f) {
#define X(Dist, Primary, Secondary, L, N) \
    f.template operator()<Dist, Primary, Secondary, L, N>()
    X(DistanceL2, svs::Float16, svs::Float16, Dynamic, Dynamic);
    X(DistanceIP, svs::Float16, svs::Float16, Dynamic, Dynamic);

    X(DistanceL2, svs::leanvec::UsingLVQ<8>, svs::Float16, Dynamic, Dynamic);
    X(DistanceIP, svs::leanvec::UsingLVQ<8>, svs::Float16, Dynamic, Dynamic);

    X(DistanceL2, svs::leanvec::UsingLVQ<8>, svs::leanvec::UsingLVQ<8>, Dynamic, Dynamic);
    X(DistanceIP, svs::leanvec::UsingLVQ<8>, svs::leanvec::UsingLVQ<8>, Dynamic, Dynamic);

    X(DistanceL2, svs::leanvec::UsingLVQ<4>, svs::leanvec::UsingLVQ<8>, Dynamic, Dynamic);
    X(DistanceIP, svs::leanvec::UsingLVQ<4>, svs::leanvec::UsingLVQ<8>, Dynamic, Dynamic);

    X(DistanceL2, svs::leanvec::UsingLVQ<4>, svs::leanvec::UsingLVQ<4>, Dynamic, Dynamic);
    X(DistanceIP, svs::leanvec::UsingLVQ<4>, svs::leanvec::UsingLVQ<4>, Dynamic, Dynamic);
#undef X
}

void wrap(pybind11::module& m);
} // namespace svs::python::dynamic_vamana
