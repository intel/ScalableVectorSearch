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

#include "svs/core/distance.h"

// Common definitions.
namespace svs::index::inverted {

// Obtain cut-off thresholds from a given base distance.
// The exact mechanics depend on the distance functor being used.
//
// In general, increasing `epsilon` should result in looser bounds.
//
// For example, with L2 distance, we try to minimize.
//
// With a `bound = nearest * (1 + epsilon)` where, distances greater than `nearest`
// but less than `bound` are accepted.
//
// With the Inner Product distance, we try to maximize.
// When `bound = nearest / (1 + epsilon)`, a higher epsilon makes the interval of
// accepted distances larger.

template <typename T> inline T bound_with(T nearest, T epsilon, svs::DistanceL2) {
    return nearest * (1 + epsilon);
}

template <typename T> inline T bound_with(T nearest, T epsilon, svs::DistanceIP) {
    // TODO: What do we do if the best match is simply bad?
    assert(nearest > 0.0f);
    return nearest / (1 + epsilon);
}

} // namespace svs::index::inverted
