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
