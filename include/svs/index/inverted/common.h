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
