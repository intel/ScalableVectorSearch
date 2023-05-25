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

// svs
#include "svs/lib/saveload.h"

// stl
#include <cstddef>

namespace svs::index::vamana {

/// @brief Parameters controlling graph construction for the Vamana graph index.
struct VamanaBuildParameters {
    /// The pruning parameter.
    float alpha;

    /// The maximum degree in the graph. A higher max degree may yield a higher quality
    /// graph in terms of recall for performance, but the memory footprint of the graph is
    /// directly proportional to the maximum degree.
    size_t graph_max_degree;

    /// The search window size to use during graph construction. A higher search window
    /// size will yield a higher quality graph since more overall vertices are considered,
    /// but will increase construction time.
    size_t window_size;

    /// Set a limit on the number of neighbors considered during pruning. In practice, set
    /// this to a high number (at least 5 times greater than the window_size) and forget
    /// about it.
    size_t max_candidate_pool_size;

    /// **Soon to be deprecated**: The number of threads to use for graph construction.
    /// Reason for deprecation: Number of threads is often obtained from external sources
    /// so including it in this struct no longer entirely makes sense.
    size_t nthreads;
};
} // namespace svs::index::vamana
