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

///
/// @file
///

#include "svs/concepts/graph.h"

#include <cstdint>

namespace svs::index::vamana::concurrent::graphs {

/// Outcome of `MemoryGraph::add_edge(src, dst)`. Distinguishes the cases so callers
/// can route dropped edges (e.g. to a backedge buffer) without a TOCTOU race between a
/// pre-check and the insert.
enum class AddEdgeResult : uint8_t {
    Added,         // Edge was inserted.
    AlreadyExists, // Edge was already present (or self-loop). Not inserted.
    Full,          // Node's adjacency list is at max_degree. Edge NOT inserted.
    Rejected,      // The source is no longer eligible for a backlink.
};

///
/// @ingroup graph_concept_entry
/// @brief Mutable graph whose edge insertion returns an ``AddEdgeResult`` status.
///
/// @tparam T Graph type to check.
/// @see svs::graphs::MemoryGraph for the shared graph requirements.
/// @see AddEdgeResult for the insertion outcomes and their meanings.
///
template <typename T>
concept MemoryGraph = svs::graphs::MemoryGraph<T, AddEdgeResult>;

} // namespace svs::index::vamana::concurrent::graphs
