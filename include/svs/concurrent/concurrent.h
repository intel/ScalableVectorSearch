/*
 * Copyright 2026 Intel Corporation
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
/// @brief Umbrella header for the concurrent dynamic Vamana index.
///
/// Pulls in the whole stack and introduces the short alias ``svs::concurrent``. See
/// ``include/svs/concurrent/README.md`` for the design.
///
/// Including the individual headers directly works equally well; this one exists so that a
/// caller who just wants the index does not have to know which of them declares it.
///

#include "svs/concurrent/blocked_data.h"
#include "svs/concurrent/consolidate.h"
#include "svs/concurrent/dynamic_index.h"
#include "svs/concurrent/dynamic_search_buffer.h"
#include "svs/concurrent/graph.h"
#include "svs/concurrent/graph_concepts.h"
#include "svs/concurrent/greedy_search.h"
#include "svs/concurrent/iterator.h"
#include "svs/concurrent/multi.h"
#include "svs/concurrent/prune.h"
#include "svs/concurrent/reverse_edges.h"
#include "svs/concurrent/spinlock.h"
#include "svs/concurrent/translation.h"
#include "svs/concurrent/vamana_build.h"

namespace svs {

/// @brief Short alias for the concurrent dynamic Vamana index namespace.
///
/// The implementation namespace is nested inside the one it extends so that unchanged
/// entities resolve to their pre-existing declarations (see the README); that makes the
/// fully-qualified name long. This alias is purely for convenience at call sites.
namespace concurrent = index::vamana::concurrent;

} // namespace svs
