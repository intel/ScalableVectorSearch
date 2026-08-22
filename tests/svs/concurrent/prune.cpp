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

// header under test
#include "svs/concurrent/prune.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Concurrent Pruning", "[concurrent][index][vamana]") {
    namespace v = svs::index::vamana::concurrent;
    // Protect against changes to the default strategies getting merged.
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceL2>,
                  v::ProgressivePruneStrategy>);
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceIP>,
                  v::IterativePruneStrategy>);
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceCosineSimilarity>,
                  v::IterativePruneStrategy>);

    CATCH_SECTION("Iterative Strategy") {
        CATCH_SECTION("Prune State") {
            // The concurrent prune is a two-round heuristic rather than the alpha
            // escalation loop: a neighbor that survives round one only conditionally is
            // marked `Candidate` and reconsidered in round two, while `Pruned` is final.
            // `reenable` therefore promotes `Candidate` -- not `Pruned` -- back to
            // `Available`. The pre-existing `svs::index::vamana::PruneState` keeps the
            // old three-state semantics and its own test.
            CATCH_REQUIRE(
                v::reenable(v::PruneState::Available) == v::PruneState::Available
            );
            CATCH_REQUIRE(v::reenable(v::PruneState::Added) == v::PruneState::Added);
            CATCH_REQUIRE(v::reenable(v::PruneState::Pruned) == v::PruneState::Pruned);
            CATCH_REQUIRE(
                v::reenable(v::PruneState::Candidate) == v::PruneState::Available
            );

            CATCH_REQUIRE(v::excluded(v::PruneState::Available) == false);
            CATCH_REQUIRE(v::excluded(v::PruneState::Added) == true);
            CATCH_REQUIRE(v::excluded(v::PruneState::Pruned) == true);
            CATCH_REQUIRE(v::excluded(v::PruneState::Candidate) == true);
        }
    }
}
