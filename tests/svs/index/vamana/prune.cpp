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
// header under test
#include "svs/index/vamana/prune.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Pruning", "[index][vamana]") {
    namespace v = svs::index::vamana;
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
            CATCH_REQUIRE(
                v::reenable(v::PruneState::Available) == v::PruneState::Available
            );
            CATCH_REQUIRE(v::reenable(v::PruneState::Added) == v::PruneState::Added);
            CATCH_REQUIRE(v::reenable(v::PruneState::Pruned) == v::PruneState::Available);

            CATCH_REQUIRE(v::excluded(v::PruneState::Available) == false);
            CATCH_REQUIRE(v::excluded(v::PruneState::Added) == true);
            CATCH_REQUIRE(v::excluded(v::PruneState::Pruned) == true);
        }
    }
}
