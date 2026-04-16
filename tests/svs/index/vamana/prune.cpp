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
#include "svs/index/vamana/prune.h"

// core
#include "svs/core/data/simple.h"
#include "svs/core/distance/euclidean.h"

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

    CATCH_SECTION("Duplicate Cluster Trap") {
        auto data = svs::data::SimpleData<float>(6, 4);
        auto d0 = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f};
        auto d4 = std::vector<float>{2.0f, 1.0f, 1.0f, 1.0f};
        auto d5 = std::vector<float>{1.5f, 1.0f, 1.0f, 1.0f};

        for (size_t i = 0; i < 4; ++i) {
            data.set_datum(i, d0);
        }
        data.set_datum(4, d4);
        data.set_datum(5, d5);

        auto dist = svs::distance::DistanceL2();
        auto accessor = svs::data::GetDatumAccessor{};

        std::vector<svs::Neighbor<size_t>> pool = {
            {size_t{0}, 0.0f},
            {size_t{1}, 0.0f},
            {size_t{2}, 0.0f},
            {size_t{3}, 0.0f},
            {size_t{4}, 1.0f}};

        CATCH_SECTION("Iterative Strategy Fix") {
            std::vector<svs::Neighbor<size_t>> result;
            v::heuristic_prune_neighbors(
                v::IterativePruneStrategy{},
                2,
                1.3f,
                data,
                accessor,
                dist,
                size_t{5},
                std::span<const svs::Neighbor<size_t>>(pool),
                result
            );

            CATCH_REQUIRE(result.size() == 2);
            CATCH_REQUIRE(result[0].id() == 0);
            CATCH_REQUIRE(result[1].id() == 4);
        }

        CATCH_SECTION("Progressive Strategy Fix") {
            std::vector<svs::Neighbor<size_t>> result;
            v::heuristic_prune_neighbors(
                v::ProgressivePruneStrategy{},
                2,
                1.3f,
                data,
                accessor,
                dist,
                size_t{5},
                std::span<const svs::Neighbor<size_t>>(pool),
                result
            );

            CATCH_REQUIRE(result.size() == 2);
            CATCH_REQUIRE(result[0].id() == 0);
            CATCH_REQUIRE(result[1].id() == 4);
        }
    }
}
