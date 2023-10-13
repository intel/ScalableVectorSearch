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

// header under test
#include "svs/index/vamana/consolidate.h"

// svs
#include "svs/core/distance.h"
#include "svs/lib/timing.h"

// test utilities
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

template <typename Graph, typename Predicate>
void check_post_conditions(const Graph& graph, Predicate&& predicate) {
    bool contains_deleted = false;
    svs::threads::UnitRange<size_t> node_range{0, graph.n_nodes()};
    for (size_t i : node_range) {
        if (predicate(i)) {
            contains_deleted = true;
            continue;
        }

        const auto& neighbors = graph.get_node(i);
        CATCH_REQUIRE(std::none_of(neighbors.begin(), neighbors.end(), predicate));

        // Don't invent nodes out of thin air.
        CATCH_REQUIRE(std::all_of(neighbors.begin(), neighbors.end(), [&](const auto& i) {
            return node_range.contains(i);
        }));
    }
    CATCH_REQUIRE(contains_deleted);
}

} // namespace

CATCH_TEST_CASE("Graph Consolidation", "[graph_index]") {
    auto graph = test_dataset::graph();
    auto data = test_dataset::data_f32();
    auto threadpool = svs::threads::NativeThreadPool(2);

    CATCH_SECTION("Remove Even Nodes") {
        auto tic = svs::lib::now();
        auto predicate = [](const auto& i) { return (i % 10) == 0; };

        svs::distance::DistanceL2 distance{};
        svs::index::vamana::consolidate(
            graph, data, threadpool, graph.max_degree(), 750, 1.2, distance, predicate
        );
        std::cout << "Pruning took " << svs::lib::time_difference(svs::lib::now(), tic)
                  << std::endl;

        // Ensure that all non-deleted nodes only have non-deleted neighbors.
        check_post_conditions(graph, predicate);
    }
}
