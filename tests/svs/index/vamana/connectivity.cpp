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

// Header under test: unreachable_nodes and repair_connectivity.
#include "svs/index/vamana/connectivity.h"

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/graph.h"
#include "svs/lib/threads.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <cstdint>
#include <span>
#include <vector>

namespace {

using Idx = uint32_t;

std::vector<std::vector<Idx>> adjacency_snapshot(const svs::graphs::SimpleGraph<Idx>& graph
) {
    std::vector<std::vector<Idx>> out(graph.n_nodes());
    for (size_t i = 0; i < graph.n_nodes(); ++i) {
        auto node = graph.get_node(static_cast<Idx>(i));
        out[i].assign(node.begin(), node.end());
    }
    return out;
}

svs::data::SimpleData<float, 1> make_positions(const std::vector<float>& values) {
    auto data = svs::data::SimpleData<float, 1>(values.size(), 1);
    for (size_t i = 0; i < values.size(); ++i) {
        data.set_datum(i, std::vector<float>{values[i]});
    }
    return data;
}

} // namespace

CATCH_TEST_CASE("unreachable_nodes", "[index][vamana][connectivity]") {
    namespace v = svs::index::vamana;

    auto graph = svs::default_graph<Idx>(5, 3);
    graph.replace_node(0, std::vector<Idx>{1, 2});
    graph.replace_node(2, std::vector<Idx>{3});
    // Nodes 1, 3, 4 keep their default empty adjacency list.

    CATCH_SECTION("One root reaches a prefix, two nodes stay unreachable") {
        std::vector<Idx> roots{0};
        auto unreachable = v::unreachable_nodes(graph, std::span<const Idx>(roots));
        CATCH_REQUIRE(unreachable == std::vector<Idx>{4});
    }

    CATCH_SECTION("Adding the straggler as a root reaches everything") {
        std::vector<Idx> roots{0, 4};
        auto unreachable = v::unreachable_nodes(graph, std::span<const Idx>(roots));
        CATCH_REQUIRE(unreachable.empty());
    }
}

CATCH_TEST_CASE(
    "repair_connectivity leaves an already-connected graph untouched",
    "[index][vamana][connectivity]"
) {
    namespace v = svs::index::vamana;

    auto graph = svs::default_graph<Idx>(5, 3);
    graph.replace_node(0, std::vector<Idx>{1});
    graph.replace_node(1, std::vector<Idx>{2});
    graph.replace_node(2, std::vector<Idx>{3});
    graph.replace_node(3, std::vector<Idx>{4});

    auto before = adjacency_snapshot(graph);

    auto data = make_positions({0.0F, 1.0F, 2.0F, 3.0F, 4.0F});
    svs::distance::DistanceL2 distance{};
    auto params = v::VamanaBuildParameters{1.2F, 3, 10, 10, 3, true};
    auto pool = svs::threads::DefaultThreadPool(1);
    std::vector<Idx> roots{0};

    auto report = v::repair_connectivity(
        graph, data, distance, params, std::span<const Idx>(roots), pool
    );

    CATCH_REQUIRE(report.unreachable_before == 0);
    CATCH_REQUIRE(report.unreachable_after == 0);
    CATCH_REQUIRE(report.rounds == 0);
    CATCH_REQUIRE(report.edges_appended == 0);
    CATCH_REQUIRE(report.edges_accepted_through_prune == 0);
    CATCH_REQUIRE(report.edges_forced == 0);
    CATCH_REQUIRE(adjacency_snapshot(graph) == before);
}

CATCH_TEST_CASE(
    "repair_connectivity reconnects a disconnected node with room to spare",
    "[index][vamana][connectivity]"
) {
    namespace v = svs::index::vamana;

    // Nodes 0-3 reachable from root 0, all with degree < graph_max_degree.
    auto graph = svs::default_graph<Idx>(5, 3);
    graph.replace_node(0, std::vector<Idx>{1});
    graph.replace_node(1, std::vector<Idx>{2});
    graph.replace_node(2, std::vector<Idx>{3});

    auto data = make_positions({0.0F, 1.0F, 2.0F, 3.0F, 2.1F});
    svs::distance::DistanceL2 distance{};
    auto params = v::VamanaBuildParameters{1.2F, 3, 10, 10, 3, true};
    auto pool = svs::threads::DefaultThreadPool(1);
    std::vector<Idx> roots{0};

    auto report = v::repair_connectivity(
        graph, data, distance, params, std::span<const Idx>(roots), pool
    );

    CATCH_REQUIRE(report.unreachable_before == 1);
    CATCH_REQUIRE(report.unreachable_after == 0);
    CATCH_REQUIRE(report.rounds >= 1);
    CATCH_REQUIRE(
        report.edges_appended + report.edges_accepted_through_prune + report.edges_forced ==
        1
    );
    CATCH_REQUIRE(v::unreachable_nodes(graph, std::span<const Idx>(roots)).empty());
    for (size_t i = 0; i < graph.n_nodes(); ++i) {
        CATCH_REQUIRE(
            graph.get_node_degree(static_cast<Idx>(i)) <= params.graph_max_degree
        );
    }
}

CATCH_TEST_CASE(
    "repair_connectivity reconnects a node whose nearest reachable neighbors are all at "
    "full degree",
    "[index][vamana][connectivity]"
) {
    namespace v = svs::index::vamana;

    // 2-regular cycle at full degree; node 4 is disconnected and nearest to root 0.
    auto graph = svs::default_graph<Idx>(5, 2);
    graph.replace_node(0, std::vector<Idx>{1, 2});
    graph.replace_node(1, std::vector<Idx>{2, 3});
    graph.replace_node(2, std::vector<Idx>{3, 0});
    graph.replace_node(3, std::vector<Idx>{0, 1});

    auto data = make_positions({0.0F, 1.0F, 2.0F, 3.0F, 0.1F});
    svs::distance::DistanceL2 distance{};
    auto params = v::VamanaBuildParameters{1.2F, 2, 10, 10, 2, true};
    auto pool = svs::threads::DefaultThreadPool(1);
    std::vector<Idx> roots{0};

    for (size_t i = 0; i < 4; ++i) {
        CATCH_REQUIRE(graph.get_node_degree(static_cast<Idx>(i)) == 2);
    }

    auto report = v::repair_connectivity(
        graph, data, distance, params, std::span<const Idx>(roots), pool
    );

    CATCH_REQUIRE(report.unreachable_before == 1);
    CATCH_REQUIRE(report.unreachable_after == 0);
    CATCH_REQUIRE(
        report.edges_appended + report.edges_accepted_through_prune + report.edges_forced ==
        1
    );
    CATCH_REQUIRE(report.edges_forced_unsafe == 0);
    CATCH_REQUIRE(!report.stalled);
    CATCH_REQUIRE(v::unreachable_nodes(graph, std::span<const Idx>(roots)).empty());
    for (size_t i = 0; i < graph.n_nodes(); ++i) {
        CATCH_REQUIRE(
            graph.get_node_degree(static_cast<Idx>(i)) <= params.graph_max_degree
        );
    }
}

CATCH_TEST_CASE(
    "repair_connectivity falls back to an unsafe forced replacement when no candidate "
    "offers a safe one, and reports it",
    "[index][vamana][connectivity]"
) {
    namespace v = svs::index::vamana;

    // Root->1->2, both at degree 1; node 2 is a leaf, so dropping it is never safe.
    auto graph = svs::default_graph<Idx>(4, 1);
    graph.replace_node(0, std::vector<Idx>{1});
    graph.replace_node(1, std::vector<Idx>{2});
    // Node 2 (a leaf) and node 3 (disconnected) keep their default empty adjacency list.

    auto data = make_positions({100.0F, 0.0F, 2.0F, -3.0F});
    svs::distance::DistanceL2 distance{};
    auto params = v::VamanaBuildParameters{1.0F, 1, 1, 10, 1, true};
    auto pool = svs::threads::DefaultThreadPool(1);
    std::vector<Idx> roots{0};

    // Stop after one round: this test is about the forced step within a single round.
    auto report = v::repair_connectivity(
        graph, data, distance, params, std::span<const Idx>(roots), pool, size_t{1}
    );

    CATCH_REQUIRE(report.rounds == 1);
    CATCH_REQUIRE(report.unreachable_before == 1);
    CATCH_REQUIRE(report.edges_appended == 0);
    CATCH_REQUIRE(report.edges_accepted_through_prune == 0);
    CATCH_REQUIRE(report.edges_forced == 0);
    CATCH_REQUIRE(report.edges_forced_unsafe == 1);

    // Node 1's forced edge to node 3 displaces node 2, leaving node 2 unreachable.
    CATCH_REQUIRE(report.stalled);
    CATCH_REQUIRE(report.unreachable_after == 1);
    CATCH_REQUIRE(
        v::unreachable_nodes(graph, std::span<const Idx>(roots)) == std::vector<Idx>{2}
    );
    for (size_t i = 0; i < graph.n_nodes(); ++i) {
        CATCH_REQUIRE(
            graph.get_node_degree(static_cast<Idx>(i)) <= params.graph_max_degree
        );
    }
}

CATCH_TEST_CASE(
    "repair_connectivity's stall guard stops an oscillating repair before max_rounds",
    "[index][vamana][connectivity]"
) {
    namespace v = svs::index::vamana;

    // Same graph as the unsafe-fallback case, run to default max_rounds instead of 1.
    auto graph = svs::default_graph<Idx>(4, 1);
    graph.replace_node(0, std::vector<Idx>{1});
    graph.replace_node(1, std::vector<Idx>{2});
    // Node 2 (a leaf) and node 3 (disconnected) keep their default empty adjacency list.

    auto data = make_positions({100.0F, 0.0F, 2.0F, -3.0F});
    svs::distance::DistanceL2 distance{};
    auto params = v::VamanaBuildParameters{1.0F, 1, 1, 10, 1, true};
    auto pool = svs::threads::DefaultThreadPool(1);
    std::vector<Idx> roots{0};

    // Default max_rounds: the stall guard, not the round cap, has to be what stops this.
    auto report = v::repair_connectivity(
        graph, data, distance, params, std::span<const Idx>(roots), pool
    );

    CATCH_REQUIRE(report.stalled);
    CATCH_REQUIRE(report.rounds == 1);
    CATCH_REQUIRE(report.unreachable_before == 1);
    CATCH_REQUIRE(report.edges_forced_unsafe == 1);
    CATCH_REQUIRE(
        report.unreachable_after ==
        v::unreachable_nodes(graph, std::span<const Idx>(roots)).size()
    );
    CATCH_REQUIRE(report.unreachable_after == 1);
    for (size_t i = 0; i < graph.n_nodes(); ++i) {
        CATCH_REQUIRE(
            graph.get_node_degree(static_cast<Idx>(i)) <= params.graph_max_degree
        );
    }
}
