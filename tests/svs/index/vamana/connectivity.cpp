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

// tests
#include "tests/svs/index/vamana/gem_test_utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using Idx = uint32_t;
namespace v = svs::index::vamana;

svs::data::SimpleData<float, 1> make_positions(const std::vector<float>& values) {
    auto data = svs::data::SimpleData<float, 1>(values.size(), 1);
    for (size_t i = 0; i < values.size(); ++i) {
        data.set_datum(i, std::vector<float>{values[i]});
    }
    return data;
}

svs::graphs::SimpleGraph<Idx> build_test_graph(
    size_t n_nodes, size_t max_degree, const std::vector<std::vector<Idx>>& adjacency
) {
    auto graph = svs::default_graph<Idx>(n_nodes, max_degree);
    for (size_t i = 0; i < adjacency.size(); ++i) {
        graph.replace_node(static_cast<Idx>(i), adjacency[i]);
    }
    return graph;
}

std::vector<std::vector<Idx>> adjacency_snapshot(const svs::graphs::SimpleGraph<Idx>& graph
) {
    std::vector<std::vector<Idx>> out(graph.n_nodes());
    for (size_t i = 0; i < graph.n_nodes(); ++i) {
        auto node = graph.get_node(static_cast<Idx>(i));
        out[i].assign(node.begin(), node.end());
    }
    return out;
}

// One row per repair_connectivity scenario. Fields left at their default (nullopt/empty)
// mirror an assertion the original cases didn't make either.
struct RepairCase {
    std::string name{};
    size_t n_nodes{};
    std::vector<std::vector<Idx>> adjacency{};
    std::vector<float> positions{};
    v::VamanaBuildParameters params{};
    size_t max_rounds = 4;
    size_t unreachable_before{};
    size_t unreachable_after{};
    std::optional<size_t> rounds{};
    bool rounds_is_min = false;
    std::optional<size_t> edges_total{};
    std::optional<size_t> edges_forced_unsafe{};
    std::optional<bool> stalled{};
    std::vector<Idx> exact_unreachable_ids{};
    bool check_unchanged = false;
};

} // namespace

CATCH_TEST_CASE("unreachable_nodes", "[index][vamana][connectivity]") {
    auto graph = build_test_graph(5, 3, {{1, 2}, {}, {3}});
    std::vector<Idx> roots{0};
    CATCH_REQUIRE(
        v::unreachable_nodes(graph, std::span<const Idx>(roots)) == std::vector<Idx>{4}
    );
    roots.push_back(4);
    CATCH_REQUIRE(v::unreachable_nodes(graph, std::span<const Idx>(roots)).empty());
}

CATCH_TEST_CASE("repair_connectivity", "[index][vamana][connectivity]") {
    std::vector<RepairCase> cases = {
        {.name = "leaves an already-connected graph untouched",
         .n_nodes = 5,
         .adjacency = {{1}, {2}, {3}, {4}},
         .positions = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F},
         .params = v::VamanaBuildParameters{1.2F, 3, 10, 10, 3, true},
         .unreachable_before = 0,
         .unreachable_after = 0,
         .rounds = 0,
         .edges_total = 0,
         .check_unchanged = true},
        {.name = "reconnects a disconnected node with room to spare",
         .n_nodes = 5,
         .adjacency = {{1}, {2}, {3}},
         .positions = {0.0F, 1.0F, 2.0F, 3.0F, 2.1F},
         .params = v::VamanaBuildParameters{1.2F, 3, 10, 10, 3, true},
         .unreachable_before = 1,
         .unreachable_after = 0,
         .rounds = 1,
         .rounds_is_min = true,
         .edges_total = 1},
        {.name = "reconnects when the nearest reachable neighbors are all at full degree",
         .n_nodes = 5,
         .adjacency = {{1, 2}, {2, 3}, {3, 0}, {0, 1}},
         .positions = {0.0F, 1.0F, 2.0F, 3.0F, 0.1F},
         .params = v::VamanaBuildParameters{1.2F, 2, 10, 10, 2, true},
         .unreachable_before = 1,
         .unreachable_after = 0,
         .edges_total = 1,
         .edges_forced_unsafe = 0,
         .stalled = false},
        {.name = "falls back to an unsafe forced replacement when no candidate is safe",
         .n_nodes = 4,
         .adjacency = {{1}, {2}},
         .positions = {100.0F, 0.0F, 2.0F, -3.0F},
         .params = v::VamanaBuildParameters{1.0F, 1, 1, 10, 1, true},
         .max_rounds = 1,
         .unreachable_before = 1,
         .unreachable_after = 1,
         .rounds = 1,
         .edges_total = 0,
         .edges_forced_unsafe = 1,
         .stalled = true,
         .exact_unreachable_ids = {2}},
        {.name = "stall guard stops an oscillating repair before max_rounds",
         .n_nodes = 4,
         .adjacency = {{1}, {2}},
         .positions = {100.0F, 0.0F, 2.0F, -3.0F},
         .params = v::VamanaBuildParameters{1.0F, 1, 1, 10, 1, true},
         .unreachable_before = 1,
         .unreachable_after = 1,
         .rounds = 1,
         .edges_forced_unsafe = 1,
         .stalled = true},
    };

    for (const auto& c : cases) {
        CATCH_SECTION(c.name) {
            auto graph =
                build_test_graph(c.n_nodes, c.params.graph_max_degree, c.adjacency);
            auto before = c.check_unchanged ? adjacency_snapshot(graph)
                                            : std::vector<std::vector<Idx>>{};
            auto data = make_positions(c.positions);
            svs::distance::DistanceL2 distance{};
            auto pool = svs::threads::DefaultThreadPool(1);
            std::vector<Idx> roots{0};
            auto report = v::repair_connectivity(
                graph,
                data,
                distance,
                c.params,
                std::span<const Idx>(roots),
                pool,
                c.max_rounds
            );

            CATCH_REQUIRE(report.unreachable_before == c.unreachable_before);
            CATCH_REQUIRE(report.unreachable_after == c.unreachable_after);
            if (c.rounds) {
                if (c.rounds_is_min) {
                    CATCH_REQUIRE(report.rounds >= *c.rounds);
                } else {
                    CATCH_REQUIRE(report.rounds == *c.rounds);
                }
            }
            if (c.edges_total) {
                CATCH_REQUIRE(
                    report.edges_appended + report.edges_accepted_through_prune +
                        report.edges_forced ==
                    *c.edges_total
                );
            }
            if (c.edges_forced_unsafe) {
                CATCH_REQUIRE(report.edges_forced_unsafe == *c.edges_forced_unsafe);
            }
            if (c.stalled) {
                CATCH_REQUIRE(report.stalled == *c.stalled);
            }
            CATCH_REQUIRE(
                v::unreachable_nodes(graph, std::span<const Idx>(roots)).size() ==
                report.unreachable_after
            );
            if (!c.exact_unreachable_ids.empty()) {
                CATCH_REQUIRE(
                    v::unreachable_nodes(graph, std::span<const Idx>(roots)) ==
                    c.exact_unreachable_ids
                );
            }
            svs_test::require_degree_bound(graph, c.params.graph_max_degree);
            if (c.check_unchanged) {
                CATCH_REQUIRE(adjacency_snapshot(graph) == before);
            }
        }
    }
}
