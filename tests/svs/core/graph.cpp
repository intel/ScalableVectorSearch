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

// svs
#include "svs/concepts/graph.h"
#include "svs/core/graph/graph.h"

// test utils
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

CATCH_TEST_CASE("Simple Graph", "[graphs][simple]") {
    using Idx = uint32_t;
    size_t n_nodes = 10;
    const size_t max_degree = 5;

    auto graph = svs::graphs::SimpleGraph<Idx>(n_nodes, max_degree);
    CATCH_REQUIRE(graph.n_nodes() == n_nodes);
    CATCH_REQUIRE(graph.max_degree() == max_degree);

    // Ensure the constructor sets initializes the adjacency list sizes to zero.
    for (Idx i = 0; i < graph.n_nodes(); ++i) {
        CATCH_REQUIRE(graph.get_node_degree(i) == 0);
    }

    // Test edge adding
    auto check_increments = [n_nodes](auto&& span, Idx start) {
        for (Idx i = 0; i < span.size(); ++i) {
            CATCH_REQUIRE(span[i] == (start + i) % n_nodes);
        }
    };

    for (size_t i = 0; i <= max_degree; ++i) {
        // Verify that the current state of the adjacency lists is consistent.
        for (Idx j = 0; j < n_nodes; ++j) {
            CATCH_REQUIRE(graph.get_node_degree(j) == i);
            auto l = graph.get_node(j);
            CATCH_REQUIRE(l.size() == i);
            check_increments(l, j + 1);
        }

        for (Idx j = 0; j < n_nodes; ++j) {
            bool should_be_added = (i < max_degree);
            auto dst = (j + i + 1) % n_nodes;
            CATCH_REQUIRE(!graph.has_edge(j, dst));
            graph.add_edge(j, dst);

            // Make sure that the edge is added (or not) depending on whether the graph
            // is full or not.
            if (should_be_added) {
                CATCH_REQUIRE(graph.has_edge(j, dst));
            } else {
                CATCH_REQUIRE(!graph.has_edge(j, dst));
            }

            // Filter out redundant assignments.
            graph.add_edge(j, dst);
        }
    }

    // The last round should have added more than the acceptable number of neighbors.
    // Make sure we didn't overwrite anything.
    for (Idx j = 0; j < n_nodes; ++j) {
        CATCH_REQUIRE(graph.get_node_degree(j) == max_degree);
        auto l = graph.get_node(j);
        CATCH_REQUIRE(l.size() == max_degree);
        check_increments(l, j + 1);
    }

    graph.reset();
    for (Idx j = 0; j < n_nodes; ++j) {
        CATCH_REQUIRE(graph.get_node_degree(j) == 0);
    }

    // Replace Node.
    // Purposely use a replacement that is too large to verify the truncating logic.
    {
        std::vector<Idx> replacement{5, 4, 3, 2, 1, 6, 7, 8, 9};
        graph.replace_node(0, std::span{replacement.data(), replacement.size()});
        CATCH_REQUIRE(graph.get_node_degree(0) == max_degree);
        auto s = graph.get_node(0);
        CATCH_REQUIRE(s.size() == max_degree);
        std::array<Idx, max_degree> expected{5, 4, 3, 2, 1};
        CATCH_REQUIRE(expected.size() == max_degree);
        CATCH_REQUIRE(std::equal(s.begin(), s.end(), expected.begin()));
        // Make sure the next elements weren't touched.
        CATCH_REQUIRE(graph.get_node_degree(1) == 0);
    }

    // Now, use fewer than the max degree to make sure that works as well.
    {
        const size_t replacement_size = 3;
        std::vector<Idx> replacement{10, 7, 6};
        CATCH_REQUIRE(replacement.size() == replacement_size);

        Idx last = n_nodes - 1;
        graph.replace_node(last, std::span{replacement.data(), replacement.size()});
        CATCH_REQUIRE(graph.get_node_degree(last) == replacement.size());
        auto s = graph.get_node(last);
        CATCH_REQUIRE(s.size() == replacement.size());
        CATCH_REQUIRE(std::equal(s.begin(), s.end(), replacement.begin()));
    }
}
