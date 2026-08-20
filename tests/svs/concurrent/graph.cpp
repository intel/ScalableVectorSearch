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

// header under test
#include "svs/concurrent/graph.h"

#include "svs/concepts/graph.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

using Graph = svs::concurrent::SeqLockGraph<uint32_t>;

// The headline claim of this design: the new graph satisfies the *unmodified* upstream
// concepts, so upstream VamanaBuilder / prune / consolidate work against it with no edits
// to any existing SVS header.
static_assert(svs::graphs::ImmutableMemoryGraph<Graph>);
static_assert(svs::graphs::MemoryGraph<Graph>);

} // namespace

CATCH_TEST_CASE("SeqLockGraph single-threaded semantics", "[concurrent][graph]") {
    Graph g{10, 4, /*segment_size=*/3};

    CATCH_REQUIRE(g.n_nodes() == 10);
    CATCH_REQUIRE(g.max_degree() == 4);
    CATCH_REQUIRE(g.get_node_degree(0) == 0);

    CATCH_SECTION("add_edge returns the new out-degree") {
        CATCH_REQUIRE(g.add_edge(0, 1) == 1);
        CATCH_REQUIRE(g.add_edge(0, 2) == 2);

        // Duplicate and self-loop are both rejected without changing the degree.
        CATCH_REQUIRE(g.add_edge(0, 1) == 2);
        CATCH_REQUIRE(g.add_edge(0, 0) == 2);

        CATCH_REQUIRE(g.get_node_degree(0) == 2);
        auto neighbors = g.get_node(0);
        CATCH_REQUIRE(neighbors.size() == 2);
        CATCH_REQUIRE(neighbors[0] == 1);
        CATCH_REQUIRE(neighbors[1] == 2);
    }

    CATCH_SECTION("adjacency lists saturate rather than overflow") {
        for (uint32_t dst = 1; dst <= 4; ++dst) {
            g.add_edge(0, dst);
        }
        CATCH_REQUIRE(g.get_node_degree(0) == 4);
        CATCH_REQUIRE(g.add_edge(0, 5) == 4);
        CATCH_REQUIRE(g.get_node_degree(0) == 4);
    }

    CATCH_SECTION("replace_node and clear_node") {
        g.add_edge(0, 1);
        g.add_edge(0, 2);

        std::vector<uint32_t> replacement{7, 8};
        g.replace_node(0, replacement);
        CATCH_REQUIRE(g.get_node_degree(0) == 2);
        auto neighbors = g.get_node(0);
        CATCH_REQUIRE(neighbors.size() == 2);
        CATCH_REQUIRE(neighbors[0] == 7);
        CATCH_REQUIRE(neighbors[1] == 8);

        g.clear_node(0);
        CATCH_REQUIRE(g.get_node_degree(0) == 0);
    }
}

CATCH_TEST_CASE("SeqLockGraph growth is address stable", "[concurrent][graph]") {
    // A small segment size forces many segments, so growth definitely crosses one.
    Graph g{3, 4, /*segment_size=*/3};
    g.add_edge(0, 1);
    const void* before = g.get_node(0).data();

    g.unsafe_resize(1000);
    CATCH_REQUIRE(g.n_nodes() == 1000);

    // This is the property the whole design rests on: a reader holding a pointer into
    // node 0's slot is unaffected by a writer growing the graph.
    CATCH_REQUIRE(g.get_node(0).data() == before);
    CATCH_REQUIRE(g.get_node_degree(0) == 1);

    // Newly added nodes are usable and zero-initialized.
    CATCH_REQUIRE(g.get_node_degree(999) == 0);
    CATCH_REQUIRE(g.add_edge(999, 1) == 1);
}

// A writer repeatedly rewrites one node's adjacency list while readers use the
// sequence-lock protocol. Readers assert that every list they *accept* is internally
// consistent, i.e. they never observe a torn mixture of two generations. A plain
// std::vector-backed graph fails this test.
CATCH_TEST_CASE("SeqLockGraph rejects torn concurrent reads", "[concurrent][graph]") {
    constexpr uint32_t kMaxDegree = 32;
#if defined(__SANITIZE_THREAD__) || defined(SVS_THREAD_SANITIZER)
    // ThreadSanitizer costs roughly an order of magnitude in time, and it is looking for
    // *races*, which show up just as readily in a shorter run.
    constexpr size_t kIterations = 20000;
#else
    constexpr size_t kIterations = 200000;
#endif

    Graph g{4, kMaxDegree, /*segment_size=*/2};

    std::atomic<bool> stop{false};
    std::atomic<size_t> torn_reads{0};
    std::atomic<size_t> accepted_reads{0};
    std::atomic<size_t> retries{0};

    // The writer alternates between two generations. Generation k writes a list of length
    // `len` where every element equals `k`, so any mixture of generations is detectable,
    // and the two generations have different lengths so a torn *degree* is detectable too.
    std::thread writer{[&] {
        std::vector<uint32_t> buffer;
        for (size_t i = 0; i < kIterations; ++i) {
            const uint32_t generation = (i % 2 == 0) ? 1u : 2u;
            const size_t len = (i % 2 == 0) ? kMaxDegree : 4;
            buffer.assign(len, generation);
            g.replace_node(0, buffer);
        }
        stop.store(true);
    }};

    auto reader_body = [&] {
        while (!stop.load(std::memory_order_relaxed)) {
            for (;;) {
                auto maybe_seq = g.seqlock(0).read_begin();
                if (!maybe_seq) {
                    retries.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                auto neighbors = g.get_node_atomic(0);
                bool uniform = true;
                const uint32_t first = neighbors.empty() ? 0u : neighbors[0];
                for (size_t k = 0; k < neighbors.size(); ++k) {
                    if (neighbors[k] != first) {
                        uniform = false;
                    }
                    // A degree covering an unwritten slot would show up as 0 here, since
                    // segments are zero-initialized and no generation writes a 0.
                    if (neighbors[k] == 0) {
                        uniform = false;
                    }
                }
                if (!g.seqlock(0).read_validate(*maybe_seq)) {
                    retries.fetch_add(1, std::memory_order_relaxed);
                    continue; // Inconsistent read, correctly rejected.
                }
                // The sequence lock accepted this read, so it must be consistent.
                if (!uniform && !neighbors.empty()) {
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                }
                accepted_reads.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
    };

    std::vector<std::thread> readers;
    for (int i = 0; i < 8; ++i) {
        readers.emplace_back(reader_body);
    }
    writer.join();
    for (auto& t : readers) {
        t.join();
    }

    CATCH_INFO(
        "accepted reads: " << accepted_reads.load()
                           << ", rejected/retried: " << retries.load()
                           << ", torn accepted reads: " << torn_reads.load()
    );
    CATCH_REQUIRE(torn_reads.load() == 0);
    CATCH_REQUIRE(accepted_reads.load() > 0);
    // If this is zero then the test proved nothing: the writer never overlapped a reader.
    CATCH_REQUIRE(retries.load() > 0);
}
