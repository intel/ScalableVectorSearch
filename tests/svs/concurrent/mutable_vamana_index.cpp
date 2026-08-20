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

// Concurrency tests for ``svs::concurrent::MutableVamanaIndex``: correctness of searches
// issued while another thread inserts and deletes.
//
// Recall thresholds here are deliberately coarse. They exist to catch a graph that has been
// corrupted into uselessness, not to track search quality -- that is the job of the
// benchmark suite.
//
// Assertions inside the hot loops accumulate into counters and are checked once at the end.
// Catch2's assertion bookkeeping is not free, and these loops run for millions of
// iterations.

// header under test
#include "svs/concurrent/mutable_vamana_index.h"

#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/index/vamana/vamana_build.h"
#include "svs/lib/threads.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

// ThreadSanitizer costs roughly an order of magnitude in both time and memory, and it is
// looking for *races*, which show up just as readily in a small index. Shrink the problem
// rather than skipping the run.
#if defined(__SANITIZE_THREAD__) || defined(SVS_THREAD_SANITIZER)
constexpr size_t kInitialPoints = 4000;
constexpr size_t kIncrementalPoints = 1000;
constexpr size_t kNumQueries = 50;
#else
constexpr size_t kInitialPoints = 20000;
constexpr size_t kIncrementalPoints = 5000;
constexpr size_t kNumQueries = 200;
#endif

constexpr size_t kDim = 32;
constexpr size_t kMaxDegree = 32;
constexpr size_t kNumNeighbors = 10;
constexpr size_t kBuildThreads = 8;

using Idx = uint32_t;
using Alloc = svs::lib::Allocator<float>;
using BlockedData = svs::data::BlockedData<float, svs::Dynamic, Alloc>;
using Distance = svs::distance::DistanceL2;
using ConcurrentIndex = svs::concurrent::MutableVamanaIndex<Idx, BlockedData, Distance>;

std::vector<float> random_vectors(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, 1.0f};
    std::vector<float> out(n * dim);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

svs::data::SimpleData<float> make_dataset(const std::vector<float>& raw, size_t dim) {
    const size_t n = raw.size() / dim;
    auto data = svs::data::SimpleData<float>(n, dim);
    for (size_t i = 0; i < n; ++i) {
        data.set_datum(i, std::span<const float>(raw.data() + i * dim, dim));
    }
    return data;
}

// Build a concurrent index over ``raw``, using the *unmodified* upstream builder. That this
// compiles at all is a load-bearing part of the design: ``SeqLockGraph`` satisfies
// ``svs::graphs::MemoryGraph`` as published, so ``VamanaBuilder`` needs no changes.
std::unique_ptr<ConcurrentIndex> build_index(
    const std::vector<float>& raw, size_t dim, std::span<const size_t> ids, size_t threads
) {
    const size_t n = raw.size() / dim;
    auto data = BlockedData(n, dim);
    for (size_t i = 0; i < n; ++i) {
        data.set_datum(i, std::span<const float>(raw.data() + i * dim, dim));
    }

    auto threadpool =
        svs::threads::ThreadPoolHandle{svs::threads::DefaultThreadPool{threads}};
    auto distance = Distance{};
    auto entry_point =
        svs::index::vamana::extensions::compute_entry_point(data, threadpool);

    auto graph = svs::concurrent::SeqLockGraph<Idx>{n, kMaxDegree};
    auto parameters = svs::index::vamana::VamanaBuildParameters{
        1.2f, kMaxDegree, 2 * kMaxDegree, 750, kMaxDegree, true};
    auto prefetch = svs::index::vamana::extensions::estimate_prefetch_parameters(data);
    auto builder = svs::index::vamana::VamanaBuilder{
        graph, data, distance, parameters, threadpool, prefetch};
    builder.construct(1.2f, static_cast<Idx>(entry_point));

    return std::make_unique<ConcurrentIndex>(
        std::move(graph),
        std::move(data),
        static_cast<Idx>(entry_point),
        distance,
        ids,
        svs::threads::DefaultThreadPool{threads}
    );
}

// Brute-force ground truth over the given set of live external IDs.
std::vector<std::vector<size_t>> ground_truth(
    const std::vector<float>& base,
    const std::unordered_set<size_t>& live,
    const std::vector<float>& queries,
    size_t dim,
    size_t k
) {
    const size_t nq = queries.size() / dim;
    std::vector<std::vector<size_t>> result(nq);
    for (size_t q = 0; q < nq; ++q) {
        std::vector<std::pair<float, size_t>> scored;
        scored.reserve(live.size());
        for (size_t id : live) {
            float d = 0;
            for (size_t j = 0; j < dim; ++j) {
                float diff = queries[q * dim + j] - base[id * dim + j];
                d += diff * diff;
            }
            scored.emplace_back(d, id);
        }
        std::partial_sort(
            scored.begin(),
            scored.begin() + static_cast<long>(std::min(k, scored.size())),
            scored.end()
        );
        for (size_t i = 0; i < std::min(k, scored.size()); ++i) {
            result[q].push_back(scored[i].second);
        }
    }
    return result;
}

double recall_at_k(
    const svs::QueryResult<size_t>& got, const std::vector<std::vector<size_t>>& expected
) {
    size_t hits = 0, total = 0;
    for (size_t q = 0; q < expected.size(); ++q) {
        std::unordered_set<size_t> truth{expected[q].begin(), expected[q].end()};
        for (size_t j = 0; j < got.n_neighbors(); ++j) {
            if (truth.count(got.index(q, j))) {
                ++hits;
            }
        }
        total += truth.size();
    }
    return total == 0 ? 1.0 : static_cast<double>(hits) / static_cast<double>(total);
}

// Insert ``[first, first + n)`` of ``base`` as a single batch.
void add_batch(
    ConcurrentIndex& index, const std::vector<float>& base, size_t first, size_t n
) {
    auto batch = svs::data::SimpleData<float>(n, kDim);
    std::vector<size_t> batch_ids(n);
    for (size_t i = 0; i < n; ++i) {
        batch.set_datum(i, std::span<const float>(base.data() + (first + i) * kDim, kDim));
        batch_ids[i] = first + i;
    }
    index.add_points(batch, batch_ids);
}

} // namespace

CATCH_TEST_CASE("Concurrent MutableVamanaIndex quiescent recall", "[concurrent][index]") {
    auto base = random_vectors(kInitialPoints, kDim, 1234);
    std::vector<size_t> ids(kInitialPoints);
    std::iota(ids.begin(), ids.end(), 0);

    auto index = build_index(base, kDim, ids, kBuildThreads);
    CATCH_REQUIRE(index->size() == kInitialPoints);

    auto queries_raw = random_vectors(kNumQueries, kDim, 999);
    auto queries = make_dataset(queries_raw, kDim);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    auto results = svs::QueryResult<size_t>{kNumQueries, kNumNeighbors};
    index->search(results.view(), queries, index->get_search_parameters());

    std::unordered_set<size_t> live{ids.begin(), ids.end()};
    auto truth = ground_truth(base, live, queries_raw, kDim, kNumNeighbors);
    const double recall = recall_at_k(results, truth);

    // A correctly built Vamana graph at this window size should be well above 0.9.
    CATCH_INFO("quiescent recall@" << kNumNeighbors << " = " << recall);
    CATCH_REQUIRE(recall > 0.90);
}

// The core test: searches run continuously while a writer inserts new vectors and deletes
// existing ones. Any torn adjacency read, use-after-free from a resize, or missing ID
// translation surfaces as a crash, an exception, or an invalid ID.
CATCH_TEST_CASE(
    "Concurrent MutableVamanaIndex search during mutation", "[concurrent][index]"
) {
    const size_t total = kInitialPoints + kIncrementalPoints;
    auto base = random_vectors(total, kDim, 4321);

    std::vector<size_t> initial_ids(kInitialPoints);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    // Build over the first ``kInitialPoints`` only; the tail is inserted concurrently
    // below.
    auto initial_slice = std::vector<float>(
        base.begin(), base.begin() + static_cast<long>(kInitialPoints * kDim)
    );
    auto index = build_index(initial_slice, kDim, initial_ids, kBuildThreads);

    auto sp = index->get_search_parameters();
    sp.buffer_config({100});
    index->set_search_parameters(sp);

    auto queries_raw = random_vectors(kNumQueries, kDim, 777);
    auto queries = make_dataset(queries_raw, kDim);

    std::atomic<bool> writer_done{false};
    std::atomic<size_t> searches_completed{0};
    std::atomic<size_t> invalid_ids{0};
    std::atomic<size_t> deleted_mid_search{0};
    std::atomic<size_t> duplicate_ids{0};
    std::atomic<size_t> exceptions{0};

    // Writer: insert the incremental points in batches, deleting some older ones as it goes
    // so that the deletion path is exercised too.
    std::thread writer{[&] {
        try {
            constexpr size_t kBatch = 500;
            for (size_t offset = 0; offset < kIncrementalPoints; offset += kBatch) {
                const size_t n = std::min(kBatch, kIncrementalPoints - offset);
                add_batch(*index, base, kInitialPoints + offset, n);

                // Delete a fresh, disjoint set of the original IDs each round, *spread*
                // across the whole ID range with a stride. A contiguous low-ID slice would
                // almost never intersect a query's top-k, so the interesting race -- a
                // result slot retired between selection and ID translation -- would go
                // unexercised. Round r takes every 40th ID starting at r, so the rounds are
                // disjoint and together retire 25% of the original vectors.
                const size_t round = offset / kBatch;
                std::vector<size_t> to_delete;
                for (size_t id = round; id < kInitialPoints; id += 40) {
                    to_delete.push_back(id);
                }
                index->delete_entries(to_delete);
            }
        } catch (const std::exception& e) {
            CATCH_WARN("writer threw: " << e.what());
            exceptions.fetch_add(1);
        }
        writer_done.store(true);
    }};

    // Searchers: hammer the index with single-query searches throughout.
    auto searcher = [&] {
        try {
            auto scratch = index->scratchspace();
            std::unordered_set<uint32_t> seen_ids;
            while (!writer_done.load(std::memory_order_relaxed)) {
                for (size_t q = 0; q < kNumQueries; ++q) {
                    auto query =
                        std::span<const float>(queries_raw.data() + q * kDim, kDim);
                    index->search(query, scratch);
                    // ``[0, valid())`` is the region a caller is allowed to read: skipped
                    // (deleted) candidates have been compacted out by this point.
                    const size_t n =
                        std::min<size_t>(kNumNeighbors, scratch.buffer.valid());
                    seen_ids.clear();
                    for (size_t j = 0; j < n; ++j) {
                        auto internal = scratch.buffer[j].id();
                        // A ranked result must never list the same vector twice. Torn
                        // adjacency reads or a botched retry in the seqlock section would
                        // show up here, because the search buffer dedupes by ID and can
                        // only be fooled by inconsistent input.
                        if (!seen_ids.insert(internal).second) {
                            duplicate_ids.fetch_add(1, std::memory_order_relaxed);
                        }
                        // Every ID a search hands back must either translate to a live
                        // external ID, or be a slot retired by the concurrent deleter.
                        // Anything else means the graph led us to a node the translator
                        // never knew about -- e.g. a slot published before its ID mapping.
                        auto external = index->try_translate_internal_id(internal);
                        if (external == ConcurrentIndex::invalid_external_id) {
                            deleted_mid_search.fetch_add(1, std::memory_order_relaxed);
                        } else if (!index->has_id(external)) {
                            invalid_ids.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    searches_completed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } catch (const std::exception& e) {
            CATCH_WARN("searcher threw: " << e.what());
            exceptions.fetch_add(1);
        }
    };

    std::vector<std::thread> searchers;
    for (int i = 0; i < 8; ++i) {
        searchers.emplace_back(searcher);
    }
    writer.join();
    for (auto& t : searchers) {
        t.join();
    }

    CATCH_INFO(
        "searches completed: " << searches_completed.load()
                               << ", neighbors retired mid-search (expected, dropped): "
                               << deleted_mid_search.load()
    );
    CATCH_REQUIRE(exceptions.load() == 0);
    CATCH_REQUIRE(invalid_ids.load() == 0);
    CATCH_REQUIRE(duplicate_ids.load() == 0);
    CATCH_REQUIRE(searches_completed.load() > 0);

    // Post-mutation the index must still be a correct index.
    std::unordered_set<size_t> live;
    index->on_ids([&live](size_t id) { live.insert(id); });
    CATCH_REQUIRE(live.size() == index->size());

    auto truth = ground_truth(base, live, queries_raw, kDim, kNumNeighbors);
    auto results = svs::QueryResult<size_t>{kNumQueries, kNumNeighbors};
    index->search(results.view(), queries, index->get_search_parameters());
    const double recall = recall_at_k(results, truth);
    CATCH_INFO("post-mutation recall@" << kNumNeighbors << " = " << recall);
    CATCH_REQUIRE(recall > 0.85);

    // And consolidation/compaction must leave it correct too.
    index->consolidate();
    index->compact();
    auto results2 = svs::QueryResult<size_t>{kNumQueries, kNumNeighbors};
    index->search(results2.view(), queries, index->get_search_parameters());
    const double recall2 = recall_at_k(results2, truth);
    CATCH_INFO("post-consolidate/compact recall@" << kNumNeighbors << " = " << recall2);
    CATCH_REQUIRE(recall2 > 0.85);
    CATCH_REQUIRE(index->size() == live.size());
}

// The batch iterator runs upstream's *unmodified* ``BatchIterator`` over a
// ``SeqLockGraphView``. This checks it works at all, and that it keeps working while a
// writer mutates the index.
CATCH_TEST_CASE("Concurrent MutableVamanaIndex batch iterator", "[concurrent][index]") {
    const size_t total = kInitialPoints + kIncrementalPoints;
    auto base = random_vectors(total, kDim, 24680);

    std::vector<size_t> initial_ids(kInitialPoints);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    auto initial_slice = std::vector<float>(
        base.begin(), base.begin() + static_cast<long>(kInitialPoints * kDim)
    );
    auto index = build_index(initial_slice, kDim, initial_ids, kBuildThreads);

    auto queries_raw = random_vectors(kNumQueries, kDim, 13579);

    CATCH_SECTION("quiescent") {
        // Batches must be non-overlapping, and each batch must be sorted.
        //
        // Batches are *not* globally monotonic and it would be wrong to assert that: the
        // iterator is approximate, so a later batch can surface a vector closer than one
        // the earlier batch's window had already returned. Count those inversions and
        // report them as a quality signal rather than a correctness one -- this is upstream
        // behaviour and has nothing to do with concurrency.
        auto query = std::span<const float>(queries_raw.data(), kDim);
        auto it = index->make_batch_iterator(query);
        std::unordered_set<size_t> all;
        float previous_worst = -1.0f;
        size_t batches = 0;
        size_t inversions = 0;
        size_t repeats = 0;
        size_t unsorted = 0;
        for (; batches < 5 && !it.done(); ++batches) {
            it.next(10);
            float last = -1.0f;
            for (const auto& n : it) {
                if (!all.insert(n.id()).second) {
                    ++repeats;
                }
                if (n.distance() < last) {
                    ++unsorted;
                }
                last = n.distance();
                if (n.distance() < previous_worst) {
                    ++inversions;
                }
            }
            if (it.size() != 0) {
                previous_worst = (it.end() - 1)->distance();
            }
        }
        CATCH_INFO(
            batches << " batches, " << all.size() << " distinct vectors, " << inversions
                    << " cross-batch inversions"
        );
        CATCH_REQUIRE(repeats == 0);
        CATCH_REQUIRE(unsorted == 0);
        CATCH_REQUIRE(all.size() >= 40);
    }

    CATCH_SECTION("during mutation") {
        // The claim under test is memory safety and per-batch consistency, *not* that a
        // long-lived cursor sees a stable snapshot.
        std::atomic<bool> writer_done{false};
        std::atomic<size_t> batches_completed{0};
        std::atomic<size_t> exceptions{0};
        std::atomic<size_t> bad_ids{0};

        std::thread writer{[&] {
            try {
                constexpr size_t kBatch = 500;
                for (size_t offset = 0; offset < kIncrementalPoints; offset += kBatch) {
                    const size_t n = std::min(kBatch, kIncrementalPoints - offset);
                    add_batch(*index, base, kInitialPoints + offset, n);
                }
            } catch (const std::exception& e) {
                CATCH_WARN("writer threw: " << e.what());
                exceptions.fetch_add(1);
            }
            writer_done.store(true);
        }};

        auto reader = [&] {
            try {
                while (!writer_done.load(std::memory_order_relaxed)) {
                    for (size_t q = 0; q < kNumQueries; ++q) {
                        auto query =
                            std::span<const float>(queries_raw.data() + q * kDim, kDim);
                        auto it = index->make_batch_iterator(query);
                        for (size_t b = 0; b < 3 && !it.done(); ++b) {
                            it.next(10);
                            for (const auto& n : it) {
                                // Any ID the iterator yields must be a real external ID (or
                                // one retired mid-flight); a bad graph read shows up here.
                                if (n.id() != ConcurrentIndex::invalid_external_id &&
                                    !index->has_id(n.id())) {
                                    bad_ids.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                            batches_completed.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            } catch (const std::exception& e) {
                CATCH_WARN("reader threw: " << e.what());
                exceptions.fetch_add(1);
            }
        };

        std::vector<std::thread> readers;
        for (int i = 0; i < 4; ++i) {
            readers.emplace_back(reader);
        }
        writer.join();
        for (auto& t : readers) {
            t.join();
        }

        CATCH_INFO("batches completed during mutation: " << batches_completed.load());
        CATCH_REQUIRE(exceptions.load() == 0);
        CATCH_REQUIRE(bad_ids.load() == 0);
        CATCH_REQUIRE(batches_completed.load() > 0);
    }
}
