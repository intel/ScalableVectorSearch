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

// Orchestrator under test
#include "svs/orchestrators/concurrent_dynamic_vamana.h"

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/core/recall.h"
#include "svs/lib/file.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <atomic>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

namespace cc = svs::index::vamana::concurrent;
using ConcurrentData = cc::SegmentedBlockedData<float>;

const size_t num_threads = 2;

// Allocator class which records allocated bytes
template <typename T> class RecordingAllocator {
  public:
    using value_type = T;

    RecordingAllocator() = default;

    T* allocate(size_t n) {
        *allocated_bytes += n * sizeof(T);
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, size_t n) {
        *allocated_bytes -= n * sizeof(T);
        ::operator delete(p);
    }

    template <typename U> bool operator==(const RecordingAllocator<U>& other) const {
        return allocated_bytes == other.allocated_bytes;
    }
    template <typename U> bool operator!=(const RecordingAllocator<U>& other) const {
        return !(*this == other);
    }

    template <typename U>
    RecordingAllocator(const RecordingAllocator<U>& other)
        : allocated_bytes(other.allocated_bytes) {}

    size_t& allocated() { return *allocated_bytes; }

    std::shared_ptr<size_t> allocated_bytes = std::make_shared<size_t>(0);
};

ConcurrentData load_data() { return ConcurrentData::load(test_dataset::data_svs_file()); }

std::vector<size_t> iota_ids(size_t n, size_t start = 0) {
    auto ids = std::vector<size_t>(n);
    std::iota(ids.begin(), ids.end(), start);
    return ids;
}

svs::index::vamana::VamanaBuildParameters small_build_parameters() {
    return svs::index::vamana::VamanaBuildParameters{1.2, 32, 64, 128, 28, true};
}

// Copy rows [begin, end) of `data` into a plain dataset suitable for `add_points`.
svs::data::SimpleData<float> rows(const ConcurrentData& data, size_t begin, size_t end) {
    auto out = svs::data::SimpleData<float>(end - begin, data.dimensions());
    for (size_t i = begin; i < end; ++i) {
        out.set_datum(i - begin, data.get_datum(i));
    }
    return out;
}

svs::ConcurrentDynamicVamana build_index(const ConcurrentData& data, size_t n) {
    auto subset = ConcurrentData(n, data.dimensions());
    for (size_t i = 0; i < n; ++i) {
        subset.set_datum(i, data.get_datum(i));
    }
    auto ids = iota_ids(n);
    return svs::ConcurrentDynamicVamana::build<float>(
        small_build_parameters(), std::move(subset), ids, svs::DistanceType::L2, num_threads
    );
}

template <typename Distance, typename... GraphAllocator>
svs::ConcurrentDynamicVamana
test_build(Distance distance, const GraphAllocator&... graph_allocator) {
    auto expected_result = test_dataset::vamana::expected_build_results(
        distance, svsbenchmark::Uncompressed(svs::DataType::float32)
    );
    auto build_params = expected_result.build_parameters_.value();
    auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    auto groundtruth = test_dataset::load_groundtruth(distance);

    auto data = load_data();
    const size_t n = data.size();
    auto ids = iota_ids(n);

    auto index = svs::ConcurrentDynamicVamana::build<float>(
        build_params, std::move(data), ids, distance, num_threads, graph_allocator...
    );

    CATCH_REQUIRE(index.size() == n);
    CATCH_REQUIRE(index.get_alpha() == Catch::Approx(build_params.alpha));
    CATCH_REQUIRE(index.get_graph_max_degree() == build_params.graph_max_degree);
    CATCH_REQUIRE(index.get_num_threads() == num_threads);
    CATCH_REQUIRE(index.has_id(0));
    CATCH_REQUIRE(index.has_id(n - 1));

    const double epsilon = 0.01;
    for (const auto& expected : expected_result.config_and_recall_) {
        auto these_queries = test_dataset::get_test_set(queries, expected.num_queries_);
        auto these_groundtruth =
            test_dataset::get_test_set(groundtruth, expected.num_queries_);
        index.set_search_parameters(expected.search_parameters_);
        auto results = index.search(these_queries, expected.num_neighbors_);
        double recall = svs::k_recall_at_n(
            these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );
        CATCH_REQUIRE(recall > expected.recall_ - epsilon);
        CATCH_REQUIRE(recall < expected.recall_ + epsilon);
    }
    return index;
}

void require_same_results(
    svs::ConcurrentDynamicVamana& expected, svs::DynamicVamana& actual, size_t k
) {
    auto queries = test_dataset::queries();
    auto a = expected.search(queries, k);
    auto b = actual.search(queries, k);
    for (size_t q = 0; q < queries.size(); ++q) {
        for (size_t i = 0; i < k; ++i) {
            CATCH_REQUIRE(a.index(q, i) == b.index(q, i));
        }
    }
}

} // namespace

CATCH_TEST_CASE(
    "ConcurrentDynamicVamana Build", "[managers][concurrent_dynamic_vamana][build]"
) {
    for (auto distance_enum : test_dataset::vamana::available_build_distances()) {
        CATCH_SECTION(std::string("Functor ") + std::string(svs::name(distance_enum))) {
            svs::DistanceDispatcher dispatcher(distance_enum);
            dispatcher([&](auto distance) { test_build(distance); });
        }
    }
}

CATCH_TEST_CASE(
    "ConcurrentDynamicVamana Build with Graph Allocator",
    "[managers][concurrent_dynamic_vamana][build]"
) {
    using GraphAllocator = cc::SegmentedBlocked<RecordingAllocator<uint32_t>>;
    auto blocking = svs::data::BlockingParameters{};
    blocking.blocksize_elements = svs::lib::PowerOfTwo(7);
    const size_t blocksize = blocking.blocksize_elements->value();
    auto recorder = RecordingAllocator<uint32_t>{};

    auto index =
        test_build(svs::distance::DistanceL2{}, GraphAllocator{blocking, recorder});
    const size_t n = index.size();

    // Graph capacity is a whole number of the custom blocks, not the 1 GiB default.
    const size_t node_bytes = (index.get_graph_max_degree() + 1) * sizeof(uint32_t);
    const size_t capacity = (n + blocksize - 1) / blocksize * blocksize;
    auto breakdown = index.get_memory_breakdown();
    CATCH_REQUIRE(breakdown.graph_bytes == capacity * node_bytes);
    // The reverse-edge index is allocated through the graph allocator as well.
    const size_t allocated = recorder.allocated();
    CATCH_REQUIRE(allocated > breakdown.graph_bytes);

    // Growing past the current capacity appends exactly one more custom-sized block.
    const size_t num_new = capacity - n + 1;
    auto new_points = rows(load_data(), 0, num_new);
    index.add_points(new_points.cview(), iota_ids(num_new, n));
    CATCH_REQUIRE(index.size() == n + num_new);
    breakdown = index.get_memory_breakdown();
    CATCH_REQUIRE(breakdown.graph_bytes == (capacity + blocksize) * node_bytes);
    CATCH_REQUIRE(recorder.allocated() >= allocated + blocksize * node_bytes);
}

CATCH_TEST_CASE(
    "ConcurrentDynamicVamana Mutation", "[managers][concurrent_dynamic_vamana]"
) {
    auto data = load_data();
    const size_t n = data.size();
    const size_t half = n / 2;
    auto index = build_index(data, half);
    CATCH_REQUIRE(index.size() == half);

    const size_t usage_before = index.get_memory_breakdown().total();
    auto rest = rows(data, half, n);
    index.add_points(rest.cview(), iota_ids(n - half, half));
    CATCH_REQUIRE(index.size() == n);
    CATCH_REQUIRE(index.get_memory_breakdown().total() > usage_before);

    auto to_delete = iota_ids(half / 2);
    index.delete_points(to_delete);
    CATCH_REQUIRE(index.size() == n - to_delete.size());
    CATCH_REQUIRE_FALSE(index.has_id(0));
    CATCH_REQUIRE(index.has_id(n - 1));

    index.consolidate().compact();
    CATCH_REQUIRE(index.size() == n - to_delete.size());

    auto all = index.all_ids();
    CATCH_REQUIRE(all.size() == index.size());
    auto unique = std::unordered_set<size_t>(all.begin(), all.end());
    CATCH_REQUIRE(unique.size() == all.size());
    CATCH_REQUIRE(unique.count(0) == 0);

    // Distance and reconstruction for a surviving id.
    const size_t id = n - 1;
    auto datum = data.get_datum(id);
    auto query = std::vector<float>(datum.begin(), datum.end());
    CATCH_REQUIRE(index.get_distance(id, query) == Catch::Approx(0.0).margin(1e-3));

    auto reconstructed = svs::data::SimpleData<float>(1, data.dimensions());
    auto reconstruct_ids = std::vector<uint64_t>{id};
    index.reconstruct_at(reconstructed.view(), reconstruct_ids);
    for (size_t j = 0; j < data.dimensions(); ++j) {
        CATCH_REQUIRE(reconstructed.get_datum(0)[j] == datum[j]);
    }

    CATCH_REQUIRE(
        index.experimental_backend_string().find("concurrent") != std::string::npos
    );
}

CATCH_TEST_CASE(
    "ConcurrentDynamicVamana Batch Iterator", "[managers][concurrent_dynamic_vamana]"
) {
    auto data = load_data();
    auto index = build_index(data, data.size());
    auto queries = test_dataset::queries();
    auto query = std::span<const float>(queries.get_datum(0));

    auto iterator = index.batch_iterator(query);
    auto seen = std::unordered_set<size_t>();
    const size_t batch_size = 10;
    for (size_t batch = 0; batch < 5; ++batch) {
        iterator.next(batch_size);
        CATCH_REQUIRE(iterator.size() == batch_size);
        for (const auto& neighbor : iterator.results()) {
            CATCH_REQUIRE(index.has_id(neighbor.id()));
            CATCH_REQUIRE(seen.insert(neighbor.id()).second);
        }
    }

    // The first batch matches a regular search.
    auto expected = index.search(queries, batch_size);
    auto restarted = index.batch_iterator(query);
    restarted.next(batch_size);
    auto first = std::unordered_set<size_t>();
    for (const auto& neighbor : restarted.results()) {
        first.insert(neighbor.id());
    }
    size_t overlap = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        overlap += first.count(expected.index(0, i));
    }
    CATCH_REQUIRE(overlap >= batch_size - 1);
}

CATCH_TEST_CASE(
    "ConcurrentDynamicVamana Save and Load", "[managers][concurrent_dynamic_vamana]"
) {
    auto data = load_data();
    auto index = build_index(data, data.size());
    const size_t k = 10;

    CATCH_SECTION("Directories") {
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        index.save(dir / "config", dir / "graph", dir / "data");
        svs::DynamicVamana loaded = svs::ConcurrentDynamicVamana::assemble<float>(
            dir / "config",
            SVS_LAZY(cc::graphs::SimpleBlockedGraph<uint32_t>::load(dir / "graph")),
            SVS_LAZY(ConcurrentData::load(dir / "data")),
            svs::DistanceType::L2,
            num_threads
        );
        CATCH_REQUIRE(loaded.size() == index.size());
        require_same_results(index, loaded, k);
    }

    CATCH_SECTION("Native stream") {
        std::stringstream stream;
        index.save(stream);
        svs::DynamicVamana loaded =
            svs::ConcurrentDynamicVamana::assemble<float, ConcurrentData>(
                stream, svs::distance::DistanceL2(), num_threads
            );
        CATCH_REQUIRE(loaded.size() == index.size());
        require_same_results(index, loaded, k);
    }

    CATCH_SECTION("Directory archive stream") {
        std::stringstream stream;
        {
            svs::lib::UniqueTempDirectory tempdir{"svs_concurrent_orchestrator_save"};
            index.save(
                tempdir.get() / "config", tempdir.get() / "graph", tempdir.get() / "data"
            );
            svs::lib::DirectoryArchiver::pack(tempdir, stream);
        }
        svs::DynamicVamana loaded =
            svs::ConcurrentDynamicVamana::assemble<float, ConcurrentData>(
                stream, svs::DistanceType::L2, num_threads
            );
        CATCH_REQUIRE(loaded.size() == index.size());
        require_same_results(index, loaded, k);
    }
}

CATCH_TEST_CASE(
    "ConcurrentDynamicVamana Concurrent Search and Add",
    "[managers][concurrent_dynamic_vamana]"
) {
    auto data = load_data();
    const size_t n = data.size();
    const size_t initial = n / 2;
    auto index = build_index(data, initial);
    auto queries = test_dataset::queries();

    constexpr size_t num_writers = 2;
    constexpr size_t num_readers = 2;
    const size_t per_writer = (n - initial) / num_writers;

    std::atomic<size_t> failures{0};
    std::vector<std::thread> threads;
    for (size_t w = 0; w < num_writers; ++w) {
        threads.emplace_back([&, w]() {
            const size_t begin = initial + w * per_writer;
            for (size_t i = begin; i < begin + per_writer; i += 8) {
                const size_t end = std::min(i + 8, begin + per_writer);
                auto points = rows(data, i, end);
                try {
                    index.add_points(points.cview(), iota_ids(end - i, i));
                } catch (...) { failures.fetch_add(1); }
            }
        });
    }
    for (size_t r = 0; r < num_readers; ++r) {
        threads.emplace_back([&]() {
            for (size_t iter = 0; iter < 20; ++iter) {
                try {
                    auto results = index.search(queries, 10);
                    for (size_t q = 0; q < results.n_queries(); ++q) {
                        if (results.index(q, 0) >= n) {
                            failures.fetch_add(1);
                        }
                    }
                } catch (...) { failures.fetch_add(1); }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    CATCH_REQUIRE(failures.load() == 0);
    CATCH_REQUIRE(index.size() == initial + num_writers * per_writer);
    for (size_t i = initial; i < initial + num_writers * per_writer; ++i) {
        CATCH_REQUIRE(index.has_id(i));
    }
}
