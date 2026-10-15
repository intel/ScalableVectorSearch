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

// C API
#include "svs/c/svs_c.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

// Test utilities
#include "c_api_test_utils.h"

// Standard library
#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t DIMENSION = 32;
constexpr size_t NUM_VECTORS = 200;
constexpr size_t NUM_QUERIES = 4;
constexpr size_t K = 5;

// Base of the ID range used by writer threads; disjoint from the initial IDs.
constexpr size_t WRITER_ID_BASE = 100'000;

// Small blocks keep the tests fast; the default block size is 1 GiB.
constexpr size_t BLOCK_SIZE = 1 << 20;

// Catch2 assertions are not thread-safe, so worker threads record failures here and the
// main thread asserts on them after joining.
class FailureLog {
  public:
    void check(bool ok, svs_error_h err, const char* what) {
        if (ok) {
            return;
        }
        failures_.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock{mutex_};
        if (first_.empty()) {
            first_ = std::string{what} + ": " +
                     (err != nullptr ? svs_error_get_message(err) : "unknown error");
        }
    }

    size_t count() const { return failures_.load(); }
    std::string first() const {
        std::lock_guard lock{mutex_};
        return first_;
    }

  private:
    std::atomic<size_t> failures_{0};
    mutable std::mutex mutex_;
    std::string first_;
};

struct SyncFixture {
    svs_error_h error = svs_error_create();
    svs_algorithm_h algorithm = nullptr;
    svs_index_builder_h builder = nullptr;
    std::vector<float> data;
    std::vector<float> queries;
    std::vector<size_t> ids;

    SyncFixture() {
        generate_test_data(data, NUM_VECTORS, DIMENSION);
        generate_test_data(queries, NUM_QUERIES, DIMENSION);
        ids.resize(NUM_VECTORS);
        for (size_t i = 0; i < NUM_VECTORS; ++i) {
            ids[i] = i;
        }
        algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_NATIVE, 2, error);
    }

    ~SyncFixture() {
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    SyncFixture(const SyncFixture&) = delete;
    SyncFixture& operator=(const SyncFixture&) = delete;

    svs_index_h build(svs_sync_kind_t sync_kind) {
        svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        params.blocksize_bytes = BLOCK_SIZE;
        params.sync_kind = sync_kind;
        return svs_index_build_dynamic_ex(
            builder, data.data(), ids.data(), NUM_VECTORS, &params, error
        );
    }
};

bool is_known_id(size_t id, size_t num_writer_ids) {
    return id < NUM_VECTORS ||
           (id >= WRITER_ID_BASE && id < WRITER_ID_BASE + num_writer_ids);
}

// Runs concurrent writers (add/delete/consolidate/compact) and readers (search, has_id,
// get_distance, size, memory, conversion) on `index`, then verifies the final state.
void run_concurrent_workload(SyncFixture& fx, svs_index_h index) {
    constexpr size_t NUM_WRITERS = 2;
    constexpr size_t NUM_READERS = 4;
    constexpr size_t ITERATIONS = 20;
    constexpr size_t READER_ITERATIONS = 40;
    constexpr size_t BATCH = 8;
    constexpr size_t NUM_WRITER_IDS = NUM_WRITERS * ITERATIONS * BATCH;

    std::vector<float> batch_data;
    generate_test_data(batch_data, BATCH, DIMENSION);

    FailureLog log;

    auto writer = [&](size_t w) {
        svs_error_h err = svs_error_create();
        std::vector<size_t> batch_ids(BATCH);
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            const size_t base = WRITER_ID_BASE + (w * ITERATIONS + iter) * BATCH;
            for (size_t i = 0; i < BATCH; ++i) {
                batch_ids[i] = base + i;
            }

            size_t added = 0;
            log.check(
                svs_index_dynamic_add_points(
                    index, batch_data.data(), batch_ids.data(), BATCH, &added, err
                ),
                err,
                "add_points"
            );
            log.check(added == BATCH, err, "add_points count");

            size_t deleted = 0;
            log.check(
                svs_index_dynamic_delete_points(
                    index, batch_ids.data(), BATCH / 2, &deleted, err
                ),
                err,
                "delete_points"
            );
            log.check(deleted == BATCH / 2, err, "delete_points count");

            if (iter % 5 == 4) {
                log.check(svs_index_dynamic_consolidate(index, err), err, "consolidate");
                log.check(svs_index_dynamic_compact(index, 0, err), err, "compact");
            }
        }
        svs_error_free(err);
    };

    auto reader = [&](size_t r) {
        svs_error_h err = svs_error_create();
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        std::mt19937 rng(static_cast<unsigned>(r));
        // std::shared_mutex may prefer readers, so pause randomly to let writers in.
        std::uniform_int_distribution<int> pause_us(0, 500);
        for (size_t iter = 0; iter < READER_ITERATIONS; ++iter) {
            std::this_thread::sleep_for(std::chrono::microseconds(pause_us(rng)));

            bool ok = svs_index_search_topk(
                index, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, err
            );
            log.check(ok, err, "search_topk");
            if (ok) {
                for (size_t i = 0; i < results.total_results; ++i) {
                    log.check(
                        is_known_id(results.indices[i], NUM_WRITER_IDS),
                        err,
                        "search returned unknown id"
                    );
                }
            }

            bool has_id = false;
            const size_t base_id = (r * 31 + iter) % NUM_VECTORS;
            log.check(
                svs_index_dynamic_has_id(index, base_id, &has_id, err), err, "has_id"
            );
            log.check(has_id, err, "initial id missing");

            float distance = 0.0f;
            log.check(
                svs_index_get_distance(index, base_id, fx.queries.data(), &distance, err),
                err,
                "get_distance"
            );

            size_t size = 0;
            log.check(svs_index_get_size(index, &size, err), err, "get_size");
            log.check(size >= NUM_VECTORS, err, "index shrank below initial size");

            size_t memory = 0;
            log.check(svs_index_get_memory_usage(index, &memory, err), err, "memory_usage");

            if (r == 0 && iter % 8 == 0) {
                svs_index_h copy =
                    svs_index_convert_dynamic(fx.builder, index, BLOCK_SIZE, err);
                log.check(copy != nullptr, err, "convert_dynamic");
                svs_index_free(copy);
            }
        }
        svs_search_results_free(&results);
        svs_error_free(err);
    };

    std::vector<std::thread> threads;
    for (size_t w = 0; w < NUM_WRITERS; ++w) {
        threads.emplace_back(writer, w);
    }
    for (size_t r = 0; r < NUM_READERS; ++r) {
        threads.emplace_back(reader, r);
    }
    for (auto& t : threads) {
        t.join();
    }

    CATCH_INFO("First failure: " << log.first());
    CATCH_REQUIRE(log.count() == 0);

    // Final state: first half of every batch deleted, second half present.
    for (size_t w = 0; w < NUM_WRITERS; ++w) {
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            const size_t base = WRITER_ID_BASE + (w * ITERATIONS + iter) * BATCH;
            for (size_t i = 0; i < BATCH; ++i) {
                bool has_id = false;
                CATCH_REQUIRE(svs_index_dynamic_has_id(index, base + i, &has_id, fx.error));
                CATCH_REQUIRE(has_id == (i >= BATCH / 2));
            }
        }
    }

    size_t size = 0;
    CATCH_REQUIRE(svs_index_get_size(index, &size, fx.error));
    CATCH_REQUIRE(size == NUM_VECTORS + NUM_WRITER_IDS / 2);
}

} // namespace

CATCH_TEST_CASE("C API Dynamic Index Params", "[c_api][index][dynamic][sync]") {
    SyncFixture fx;
    CATCH_REQUIRE(fx.builder != nullptr);

    CATCH_SECTION("Default params build a usable index") {
        svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        svs_index_h index = svs_index_build_dynamic_ex(
            fx.builder, fx.data.data(), nullptr, NUM_VECTORS, &params, fx.error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(fx.error));
        size_t size = 0;
        CATCH_REQUIRE(svs_index_get_size(index, &size, fx.error));
        CATCH_REQUIRE(size == NUM_VECTORS);
        svs_index_free(index);
    }

    CATCH_SECTION("NULL params mean defaults") {
        svs_index_h index = svs_index_build_dynamic_ex(
            fx.builder, fx.data.data(), nullptr, NUM_VECTORS, nullptr, fx.error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(fx.error));
        svs_sync_kind_t kind = SVS_SYNC_KIND_GLOBAL;
        CATCH_REQUIRE(svs_index_dynamic_get_sync_kind(index, &kind, fx.error));
        CATCH_REQUIRE(kind == SVS_SYNC_KIND_NONE);

        TempDir dir;
        CATCH_REQUIRE(svs_index_save(index, dir.string().c_str(), fx.error));
        svs_index_h loaded =
            svs_index_load_dynamic_ex(fx.builder, dir.string().c_str(), nullptr, fx.error);
        CATCH_REQUIRE(loaded != nullptr);
        CATCH_REQUIRE(svs_error_ok(fx.error));
        CATCH_REQUIRE(svs_index_dynamic_get_sync_kind(loaded, &kind, fx.error));
        CATCH_REQUIRE(kind == SVS_SYNC_KIND_NONE);

        svs_index_h converted =
            svs_index_convert_dynamic_ex(fx.builder, index, nullptr, fx.error);
        CATCH_REQUIRE(converted != nullptr);
        CATCH_REQUIRE(svs_error_ok(fx.error));
        CATCH_REQUIRE(svs_index_dynamic_get_sync_kind(converted, &kind, fx.error));
        CATCH_REQUIRE(kind == SVS_SYNC_KIND_NONE);

        svs_memory_breakdown_t legacy = SVS_INIT_MEMORY_BREAKDOWN();
        CATCH_REQUIRE(svs_index_builder_estimate_memory_dynamic(
            fx.builder, NUM_VECTORS, 0, &legacy, fx.error
        ));
        svs_memory_breakdown_t by_null = SVS_INIT_MEMORY_BREAKDOWN();
        CATCH_REQUIRE(svs_index_builder_estimate_memory_dynamic_ex(
            fx.builder, NUM_VECTORS, nullptr, &by_null, fx.error
        ));
        CATCH_REQUIRE(by_null.graph_bytes == legacy.graph_bytes);
        CATCH_REQUIRE(by_null.data_bytes == legacy.data_bytes);
        CATCH_REQUIRE(by_null.metadata_bytes == legacy.metadata_bytes);

        size_t legacy_search = 0;
        CATCH_REQUIRE(svs_index_builder_estimate_search_memory_dynamic(
            fx.builder, NUM_QUERIES, K, nullptr, nullptr, 0, &legacy_search, fx.error
        ));
        size_t null_search = 0;
        CATCH_REQUIRE(svs_index_builder_estimate_search_memory_dynamic_ex(
            fx.builder, NUM_QUERIES, K, nullptr, nullptr, nullptr, &null_search, fx.error
        ));
        CATCH_REQUIRE(null_search == legacy_search);

        svs_index_free(converted);
        svs_index_free(loaded);
        svs_index_free(index);
    }

    CATCH_SECTION("Invalid params are rejected") {
        auto expect_invalid = [&](const svs_dynamic_index_params_t* params) {
            svs_index_h index = svs_index_build_dynamic_ex(
                fx.builder, fx.data.data(), nullptr, NUM_VECTORS, params, fx.error
            );
            CATCH_REQUIRE(index == nullptr);
            CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
        };

        svs_dynamic_index_params_t bad_bytes = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        bad_bytes.blocksize_bytes = 3000;
        expect_invalid(&bad_bytes);

        svs_dynamic_index_params_t bad_elements = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        bad_elements.blocksize_elements = 100;
        expect_invalid(&bad_elements);

        svs_dynamic_index_params_t bad_sync = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        bad_sync.sync_kind = 3;
        expect_invalid(&bad_sync);

        svs_dynamic_index_params_t bad_version = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        bad_version.version = svs_get_version() + 1;
        expect_invalid(&bad_version);

        svs_dynamic_index_params_t bad_size = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        bad_size.struct_size = sizeof(svs_dynamic_index_params_t) + 1;
        expect_invalid(&bad_size);

        // Other _ex entry points validate params too.
        svs_index_h source = fx.build(SVS_SYNC_KIND_NONE);
        CATCH_REQUIRE(source != nullptr);
        CATCH_REQUIRE(
            svs_index_convert_dynamic_ex(fx.builder, source, &bad_sync, fx.error) == nullptr
        );
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);

        TempDir dir;
        CATCH_REQUIRE(svs_index_save(source, dir.string().c_str(), fx.error));
        svs_index_free(source);
        CATCH_REQUIRE(
            svs_index_load_dynamic_ex(
                fx.builder, dir.string().c_str(), &bad_bytes, fx.error
            ) == nullptr
        );
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    CATCH_SECTION("Sync kind getter") {
        svs_sync_kind_t kind = SVS_SYNC_KIND_NONE;
        CATCH_REQUIRE_FALSE(svs_index_dynamic_get_sync_kind(nullptr, &kind, fx.error));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);

        svs_index_h index = fx.build(SVS_SYNC_KIND_GLOBAL);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE_FALSE(svs_index_dynamic_get_sync_kind(index, nullptr, fx.error));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
        svs_index_free(index);

        svs_index_h static_index =
            svs_index_build(fx.builder, fx.data.data(), NUM_VECTORS, fx.error);
        CATCH_REQUIRE(static_index != nullptr);
        CATCH_REQUIRE_FALSE(svs_index_dynamic_get_sync_kind(static_index, &kind, fx.error));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
        svs_index_free(static_index);
    }

    CATCH_SECTION("Fields beyond struct_size are ignored") {
        svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        params.struct_size = offsetof(svs_dynamic_index_params_t, blocksize_bytes);
        params.blocksize_bytes = 3000;   // invalid, but not covered by struct_size
        params.blocksize_elements = 100; // invalid, but not covered by struct_size
        params.sync_kind = 3;
        svs_index_h index = svs_index_build_dynamic_ex(
            fx.builder, fx.data.data(), nullptr, NUM_VECTORS, &params, fx.error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(fx.error));
        svs_index_free(index);
    }

    CATCH_SECTION("Memory estimates honor block parameters") {
        svs_memory_breakdown_t legacy = SVS_INIT_MEMORY_BREAKDOWN();
        CATCH_REQUIRE(svs_index_builder_estimate_memory_dynamic(
            fx.builder, NUM_VECTORS, BLOCK_SIZE, &legacy, fx.error
        ));

        svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        params.blocksize_bytes = BLOCK_SIZE;
        svs_memory_breakdown_t by_bytes = SVS_INIT_MEMORY_BREAKDOWN();
        CATCH_REQUIRE(svs_index_builder_estimate_memory_dynamic_ex(
            fx.builder, NUM_VECTORS, &params, &by_bytes, fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        CATCH_REQUIRE(by_bytes.graph_bytes == legacy.graph_bytes);
        CATCH_REQUIRE(by_bytes.data_bytes == legacy.data_bytes);
        CATCH_REQUIRE(by_bytes.metadata_bytes == legacy.metadata_bytes);

        // Small element-based blocks take precedence and shrink the padded allocation.
        params.blocksize_elements = 16;
        svs_memory_breakdown_t by_elements = SVS_INIT_MEMORY_BREAKDOWN();
        CATCH_REQUIRE(svs_index_builder_estimate_memory_dynamic_ex(
            fx.builder, NUM_VECTORS, &params, &by_elements, fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        CATCH_REQUIRE(by_elements.data_bytes < by_bytes.data_bytes);
        CATCH_REQUIRE(by_elements.graph_bytes < by_bytes.graph_bytes);

        size_t legacy_search = 0;
        CATCH_REQUIRE(svs_index_builder_estimate_search_memory_dynamic(
            fx.builder,
            NUM_QUERIES,
            K,
            nullptr,
            nullptr,
            BLOCK_SIZE,
            &legacy_search,
            fx.error
        ));
        size_t ex_search = 0;
        CATCH_REQUIRE(svs_index_builder_estimate_search_memory_dynamic_ex(
            fx.builder, NUM_QUERIES, K, nullptr, nullptr, &params, &ex_search, fx.error
        ));
        CATCH_REQUIRE(svs_error_ok(fx.error));
        CATCH_REQUIRE(ex_search == legacy_search);

        svs_dynamic_index_params_t bad_params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        bad_params.blocksize_elements = 100;
        CATCH_REQUIRE_FALSE(svs_index_builder_estimate_memory_dynamic_ex(
            fx.builder, NUM_VECTORS, &bad_params, &by_bytes, fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE_FALSE(svs_index_builder_estimate_search_memory_dynamic_ex(
            fx.builder, NUM_QUERIES, K, nullptr, nullptr, &bad_params, &ex_search, fx.error
        ));
        CATCH_REQUIRE(svs_error_get_code(fx.error) == SVS_ERROR_INVALID_ARGUMENT);
    }
}

CATCH_TEST_CASE("C API Dynamic Index Sync Sequential", "[c_api][index][dynamic][sync]") {
    // Every operation must work (and not self-deadlock) for every sync kind.
    const auto sync_kind =
        GENERATE(SVS_SYNC_KIND_NONE, SVS_SYNC_KIND_GLOBAL, SVS_SYNC_KIND_FINE_GRAIN);
    CATCH_CAPTURE(sync_kind);

    SyncFixture fx;
    svs_index_h index = fx.build(sync_kind);
    CATCH_REQUIRE(index != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));

    svs_sync_kind_t actual_kind = SVS_SYNC_KIND_NONE;
    CATCH_REQUIRE(svs_index_dynamic_get_sync_kind(index, &actual_kind, fx.error));
    CATCH_REQUIRE(actual_kind == sync_kind);

    std::vector<size_t> new_ids = {NUM_VECTORS, NUM_VECTORS + 1};
    std::vector<float> new_data;
    generate_test_data(new_data, new_ids.size(), DIMENSION);
    size_t count = 0;
    CATCH_REQUIRE(svs_index_dynamic_add_points(
        index, new_data.data(), new_ids.data(), new_ids.size(), &count, fx.error
    ));
    CATCH_REQUIRE(count == new_ids.size());
    CATCH_REQUIRE(
        svs_index_dynamic_delete_points(index, new_ids.data(), 1, &count, fx.error)
    );
    CATCH_REQUIRE(count == 1);

    bool has_id = true;
    CATCH_REQUIRE(svs_index_dynamic_has_id(index, new_ids[0], &has_id, fx.error));
    CATCH_REQUIRE_FALSE(has_id);

    svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
    CATCH_REQUIRE(svs_index_search_topk(
        index, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, fx.error
    ));
    svs_search_results_free(&results);

    float distance = 0.0f;
    CATCH_REQUIRE(svs_index_get_distance(index, 0, fx.queries.data(), &distance, fx.error));
    std::vector<float> reconstructed(DIMENSION);
    CATCH_REQUIRE(svs_index_reconstruct(
        index, fx.ids.data(), 1, reconstructed.data(), DIMENSION, fx.error
    ));

    CATCH_REQUIRE(svs_index_dynamic_consolidate(index, fx.error));
    CATCH_REQUIRE(svs_index_dynamic_compact(index, 0, fx.error));

    CATCH_REQUIRE(svs_index_set_num_threads(index, 1, fx.error));
    size_t num_threads = 0;
    CATCH_REQUIRE(svs_index_get_num_threads(index, &num_threads, fx.error));
    CATCH_REQUIRE(num_threads == 1);

    size_t memory = 0;
    CATCH_REQUIRE(svs_index_get_memory_usage(index, &memory, fx.error));
    CATCH_REQUIRE(memory > 0);

    TempDir dir;
    CATCH_REQUIRE(svs_index_save(index, dir.string().c_str(), fx.error));

    svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
    params.blocksize_bytes = BLOCK_SIZE;
    params.sync_kind = sync_kind;
    svs_index_h loaded =
        svs_index_load_dynamic_ex(fx.builder, dir.string().c_str(), &params, fx.error);
    CATCH_REQUIRE(loaded != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));

    size_t size = 0;
    CATCH_REQUIRE(svs_index_get_size(loaded, &size, fx.error));
    CATCH_REQUIRE(size == NUM_VECTORS + 1);
    CATCH_REQUIRE(svs_index_dynamic_get_sync_kind(loaded, &actual_kind, fx.error));
    CATCH_REQUIRE(actual_kind == sync_kind);

    // The converted index takes its sync kind from params, not from the source.
    params.sync_kind = SVS_SYNC_KIND_GLOBAL;
    svs_index_h converted =
        svs_index_convert_dynamic_ex(fx.builder, index, &params, fx.error);
    CATCH_REQUIRE(converted != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));
    CATCH_REQUIRE(svs_index_dynamic_get_sync_kind(converted, &actual_kind, fx.error));
    CATCH_REQUIRE(actual_kind == SVS_SYNC_KIND_GLOBAL);
    CATCH_REQUIRE(svs_index_get_size(converted, &size, fx.error));
    CATCH_REQUIRE(size == NUM_VECTORS + 1);

    svs_index_free(converted);
    svs_index_free(loaded);
    svs_index_free(index);
}

CATCH_TEST_CASE(
    "C API Dynamic Index Sync Kind Conversion", "[c_api][index][dynamic][sync]"
) {
    // FINE_GRAIN uses a different index implementation; convert between all pairs.
    const auto src_kind = GENERATE(SVS_SYNC_KIND_NONE, SVS_SYNC_KIND_FINE_GRAIN);
    const auto dst_kind = GENERATE(SVS_SYNC_KIND_NONE, SVS_SYNC_KIND_FINE_GRAIN);
    CATCH_CAPTURE(src_kind, dst_kind);

    SyncFixture fx;
    svs_index_h source = fx.build(src_kind);
    CATCH_REQUIRE(source != nullptr);

    // Leave deletions unconsolidated so the copy must skip them.
    constexpr size_t NUM_DELETED = 10;
    size_t count = 0;
    CATCH_REQUIRE(svs_index_dynamic_delete_points(
        source, fx.ids.data(), NUM_DELETED, &count, fx.error
    ));
    CATCH_REQUIRE(count == NUM_DELETED);

    svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
    params.blocksize_bytes = BLOCK_SIZE;
    params.sync_kind = dst_kind;
    svs_index_h converted =
        svs_index_convert_dynamic_ex(fx.builder, source, &params, fx.error);
    CATCH_REQUIRE(converted != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));

    size_t size = 0;
    CATCH_REQUIRE(svs_index_get_size(converted, &size, fx.error));
    CATCH_REQUIRE(size == NUM_VECTORS - NUM_DELETED);
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        bool has_id = false;
        CATCH_REQUIRE(svs_index_dynamic_has_id(converted, fx.ids[i], &has_id, fx.error));
        CATCH_REQUIRE(has_id == (i >= NUM_DELETED));
    }

    // The converted index stays fully usable.
    svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
    CATCH_REQUIRE(svs_index_search_topk(
        converted, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, fx.error
    ));
    for (size_t i = 0; i < results.total_results; ++i) {
        CATCH_REQUIRE(results.indices[i] >= NUM_DELETED);
        CATCH_REQUIRE(results.indices[i] < NUM_VECTORS);
    }
    svs_search_results_free(&results);

    std::vector<size_t> new_ids = {fx.ids[0], NUM_VECTORS};
    std::vector<float> new_data;
    generate_test_data(new_data, new_ids.size(), DIMENSION);
    CATCH_REQUIRE(svs_index_dynamic_add_points(
        converted, new_data.data(), new_ids.data(), new_ids.size(), &count, fx.error
    ));
    CATCH_REQUIRE(count == new_ids.size());
    CATCH_REQUIRE(svs_index_dynamic_consolidate(converted, fx.error));
    CATCH_REQUIRE(svs_index_dynamic_compact(converted, 0, fx.error));
    CATCH_REQUIRE(svs_index_get_size(converted, &size, fx.error));
    CATCH_REQUIRE(size == NUM_VECTORS - NUM_DELETED + new_ids.size());

    svs_index_free(converted);
    svs_index_free(source);
}

CATCH_TEST_CASE("C API Dynamic Index Fine Grain Storage", "[c_api][index][dynamic][sync]") {
    const int storage_id = GENERATE(0, 1, 2, 3, 4);
    CATCH_CAPTURE(storage_id);

    SyncFixture fx;
    svs_storage_h storage = nullptr;
    switch (storage_id) {
        case 0:
            storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, fx.error);
            break;
        case 1:
            storage = svs_storage_create_sq(SVS_DATA_TYPE_INT8, fx.error);
            break;
        case 2:
            storage =
                svs_storage_create_lvq(SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, fx.error);
            break;
        case 3:
            storage =
                svs_storage_create_lvq(SVS_DATA_TYPE_INT8, SVS_DATA_TYPE_VOID, fx.error);
            break;
        default:
            storage = svs_storage_create_leanvec(
                DIMENSION / 2, SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, fx.error
            );
            break;
    }
    CATCH_REQUIRE(check_storage_support(storage, fx.error));
    if (!storage_usable(storage)) {
        return;
    }

    svs_index_builder_h builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, fx.algorithm, fx.error
    );
    CATCH_REQUIRE(builder != nullptr);
    CATCH_REQUIRE(svs_index_builder_set_storage(builder, storage, fx.error));
    CATCH_REQUIRE(
        svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_NATIVE, 2, fx.error)
    );

    svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
    params.blocksize_bytes = BLOCK_SIZE;
    params.sync_kind = SVS_SYNC_KIND_FINE_GRAIN;
    svs_index_h index = svs_index_build_dynamic_ex(
        builder, fx.data.data(), fx.ids.data(), NUM_VECTORS, &params, fx.error
    );
    CATCH_REQUIRE(index != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));

    run_concurrent_workload(fx, index);

    // Save and load keep the concurrent index usable.
    TempDir dir;
    CATCH_REQUIRE(svs_index_save(index, dir.string().c_str(), fx.error));
    svs_index_h loaded =
        svs_index_load_dynamic_ex(builder, dir.string().c_str(), &params, fx.error);
    CATCH_REQUIRE(loaded != nullptr);
    CATCH_REQUIRE(svs_error_ok(fx.error));
    size_t expected_size = 0;
    size_t size = 0;
    CATCH_REQUIRE(svs_index_get_size(index, &expected_size, fx.error));
    CATCH_REQUIRE(svs_index_get_size(loaded, &size, fx.error));
    CATCH_REQUIRE(size == expected_size);

    svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
    CATCH_REQUIRE(svs_index_search_topk(
        loaded, fx.queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, fx.error
    ));
    svs_search_results_free(&results);

    // Compressed (concurrent) -> simple (regular), then simple (concurrent) -> compressed.
    params.sync_kind = SVS_SYNC_KIND_NONE;
    svs_index_h decompressed =
        svs_index_convert_dynamic_ex(fx.builder, loaded, &params, fx.error);
    CATCH_REQUIRE(decompressed != nullptr);
    CATCH_REQUIRE(svs_index_get_size(decompressed, &size, fx.error));
    CATCH_REQUIRE(size == expected_size);

    svs_index_h simple = fx.build(SVS_SYNC_KIND_FINE_GRAIN);
    CATCH_REQUIRE(simple != nullptr);
    params.sync_kind = SVS_SYNC_KIND_FINE_GRAIN;
    svs_index_h compressed =
        svs_index_convert_dynamic_ex(builder, simple, &params, fx.error);
    CATCH_REQUIRE(compressed != nullptr);
    CATCH_REQUIRE(svs_index_get_size(compressed, &size, fx.error));
    CATCH_REQUIRE(size == NUM_VECTORS);

    svs_index_free(compressed);
    svs_index_free(simple);
    svs_index_free(decompressed);
    svs_index_free(loaded);
    svs_index_free(index);
    svs_index_builder_free(builder);
    svs_storage_free(storage);
}

CATCH_TEST_CASE("C API Dynamic Index Sync Kind Memory", "[c_api][index][dynamic][sync]") {
    SyncFixture fx;

    auto estimate = [&](svs_sync_kind_t kind) {
        svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        params.blocksize_bytes = BLOCK_SIZE;
        params.sync_kind = kind;
        svs_memory_breakdown_t breakdown = SVS_INIT_MEMORY_BREAKDOWN();
        CATCH_REQUIRE(svs_index_builder_estimate_memory_dynamic_ex(
            fx.builder, NUM_VECTORS, &params, &breakdown, fx.error
        ));
        return breakdown;
    };

    const auto regular = estimate(SVS_SYNC_KIND_NONE);
    const auto global = estimate(SVS_SYNC_KIND_GLOBAL);
    const auto fine_grain = estimate(SVS_SYNC_KIND_FINE_GRAIN);
    CATCH_REQUIRE(global.graph_bytes == regular.graph_bytes);
    // The concurrent index also keeps reverse edges, counted as graph memory.
    CATCH_REQUIRE(fine_grain.graph_bytes > regular.graph_bytes);
    CATCH_REQUIRE(fine_grain.data_bytes == regular.data_bytes);
    CATCH_REQUIRE(fine_grain.metadata_bytes == regular.metadata_bytes);

    svs_index_h index = fx.build(SVS_SYNC_KIND_FINE_GRAIN);
    CATCH_REQUIRE(index != nullptr);
    svs_memory_breakdown_t breakdown = SVS_INIT_MEMORY_BREAKDOWN();
    CATCH_REQUIRE(svs_index_get_memory_breakdown(index, &breakdown, fx.error));
    size_t usage = 0;
    CATCH_REQUIRE(svs_index_get_memory_usage(index, &usage, fx.error));
    CATCH_REQUIRE(
        usage == breakdown.graph_bytes + breakdown.data_bytes + breakdown.metadata_bytes
    );
    CATCH_REQUIRE(breakdown.graph_bytes > 0);
    svs_index_free(index);
}

CATCH_TEST_CASE("C API Dynamic Index Sync Concurrent", "[c_api][index][dynamic][sync]") {
    const auto sync_kind = GENERATE(SVS_SYNC_KIND_GLOBAL, SVS_SYNC_KIND_FINE_GRAIN);
    CATCH_CAPTURE(sync_kind);

    SyncFixture fx;
    CATCH_REQUIRE(fx.builder != nullptr);

    CATCH_SECTION("Built index") {
        svs_index_h index = fx.build(sync_kind);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(fx.error));
        run_concurrent_workload(fx, index);
        svs_index_free(index);
    }

    CATCH_SECTION("Loaded index") {
        svs_index_h source = fx.build(SVS_SYNC_KIND_NONE);
        CATCH_REQUIRE(source != nullptr);
        TempDir dir;
        CATCH_REQUIRE(svs_index_save(source, dir.string().c_str(), fx.error));
        svs_index_free(source);

        svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        params.blocksize_elements = 64;
        params.sync_kind = sync_kind;
        svs_index_h index =
            svs_index_load_dynamic_ex(fx.builder, dir.string().c_str(), &params, fx.error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(fx.error));
        run_concurrent_workload(fx, index);
        svs_index_free(index);
    }

    CATCH_SECTION("Converted index") {
        svs_index_h source = fx.build(SVS_SYNC_KIND_NONE);
        CATCH_REQUIRE(source != nullptr);

        svs_dynamic_index_params_t params = SVS_INIT_DYNAMIC_INDEX_PARAMS();
        params.blocksize_bytes = BLOCK_SIZE;
        params.sync_kind = sync_kind;
        svs_index_h index =
            svs_index_convert_dynamic_ex(fx.builder, source, &params, fx.error);
        svs_index_free(source);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(fx.error));
        run_concurrent_workload(fx, index);
        svs_index_free(index);
    }
}
