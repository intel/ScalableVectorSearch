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

// Test utilities
#include "c_api_test_utils.h"

// Standard library
#include <cstring>
#include <limits>
#include <vector>

namespace {

// The streambuf adapter's buffer size (bindings/c/src/stream.hpp); not part of the public
// API, so tests that depend on it hardcode the value.
constexpr size_t STREAM_BUFFER_SIZE = 64 * 1024;

// In-memory sink/source backing the stream interface tests. write() appends to `bytes`;
// read() copies out of `bytes` starting at `pos`, capped at `max_read` per call so tests
// can force short reads.
struct MemoryStream {
    std::vector<char> bytes;
    size_t pos = 0;
    size_t max_read = std::numeric_limits<size_t>::max();
};

size_t memory_stream_read(void* self, void* buf, size_t n, svs_error_h /*out_err*/) {
    auto* stream = static_cast<MemoryStream*>(self);
    size_t remaining = stream->bytes.size() - stream->pos;
    size_t to_copy = std::min({n, remaining, stream->max_read});
    std::memcpy(buf, stream->bytes.data() + stream->pos, to_copy);
    stream->pos += to_copy;
    return to_copy;
}

bool memory_stream_write(void* self, const void* buf, size_t n, svs_error_h /*out_err*/) {
    auto* stream = static_cast<MemoryStream*>(self);
    const auto* src = static_cast<const char*>(buf);
    stream->bytes.insert(stream->bytes.end(), src, src + n);
    return true;
}

// Fails a write smaller than the adapter's buffer, which every write during a save is
// except the final flush of a partial one, so this deterministically targets only that
// last write regardless of how many full buffers preceded it.
bool fail_partial_write(void* self, const void* buf, size_t n, svs_error_h out_err) {
    if (n < STREAM_BUFFER_SIZE) {
        svs_error_set(out_err, SVS_ERROR_RUNTIME, "refusing partial write");
        return false;
    }
    return memory_stream_write(self, buf, n, out_err);
}

bool always_fail_write(
    void* /*self*/, const void* /*buf*/, size_t /*n*/, svs_error_h /*out_err*/
) {
    return false;
}

bool oom_write(void* /*self*/, const void* /*buf*/, size_t /*n*/, svs_error_h out_err) {
    svs_error_set(out_err, SVS_ERROR_OUT_OF_MEMORY, "simulated allocator exhaustion");
    return false;
}

size_t oom_read(void* /*self*/, void* /*buf*/, size_t /*n*/, svs_error_h out_err) {
    svs_error_set(out_err, SVS_ERROR_OUT_OF_MEMORY, "simulated allocator exhaustion");
    return 0;
}

} // namespace

CATCH_TEST_CASE("C API Stream Save and Load", "[c_api][index][stream]") {
    const size_t NUM_VECTORS = 100;
    const size_t DIMENSION = 32;
    const size_t K = 5;

    std::vector<float> data;
    std::vector<float> queries;
    generate_test_data(data, NUM_VECTORS, DIMENSION);
    generate_test_data(queries, 3, DIMENSION);

    svs_error_h error = svs_error_create();

    svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
    CATCH_REQUIRE(algorithm != nullptr);

    svs_index_builder_h builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
    );
    CATCH_REQUIRE(builder != nullptr);

    // Single-threaded so a greedy search visits the same path on both indexes.
    bool success = svs_index_builder_set_threadpool(
        builder, SVS_THREADPOOL_KIND_SINGLE_THREAD, 1, error
    );
    CATCH_REQUIRE(success);
    CATCH_REQUIRE(svs_error_ok(error));

    CATCH_SECTION("Static round-trip through an in-memory stream") {
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        svs_search_results_t before = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index, queries.data(), 3, K, &before, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));

        MemoryStream stream;
        svs_stream_interface_ops write_ops =
            SVS_INIT_STREAM_OPS(nullptr, memory_stream_write);
        svs_stream_interface out_stream = SVS_MAKE_INTERFACE(&stream, write_ops);
        CATCH_REQUIRE(svs_index_save_stream(index, &out_stream, error));
        CATCH_REQUIRE(svs_error_ok(error));

        svs_stream_interface_ops read_ops =
            SVS_INIT_STREAM_OPS(memory_stream_read, nullptr);
        svs_stream_interface in_stream = SVS_MAKE_INTERFACE(&stream, read_ops);
        svs_index_h loaded = svs_index_load_stream(builder, &in_stream, error);
        CATCH_REQUIRE(loaded != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_results_t after = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            loaded, queries.data(), 3, K, &after, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(after.num_queries == before.num_queries);
        for (size_t i = 0; i < before.num_queries * K; ++i) {
            CATCH_REQUIRE(after.indices[i] == before.indices[i]);
            CATCH_REQUIRE(after.distances[i] == before.distances[i]);
        }

        svs_search_results_free(&before);
        svs_search_results_free(&after);
        svs_index_free(loaded);
        svs_index_free(index);
    }

    CATCH_SECTION("Save fails when only the final partial flush fails") {
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        MemoryStream probe;
        svs_stream_interface_ops probe_ops =
            SVS_INIT_STREAM_OPS(nullptr, memory_stream_write);
        svs_stream_interface probe_stream = SVS_MAKE_INTERFACE(&probe, probe_ops);
        CATCH_REQUIRE(svs_index_save_stream(index, &probe_stream, error));
        // The failing sink below only ever rejects a write shorter than the buffer; if the
        // payload happened to land exactly on a buffer boundary there would be no partial
        // write left for it to catch.
        CATCH_REQUIRE(probe.bytes.size() % STREAM_BUFFER_SIZE != 0);

        MemoryStream sink;
        svs_stream_interface_ops fail_ops =
            SVS_INIT_STREAM_OPS(nullptr, fail_partial_write);
        svs_stream_interface fail_stream = SVS_MAKE_INTERFACE(&sink, fail_ops);
        CATCH_REQUIRE_FALSE(svs_index_save_stream(index, &fail_stream, error));
        CATCH_REQUIRE_FALSE(svs_error_ok(error));

        svs_index_free(index);
    }

    CATCH_SECTION("Write callback failure aborts save") {
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        svs_stream_interface_ops fail_ops = SVS_INIT_STREAM_OPS(nullptr, always_fail_write);
        svs_stream_interface fail_stream = SVS_MAKE_INTERFACE(nullptr, fail_ops);
        CATCH_REQUIRE_FALSE(svs_index_save_stream(index, &fail_stream, error));
        CATCH_REQUIRE_FALSE(svs_error_ok(error));

        svs_index_free(index);
    }

    CATCH_SECTION("Write callback reports a specific error code") {
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        svs_stream_interface_ops fail_ops = SVS_INIT_STREAM_OPS(nullptr, oom_write);
        svs_stream_interface fail_stream = SVS_MAKE_INTERFACE(nullptr, fail_ops);
        CATCH_REQUIRE_FALSE(svs_index_save_stream(index, &fail_stream, error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_OUT_OF_MEMORY);

        svs_index_free(index);
    }

    CATCH_SECTION("Read callback reports a specific error code") {
        svs_stream_interface_ops fail_ops = SVS_INIT_STREAM_OPS(oom_read, nullptr);
        svs_stream_interface fail_stream = SVS_MAKE_INTERFACE(nullptr, fail_ops);
        svs_index_h loaded = svs_index_load_stream(builder, &fail_stream, error);
        CATCH_REQUIRE(loaded == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_OUT_OF_MEMORY);
    }

    CATCH_SECTION("Short reads and EOF on load") {
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        MemoryStream stream;
        svs_stream_interface_ops write_ops =
            SVS_INIT_STREAM_OPS(nullptr, memory_stream_write);
        svs_stream_interface out_stream = SVS_MAKE_INTERFACE(&stream, write_ops);
        CATCH_REQUIRE(svs_index_save_stream(index, &out_stream, error));

        // Hand back at most 3 bytes per call, forcing many short reads before the final
        // 0-byte EOF.
        stream.max_read = 3;
        svs_stream_interface_ops read_ops =
            SVS_INIT_STREAM_OPS(memory_stream_read, nullptr);
        svs_stream_interface in_stream = SVS_MAKE_INTERFACE(&stream, read_ops);
        svs_index_h loaded = svs_index_load_stream(builder, &in_stream, error);
        CATCH_REQUIRE(loaded != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            loaded, queries.data(), 3, K, &results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results.num_queries == 3);

        svs_search_results_free(&results);
        svs_index_free(loaded);
        svs_index_free(index);
    }

    CATCH_SECTION("Dynamic round-trip, add_points, and search") {
        std::vector<size_t> ids(NUM_VECTORS);
        for (size_t i = 0; i < NUM_VECTORS; ++i) {
            ids[i] = i;
        }
        const size_t BLOCK_SIZE = 1024 * 1024;
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(index != nullptr);

        MemoryStream stream;
        svs_stream_interface_ops write_ops =
            SVS_INIT_STREAM_OPS(nullptr, memory_stream_write);
        svs_stream_interface out_stream = SVS_MAKE_INTERFACE(&stream, write_ops);
        CATCH_REQUIRE(svs_index_save_stream(index, &out_stream, error));
        CATCH_REQUIRE(svs_error_ok(error));

        svs_stream_interface_ops read_ops =
            SVS_INIT_STREAM_OPS(memory_stream_read, nullptr);
        svs_stream_interface in_stream = SVS_MAKE_INTERFACE(&stream, read_ops);
        svs_index_h loaded =
            svs_index_load_stream_dynamic(builder, &in_stream, BLOCK_SIZE, error);
        CATCH_REQUIRE(loaded != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        std::vector<float> new_data;
        std::vector<size_t> new_ids = {NUM_VECTORS, NUM_VECTORS + 1};
        generate_test_data(new_data, 2, DIMENSION);
        size_t added_count = 0;
        CATCH_REQUIRE(svs_index_dynamic_add_points(
            loaded, new_data.data(), new_ids.data(), 2, &added_count, error
        ));
        CATCH_REQUIRE(added_count == 2);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            loaded, queries.data(), 3, K, &results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results.num_queries == 3);

        svs_search_results_free(&results);
        svs_index_free(loaded);
        svs_index_free(index);
    }

    svs_index_builder_free(builder);
    svs_algorithm_free(algorithm);
    svs_error_free(error);
}

CATCH_TEST_CASE("C API Stream Interface Validation", "[c_api][index][stream][error]") {
    const size_t NUM_VECTORS = 20;
    const size_t DIMENSION = 8;

    std::vector<float> data;
    generate_test_data(data, NUM_VECTORS, DIMENSION);

    svs_error_h error = svs_error_create();

    svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
    CATCH_REQUIRE(algorithm != nullptr);

    svs_index_builder_h builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
    );
    CATCH_REQUIRE(builder != nullptr);

    bool success = svs_index_builder_set_threadpool(
        builder, SVS_THREADPOOL_KIND_SINGLE_THREAD, 1, error
    );
    CATCH_REQUIRE(success);

    svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
    CATCH_REQUIRE(index != nullptr);

    CATCH_SECTION("Null interface pointer is rejected") {
        CATCH_REQUIRE_FALSE(svs_index_save_stream(index, nullptr, error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        CATCH_REQUIRE(svs_index_load_stream(builder, nullptr, error) == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    CATCH_SECTION("Null ops pointer is rejected") {
        svs_stream_interface stream{nullptr, nullptr};
        CATCH_REQUIRE_FALSE(svs_index_save_stream(index, &stream, error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        CATCH_REQUIRE(svs_index_load_stream(builder, &stream, error) == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    CATCH_SECTION("struct_size smaller than expected is rejected") {
        svs_stream_interface_ops ops =
            SVS_INIT_STREAM_OPS(memory_stream_read, memory_stream_write);
        ops.struct_size = sizeof(uint32_t) + sizeof(size_t);
        svs_stream_interface stream = SVS_MAKE_INTERFACE(nullptr, ops);
        CATCH_REQUIRE_FALSE(svs_index_save_stream(index, &stream, error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        CATCH_REQUIRE(svs_index_load_stream(builder, &stream, error) == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    CATCH_SECTION("version above svs_get_version() is rejected") {
        svs_stream_interface_ops ops =
            SVS_INIT_STREAM_OPS(memory_stream_read, memory_stream_write);
        ops.version = svs_get_version() + 1;
        svs_stream_interface stream = SVS_MAKE_INTERFACE(nullptr, ops);
        CATCH_REQUIRE_FALSE(svs_index_save_stream(index, &stream, error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        CATCH_REQUIRE(svs_index_load_stream(builder, &stream, error) == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    CATCH_SECTION("NULL write on save is rejected") {
        svs_stream_interface_ops ops = SVS_INIT_STREAM_OPS(memory_stream_read, nullptr);
        svs_stream_interface stream = SVS_MAKE_INTERFACE(nullptr, ops);
        CATCH_REQUIRE_FALSE(svs_index_save_stream(index, &stream, error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    CATCH_SECTION("NULL read on load is rejected") {
        svs_stream_interface_ops ops = SVS_INIT_STREAM_OPS(nullptr, memory_stream_write);
        svs_stream_interface stream = SVS_MAKE_INTERFACE(nullptr, ops);
        CATCH_REQUIRE(svs_index_load_stream(builder, &stream, error) == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
    }

    svs_index_free(index);
    svs_index_builder_free(builder);
    svs_algorithm_free(algorithm);
    svs_error_free(error);
}
