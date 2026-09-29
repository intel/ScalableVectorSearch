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

// Sized so the saved payload (data + graph) provably exceeds two StreamBuf write buffers:
// data alone is num_vectors * dimension * sizeof(float) = 200 * 700 * 4 = 560'000 bytes,
// well over 2 * STREAM_BUFFER_SIZE = 131'072, before the graph even adds its share.
constexpr size_t MULTIBUFFER_NUM_VECTORS = 200;
constexpr size_t MULTIBUFFER_DIMENSION = 700;

// Owns the algorithm/builder/index triple built over the oversized data set above, so the
// two sections that need a multi-buffer payload (round-trip and partial-flush-failure)
// share one construction path instead of duplicating build setup.
struct MultiBufferIndex {
    svs_algorithm_h algorithm = nullptr;
    svs_index_builder_h builder = nullptr;
    svs_index_h index = nullptr;
};

MultiBufferIndex build_multibuffer_index(std::vector<float>& data, svs_error_h error) {
    MultiBufferIndex result;
    result.algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
    result.builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, MULTIBUFFER_DIMENSION, result.algorithm, error
    );
    svs_index_builder_set_threadpool(
        result.builder, SVS_THREADPOOL_KIND_SINGLE_THREAD, 1, error
    );
    generate_test_data(data, MULTIBUFFER_NUM_VECTORS, MULTIBUFFER_DIMENSION);
    result.index =
        svs_index_build(result.builder, data.data(), MULTIBUFFER_NUM_VECTORS, error);
    return result;
}

// Like fail_partial_write, but also counts the full-size writes that succeeded before the
// partial one it rejects, so a test can assert the failure was the *last* of several writes
// rather than the only one.
struct CountingFailSink {
    MemoryStream stream;
    size_t full_write_count = 0;
};

bool fail_partial_write_counted(
    void* self, const void* buf, size_t n, svs_error_h out_err
) {
    auto* sink = static_cast<CountingFailSink*>(self);
    if (n < STREAM_BUFFER_SIZE) {
        svs_error_set(out_err, SVS_ERROR_RUNTIME, "refusing partial write");
        return false;
    }
    sink->full_write_count++;
    return memory_stream_write(&sink->stream, buf, n, out_err);
}

// Fails exactly once, on the fail_at_invocation'th call, then records whether it is ever
// invoked again. Regression coverage for flush_write_buffer() resetting the put area before
// calling out, not after: previously a failed buffer's bytes were still sitting in the put
// area when ~StreamBuf tried to flush again, redelivering them to the callback a second
// time.
struct RecordingFailSink {
    size_t fail_at_invocation = 0;
    size_t invocation_count = 0;
    size_t total_bytes = 0;
    size_t invocations_after_failure = 0;
    bool has_failed = false;
};

bool fail_after_n_write(void* self, const void* /*buf*/, size_t n, svs_error_h out_err) {
    auto* sink = static_cast<RecordingFailSink*>(self);
    if (sink->has_failed) {
        sink->invocations_after_failure++;
        return false;
    }
    sink->invocation_count++;
    if (sink->invocation_count == sink->fail_at_invocation) {
        sink->has_failed = true;
        svs_error_set(out_err, SVS_ERROR_RUNTIME, "simulated failure mid-stream");
        return false;
    }
    sink->total_bytes += n;
    return true;
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

    CATCH_SECTION("Round-trip through an in-memory stream spanning multiple write buffers"
    ) {
        std::vector<float> big_data;
        MultiBufferIndex mb = build_multibuffer_index(big_data, error);
        CATCH_REQUIRE(mb.algorithm != nullptr);
        CATCH_REQUIRE(mb.builder != nullptr);
        CATCH_REQUIRE(mb.index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        std::vector<float> big_queries;
        generate_test_data(big_queries, 3, MULTIBUFFER_DIMENSION);

        svs_search_results_t before = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            mb.index, big_queries.data(), 3, K, &before, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));

        MemoryStream stream;
        svs_stream_interface_ops write_ops =
            SVS_INIT_STREAM_OPS(nullptr, memory_stream_write);
        svs_stream_interface out_stream = SVS_MAKE_INTERFACE(&stream, write_ops);
        CATCH_REQUIRE(svs_index_save_stream(mb.index, &out_stream, error));
        CATCH_REQUIRE(svs_error_ok(error));
        // Proves the write path actually flushed several full buffers and a trailing
        // partial one, and the read path below refills the get area several times, rather
        // than relying on the vector counts above staying big enough by construction.
        CATCH_REQUIRE(stream.bytes.size() > 2 * STREAM_BUFFER_SIZE);

        svs_stream_interface_ops read_ops =
            SVS_INIT_STREAM_OPS(memory_stream_read, nullptr);
        svs_stream_interface in_stream = SVS_MAKE_INTERFACE(&stream, read_ops);
        svs_index_h loaded = svs_index_load_stream(mb.builder, &in_stream, error);
        CATCH_REQUIRE(loaded != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_results_t after = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            loaded, big_queries.data(), 3, K, &after, nullptr, nullptr, error
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
        svs_index_free(mb.index);
        svs_index_builder_free(mb.builder);
        svs_algorithm_free(mb.algorithm);
    }

    CATCH_SECTION("Save fails when only the final partial flush fails") {
        std::vector<float> big_data;
        MultiBufferIndex mb = build_multibuffer_index(big_data, error);
        CATCH_REQUIRE(mb.index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        MemoryStream probe;
        svs_stream_interface_ops probe_ops =
            SVS_INIT_STREAM_OPS(nullptr, memory_stream_write);
        svs_stream_interface probe_stream = SVS_MAKE_INTERFACE(&probe, probe_ops);
        CATCH_REQUIRE(svs_index_save_stream(mb.index, &probe_stream, error));
        // The payload must span at least two full buffers plus a partial remainder, or this
        // test cannot distinguish "the one and only flush failed" from "the final partial
        // flush failed after several successful full flushes" - the regression it targets.
        CATCH_REQUIRE(probe.bytes.size() > 2 * STREAM_BUFFER_SIZE);
        // The failing sink below only ever rejects a write shorter than the buffer; if the
        // payload happened to land exactly on a buffer boundary there would be no partial
        // write left for it to catch.
        CATCH_REQUIRE(probe.bytes.size() % STREAM_BUFFER_SIZE != 0);

        CountingFailSink sink;
        svs_stream_interface_ops fail_ops =
            SVS_INIT_STREAM_OPS(nullptr, fail_partial_write_counted);
        svs_stream_interface fail_stream = SVS_MAKE_INTERFACE(&sink, fail_ops);
        CATCH_REQUIRE_FALSE(svs_index_save_stream(mb.index, &fail_stream, error));
        CATCH_REQUIRE_FALSE(svs_error_ok(error));
        // At least two full-size writes must have succeeded before the failing partial one,
        // or this is once again just "the single flush failed".
        CATCH_REQUIRE(sink.full_write_count >= 2);

        svs_index_free(mb.index);
        svs_index_builder_free(mb.builder);
        svs_algorithm_free(mb.algorithm);
    }

    CATCH_SECTION("Write failure never redelivers the same bytes during destructor unwind"
    ) {
        std::vector<float> big_data;
        MultiBufferIndex mb = build_multibuffer_index(big_data, error);
        CATCH_REQUIRE(mb.index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Fails on the 3rd invocation, well before the last one, so several full buffers
        // must have already been accepted and there is guaranteed to be more payload left
        // that a double-delivery bug would have handed to the callback a second time.
        RecordingFailSink sink;
        sink.fail_at_invocation = 3;
        svs_stream_interface_ops fail_ops =
            SVS_INIT_STREAM_OPS(nullptr, fail_after_n_write);
        svs_stream_interface fail_stream = SVS_MAKE_INTERFACE(&sink, fail_ops);
        CATCH_REQUIRE_FALSE(svs_index_save_stream(mb.index, &fail_stream, error));
        CATCH_REQUIRE_FALSE(svs_error_ok(error));
        CATCH_REQUIRE(sink.invocation_count == 3);
        CATCH_REQUIRE(sink.invocations_after_failure == 0);

        svs_index_free(mb.index);
        svs_index_builder_free(mb.builder);
        svs_algorithm_free(mb.algorithm);
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

    CATCH_SECTION("Load fails on an empty stream instead of hanging or crashing") {
        MemoryStream stream;
        svs_stream_interface_ops read_ops =
            SVS_INIT_STREAM_OPS(memory_stream_read, nullptr);
        svs_stream_interface in_stream = SVS_MAKE_INTERFACE(&stream, read_ops);
        svs_index_h loaded = svs_index_load_stream(builder, &in_stream, error);
        CATCH_REQUIRE(loaded == nullptr);
        CATCH_REQUIRE_FALSE(svs_error_ok(error));
    }

    CATCH_SECTION("Load fails on a stream truncated partway through a valid payload") {
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        MemoryStream stream;
        svs_stream_interface_ops write_ops =
            SVS_INIT_STREAM_OPS(nullptr, memory_stream_write);
        svs_stream_interface out_stream = SVS_MAKE_INTERFACE(&stream, write_ops);
        CATCH_REQUIRE(svs_index_save_stream(index, &out_stream, error));
        CATCH_REQUIRE(svs_error_ok(error));

        // Cut the valid payload in half: the read callback hands back real bytes for a
        // while, then reports EOF (0 bytes) before the format is fully consumed.
        CATCH_REQUIRE(stream.bytes.size() > 1);
        stream.bytes.resize(stream.bytes.size() / 2);
        stream.pos = 0;

        svs_stream_interface_ops read_ops =
            SVS_INIT_STREAM_OPS(memory_stream_read, nullptr);
        svs_stream_interface in_stream = SVS_MAKE_INTERFACE(&stream, read_ops);
        svs_index_h loaded = svs_index_load_stream(builder, &in_stream, error);
        CATCH_REQUIRE(loaded == nullptr);
        CATCH_REQUIRE_FALSE(svs_error_ok(error));

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
        svs_index_h loaded =
            svs_index_load_stream_dynamic(builder, &in_stream, BLOCK_SIZE, error);
        CATCH_REQUIRE(loaded != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // A stream load that corrupted data or graph structure must not be able to pass
        // this: compare against the pre-save index element-by-element, as the static
        // round-trip does, instead of only checking the query count came back.
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

        // Query with the exact vector of a newly added point and require its id shows up
        // among the neighbors, so a load that silently dropped or corrupted the loaded
        // graph/data cannot pass just because a search still returns *some* results.
        std::vector<float> probe_query(
            new_data.begin(), new_data.begin() + static_cast<ptrdiff_t>(DIMENSION)
        );
        svs_search_results_t probe_results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            loaded, probe_query.data(), 1, K, &probe_results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        bool found_new_id = false;
        for (size_t i = 0; i < K; ++i) {
            if (probe_results.indices[i] == new_ids[0]) {
                found_new_id = true;
                break;
            }
        }
        CATCH_REQUIRE(found_new_id);

        svs_search_results_free(&results);
        svs_search_results_free(&probe_results);
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
