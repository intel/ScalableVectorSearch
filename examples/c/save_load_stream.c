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

#include "svs/c/svs_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_VECTORS 10000
#define NUM_QUERIES 1
#define DIMENSION 128
#define K 10

// `size` is the high-water mark reached while writing; `pos` is the read/write cursor and
// gets rewound to 0 between the save pass and the load pass.
typedef struct {
    unsigned char* data;
    size_t capacity;
    size_t size;
    size_t pos;
} memory_stream_t;

memory_stream_t* memory_stream_create(size_t initial_capacity) {
    memory_stream_t* stream = (memory_stream_t*)malloc(sizeof(memory_stream_t));
    if (!stream) {
        return NULL;
    }
    stream->data = (unsigned char*)malloc(initial_capacity);
    if (!stream->data) {
        free(stream);
        return NULL;
    }
    stream->capacity = initial_capacity;
    stream->size = 0;
    stream->pos = 0;
    return stream;
}

void memory_stream_free(memory_stream_t* stream) {
    if (stream) {
        free(stream->data);
        free(stream);
    }
}

// A short read is legal here and the library calls again for the remainder; this callback
// always fills the whole request, but a caller need not.
static size_t memory_stream_read(void* self, void* buf, size_t n, svs_error_h out_err) {
    (void)out_err;
    memory_stream_t* stream = (memory_stream_t*)self;
    size_t available = stream->size - stream->pos;
    size_t to_read = (n < available) ? n : available;
    if (to_read > 0) {
        memcpy(buf, stream->data + stream->pos, to_read);
        stream->pos += to_read;
    }
    return to_read;
}

// A partial write must be reported as failure; otherwise the saved stream is silently
// truncated and the failure surfaces only much later, as a load error.
static bool
memory_stream_write(void* self, const void* buf, size_t n, svs_error_h out_err) {
    (void)out_err;
    memory_stream_t* stream = (memory_stream_t*)self;
    while (stream->pos + n > stream->capacity) {
        size_t new_capacity = stream->capacity * 2;
        unsigned char* new_data = (unsigned char*)realloc(stream->data, new_capacity);
        if (!new_data) {
            return false;
        }
        stream->data = new_data;
        stream->capacity = new_capacity;
    }
    memcpy(stream->data + stream->pos, buf, n);
    stream->pos += n;
    if (stream->pos > stream->size) {
        stream->size = stream->pos;
    }
    return true;
}

void generate_random_data(float* data, size_t count, size_t dim) {
    for (size_t i = 0; i < count * dim; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    int ret = 0;
    srand(time(NULL));
    svs_error_h error = svs_error_create();

    float* data = NULL;
    float* queries = NULL;
    svs_algorithm_h algorithm = NULL;
    svs_storage_h storage = NULL;
    svs_index_builder_h builder = NULL;
    svs_index_h index = NULL;
    svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
    memory_stream_t* stream = NULL;
    svs_search_results_t loaded_results = SVS_INIT_SEARCH_RESULTS();

    // Allocate random data
    data = (float*)malloc(NUM_VECTORS * DIMENSION * sizeof(float));
    queries = (float*)malloc(NUM_QUERIES * DIMENSION * sizeof(float));

    if (!data || !queries) {
        fprintf(stderr, "Failed to allocate memory\n");
        ret = 1;
        goto cleanup;
    }

    generate_random_data(data, NUM_VECTORS, DIMENSION);
    generate_random_data(queries, NUM_QUERIES, DIMENSION);

    // Create Vamana algorithm
    algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
    if (!algorithm) {
        fprintf(stderr, "Failed to create algorithm: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }

    // Create storage (simple float32)
    storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error);
    if (!storage) {
        fprintf(stderr, "Failed to create storage: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }

    // Create index builder
    builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
    );
    if (!builder) {
        fprintf(
            stderr, "Failed to create index builder: %s\n", svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }

    if (!svs_index_builder_set_storage(builder, storage, error)) {
        fprintf(stderr, "Failed to set storage: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }

    // Build index
    printf("Building index with %d vectors of dimension %d...\n", NUM_VECTORS, DIMENSION);
    index = svs_index_build(builder, data, NUM_VECTORS, error);
    if (!index) {
        fprintf(stderr, "Failed to build index: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }
    printf("Index built successfully!\n");

    // Search
    printf("Searching %d queries for top-%d neighbors...\n", NUM_QUERIES, K);
    if (!svs_index_search_topk(
            index,
            queries,
            NUM_QUERIES,
            K,
            &results,
            NULL /* search_params */,
            NULL /* id_filter */,
            error
        )) {
        fprintf(stderr, "Failed to search index: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }
    printf("Search completed successfully!\n");

    // Create in-memory stream for saving
    stream = memory_stream_create(1024 * 1024);
    if (!stream) {
        fprintf(stderr, "Failed to create stream\n");
        ret = 1;
        goto cleanup;
    }

    // Construct stream interface and save through it
    static svs_stream_ops_t stream_ops =
        SVS_INIT_STREAM_OPS(memory_stream_read, memory_stream_write);
    svs_stream_t stream_iface = SVS_MAKE_INTERFACE(stream, stream_ops);

    printf("Saving index to in-memory stream...\n");
    if (!svs_index_save_stream(index, &stream_iface, error)) {
        fprintf(
            stderr, "Failed to save index to stream: %s\n", svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }
    printf("Index saved successfully! Stream size: %zu bytes\n", stream->size);

    svs_index_free(index);
    index = NULL;

    // Reset stream position for reading
    stream->pos = 0;

    // Load the index from the stream
    printf("Loading index from in-memory stream...\n");
    index = svs_index_load_stream(builder, &stream_iface, error);
    if (!index) {
        fprintf(
            stderr, "Failed to load index from stream: %s\n", svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }
    printf("Index loaded successfully!\n");

    // Search the loaded index
    printf(
        "Searching loaded index for %d queries for top-%d neighbors...\n", NUM_QUERIES, K
    );
    if (!svs_index_search_topk(
            index,
            queries,
            NUM_QUERIES,
            K,
            &loaded_results,
            NULL /* search_params */,
            NULL /* id_filter */,
            error
        )) {
        fprintf(
            stderr, "Failed to search loaded index: %s\n", svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }
    printf("Search on loaded index completed successfully!\n");

    // Compare results
    if (results.num_queries != loaded_results.num_queries) {
        fprintf(
            stderr, "Mismatch in number of queries between original and loaded results\n"
        );
        ret = 1;
        goto cleanup;
    }

    size_t offset = 0;
    for (size_t q = 0; q < results.num_queries; q++) {
        size_t count = results.offsets[q + 1] - results.offsets[q];
        size_t loaded_count = loaded_results.offsets[q + 1] - loaded_results.offsets[q];
        if (count != loaded_count) {
            fprintf(stderr, "Mismatch in number of results for query %zu\n", q);
            ret = 1;
            goto cleanup;
        }
        printf("Query %zu results:\n", q);
        for (size_t i = 0; i < count; i++) {
            if (results.indices[offset + i] != loaded_results.indices[offset + i]) {
                fprintf(
                    stderr, "Mismatch in neighbor indices for query %zu, result %zu\n", q, i
                );
                ret = 1;
                goto cleanup;
            }
            printf(
                "  [%zu] id=%zu, distance=%.4f, diff=%.4f\n",
                i,
                results.indices[offset + i],
                results.distances[offset + i],
                results.distances[offset + i] - loaded_results.distances[offset + i]
            );
        }
        offset += count;
    }

    printf("Done!\n");

cleanup:
    svs_search_results_free(&results);
    svs_search_results_free(&loaded_results);
    svs_index_free(index);
    svs_index_builder_free(builder);
    svs_storage_free(storage);
    svs_algorithm_free(algorithm);
    free(data);
    free(queries);
    memory_stream_free(stream);
    svs_error_free(error);

    return ret;
}
