#include "svs/c_api/svs_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INITIAL_VECTORS 10000
#define TAILING_VECTORS 1000
#define NUM_VECTORS (INITIAL_VECTORS + TAILING_VECTORS)
#define DELETE_VECTORS_BEGIN 5000
#define DELETE_VECTORS_END 8000
#define NUM_QUERIES 5
#define DIMENSION 128
#define K 10

void generate_random_data(float* data, size_t count, size_t dim) {
    for (size_t i = 0; i < count * dim; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

size_t sequential_tp_size(void* self) { return 1; }

void sequential_tp_parallel_for(
    void* self, void (*func)(void*, size_t), void* svs_param, size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        func(svs_param, i);
    }
}

static struct svs_threadpool_interface sequential_threadpool = {
    {
        &sequential_tp_size,
        &sequential_tp_parallel_for,
    },
    NULL,
};

int main() {
    int ret = 0;
    srand(time(NULL));
    svs_error_h error = svs_error_create();

    float* data = NULL;
    size_t* ids = NULL;
    float* queries = NULL;
    svs_algorithm_h algorithm = NULL;
    svs_storage_h storage = NULL;
    svs_index_builder_h builder = NULL;
    svs_index_h index = NULL;
    svs_search_params_h search_params = NULL;
    svs_search_results_t results = NULL;

    // Allocate random data
    data = (float*)malloc(NUM_VECTORS * DIMENSION * sizeof(float));
    ids = (size_t*)malloc(NUM_VECTORS * sizeof(size_t));
    queries = (float*)malloc(NUM_QUERIES * DIMENSION * sizeof(float));

    if (!data || !ids || !queries) {
        fprintf(stderr, "Failed to allocate memory\n");
        ret = 1;
        goto cleanup;
    }

    generate_random_data(data, NUM_VECTORS, DIMENSION);
    for (size_t i = 0; i < NUM_VECTORS; i++) {
        ids[i] = i;
    }
    generate_random_data(queries, NUM_QUERIES, DIMENSION);

    // Create Vamana algorithm
    algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
    if (!algorithm) {
        fprintf(stderr, "Failed to create algorithm: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }

    // Create storage
    // Simple storage
    // storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error);
    // storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);

    // LeanVec storage
    size_t leanvec_dims = DIMENSION / 2;
    // OK:
    // storage = svs_storage_create_leanvec(leanvec_dims, SVS_DATA_TYPE_UINT4,
    // SVS_DATA_TYPE_UINT4, error);

    // OK:
    // storage = svs_storage_create_leanvec(leanvec_dims, SVS_DATA_TYPE_UINT4,
    // SVS_DATA_TYPE_UINT8, error);

    // OK:
    storage = svs_storage_create_leanvec(
        leanvec_dims, SVS_DATA_TYPE_UINT8, SVS_DATA_TYPE_UINT8, error
    );

    // ERROR:
    // storage = svs_storage_create_leanvec(leanvec_dims, SVS_DATA_TYPE_UINT8,
    // SVS_DATA_TYPE_UINT4, error);

    // LVQ Storage
    // storage = svs_storage_create_lvq(SVS_DATA_TYPE_UINT4, SVS_DATA_TYPE_VOID, error);

    // storage = svs_storage_create_lvq(SVS_DATA_TYPE_UINT8, SVS_DATA_TYPE_VOID, error);

    // storage = svs_storage_create_lvq(SVS_DATA_TYPE_UINT4, SVS_DATA_TYPE_UINT4, error);

    // storage = svs_storage_create_lvq(SVS_DATA_TYPE_UINT4, SVS_DATA_TYPE_UINT8, error);

    // Scalar Quantized Storage
    // storage = svs_storage_create_sq(SVS_DATA_TYPE_UINT8, error);

    // storage = svs_storage_create_sq(SVS_DATA_TYPE_INT8, error);

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

    // Set custom sequential threadpool
    if (!svs_index_builder_set_threadpool_custom(builder, &sequential_threadpool, error)) {
        fprintf(stderr, "Failed to set threadpool: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }

    // Build index
    printf(
        "Building dynamic index with %d vectors of dimension %d...\n",
        INITIAL_VECTORS,
        DIMENSION
    );
    index = svs_index_build_dynamic(builder, data, ids, INITIAL_VECTORS, 0, error);
    if (!index) {
        fprintf(stderr, "Failed to build index: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }
    printf("Index built successfully!\n");

    // Add more points to the index
    printf("Adding %d more vectors to the index...\n", TAILING_VECTORS);
    size_t num_added = svs_index_dynamic_add_points(
        index,
        data + INITIAL_VECTORS * DIMENSION,
        ids + INITIAL_VECTORS,
        TAILING_VECTORS,
        error
    );
    if (num_added == (size_t)-1) {
        fprintf(
            stderr, "Failed to add points to index: %s\n", svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }
    printf("Points added successfully!\n");

    // Search params
    search_params = svs_search_params_create_vamana(100, error);
    if (!search_params) {
        fprintf(
            stderr, "Failed to create search params: %s\n", svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }

    // Search
    printf("Searching %d queries for top-%d neighbors...\n", NUM_QUERIES, K);
    results = svs_index_search(index, queries, NUM_QUERIES, K, search_params, error);
    if (!results) {
        fprintf(stderr, "Failed to search index: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }
    printf("Search completed successfully!\n");

    // Print results
    size_t offset = 0;
    for (size_t q = 0; q < results->num_queries; q++) {
        printf("Query %zu results:\n", q);
        for (size_t i = 0; i < results->results_per_query[q]; i++) {
            printf(
                "  [%zu] id=%zu, distance=%.4f\n",
                i,
                results->indices[offset + i],
                results->distances[offset + i]
            );
        }
        offset += results->results_per_query[q];
    }
    svs_search_results_free(results);
    results = NULL;

    // Delete some points
    printf(
        "Deleting vectors %d-%d from the index...\n",
        DELETE_VECTORS_BEGIN,
        DELETE_VECTORS_END - 1
    );
    size_t num_deleted = svs_index_dynamic_delete_points(
        index, ids + DELETE_VECTORS_BEGIN, DELETE_VECTORS_END - DELETE_VECTORS_BEGIN, error
    );
    if (num_deleted == (size_t)-1) {
        fprintf(
            stderr, "Failed to delete points from index: %s\n", svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }
    printf("Deleted %zu points successfully!\n", num_deleted);

    // Search again after deletion
    printf("Searching again after deletion...\n");
    results = svs_index_search(index, queries, NUM_QUERIES, K, search_params, error);
    if (!results) {
        fprintf(
            stderr,
            "Failed to search index after deletion: %s\n",
            svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }
    printf("Search after deletion completed successfully!\n");

    // Validate that deleted points are not returned in search results
    printf("Validating results after deletion...\n");
    offset = 0;
    for (size_t q = 0; q < results->num_queries; q++) {
        for (size_t i = 0; i < results->results_per_query[q]; i++) {
            size_t id = results->indices[offset + i];
            if (id >= DELETE_VECTORS_BEGIN && id < DELETE_VECTORS_END) {
                fprintf(stderr, "Error: Deleted id %zu returned in search results!\n", id);
                ret = 1;
            }
        }
        offset += results->results_per_query[q];
    }

    // Check if specific IDs exist in the index
    printf("Checking if specific IDs exist in the index...\n");
    size_t check_ids[] = {
        DELETE_VECTORS_BEGIN - 1,
        DELETE_VECTORS_BEGIN,
        DELETE_VECTORS_BEGIN + 1,
        DELETE_VECTORS_END - 1,
        DELETE_VECTORS_END,
        DELETE_VECTORS_END + 1};
    for (size_t i = 0; i < sizeof(check_ids) / sizeof(check_ids[0]); i++) {
        size_t id = check_ids[i];
        bool has_id = false;
        if (!svs_index_dynamic_has_id(index, id, &has_id, error)) {
            fprintf(
                stderr,
                "Failed to check if index has id %zu: %s\n",
                id,
                svs_error_get_message(error)
            );
            ret = 1;
            continue;
        }
        if (id >= DELETE_VECTORS_BEGIN && id < DELETE_VECTORS_END) {
            if (has_id) {
                fprintf(stderr, "Error: Deleted id %zu still exists in the index!\n", id);
                ret = 1;
            }
        } else {
            if (!has_id) {
                fprintf(stderr, "Error: Existing id %zu not found in the index!\n", id);
                ret = 1;
            }
        }
    }
    if (ret == 0) {
        printf("ID existence validation passed!\n");
    }

    // Get distance to a specific ID
    printf("Getting distance to a specific ID...\n");
    size_t test_id = DELETE_VECTORS_BEGIN - 1;
    float distance = 0.0f;
    if (!svs_index_get_distance(index, test_id, queries, &distance, error)) {
        fprintf(
            stderr,
            "Failed to get distance to id %zu: %s\n",
            test_id,
            svs_error_get_message(error)
        );
        ret = 1;
    } else {
        printf("Distance from id %zu to query[0]: %.4f\n", test_id, distance);
    }

    // Reconstruct a specific ID
    printf("Reconstructing a specific ID...\n");
    float* origin_vector = data + test_id * DIMENSION;
    float* reconstructed = (float*)malloc(DIMENSION * sizeof(float));
    if (!reconstructed) {
        fprintf(stderr, "Failed to allocate memory for reconstruction\n");
        ret = 1;
        goto cleanup;
    }
    if (!svs_index_reconstruct(index, &test_id, 1, reconstructed, DIMENSION, error)) {
        fprintf(
            stderr,
            "Failed to reconstruct id %zu: %s\n",
            test_id,
            svs_error_get_message(error)
        );
        ret = 1;
    } else {
        printf(
            "Original vector for id %zu: [%.4f, %.4f, ...]\n",
            test_id,
            origin_vector[0],
            origin_vector[1]
        );
        printf(
            "Reconstructed vector for id %zu: [%.4f, %.4f, ...]\n",
            test_id,
            reconstructed[0],
            reconstructed[1]
        );
    }
    free(reconstructed);

    // Consolidate the dynamic index
    printf("Consolidating the dynamic index...\n");
    if (!svs_index_dynamic_consolidate(index, error)) {
        fprintf(stderr, "Failed to consolidate index: %s\n", svs_error_get_message(error));
        ret = 1;
    } else {
        printf("Index consolidated successfully!\n");
    }

    // Compact the dynamic index
    printf("Compacting the dynamic index...\n");
    if (!svs_index_dynamic_compact(index, 0, error)) {
        fprintf(stderr, "Failed to compact index: %s\n", svs_error_get_message(error));
        ret = 1;
    } else {
        printf("Index compacted successfully!\n");
    }

    printf("Done!\n");

cleanup:
    // Cleanup
    svs_search_results_free(results);
    svs_search_params_free(search_params);
    svs_index_free(index);
    svs_index_builder_free(builder);
    svs_storage_free(storage);
    svs_algorithm_free(algorithm);
    free(data);
    free(ids);
    free(queries);
    svs_error_free(error);

    return ret;
}
