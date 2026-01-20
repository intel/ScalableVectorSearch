#include "svs/c_api/svs_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_VECTORS 10000
#define NUM_QUERIES 5
#define DIMENSION 128
#define K 10

void generate_random_data(float* data, size_t count, size_t dim) {
    for (size_t i = 0; i < count * dim; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    int ret = 0;
    srand(time(NULL));
    svs_error_h error = svs_error_init();

    float* data = NULL;
    float* queries = NULL;
    svs_algorithm_h algorithm = NULL;
    svs_storage_h storage = NULL;
    svs_index_builder_h builder = NULL;
    svs_index_h index = NULL;
    svs_search_results_t results = NULL;

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

    // Build index
    printf("Building index with %d vectors of dimension %d...\n", NUM_VECTORS, DIMENSION);
    index = svs_index_build(builder, data, NUM_VECTORS, error);
    if (!index) {
        fprintf(stderr, "Failed to build index: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }
    printf("Index built successfully!\n");

    // Search params
    svs_search_params_h search_params = svs_search_params_create_vamana(100, error);
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

    printf("Done!\n");

cleanup:
    // Cleanup
    svs_search_results_free(results);
    svs_index_free(index);
    svs_index_builder_free(builder);
    svs_storage_free(storage);
    svs_algorithm_free(algorithm);
    free(data);
    free(queries);
    svs_error_free(error);

    return ret;
}