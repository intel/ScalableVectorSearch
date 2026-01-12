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
    svs_error_code_t error = SVS_OK;

    float* data = NULL;
    float* queries = NULL;
    svs_algorithm_t algorithm = NULL;
    svs_storage_t storage = NULL;
    svs_index_builder_t builder = NULL;
    svs_index_t index = NULL;
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
    algorithm = svs_algorithm_create_vamana(64, 128, 100, &error);
    if (error != SVS_OK) {
        fprintf(stderr, "Failed to create algorithm: %d\n", error);
        ret = 1;
        goto cleanup;
    }

    // Create storage
    storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, &error);
    if (error != SVS_OK) {
        fprintf(stderr, "Failed to create storage: %d\n", error);
        ret = 1;
        goto cleanup;
    }

    // Create index builder
    builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, &error);
    if (error != SVS_OK) {
        fprintf(stderr, "Failed to create index builder: %d\n", error);
        ret = 1;
        goto cleanup;
    }

    svs_index_builder_set_storage(builder, storage, &error);
    if (error != SVS_OK) {
        fprintf(stderr, "Failed to set storage: %d\n", error);
        ret = 1;
        goto cleanup;
    }

    // Build index
    printf("Building index with %d vectors of dimension %d...\n", NUM_VECTORS, DIMENSION);
    index = svs_index_build(builder, data, NUM_VECTORS, &error);
    if (error != SVS_OK) {
        fprintf(stderr, "Failed to build index: %d\n", error);
        ret = 1;
        goto cleanup;
    }
    printf("Index built successfully!\n");

    // Search
    printf("Searching %d queries for top-%d neighbors...\n", NUM_QUERIES, K);
    results = svs_index_search(index, queries, NUM_QUERIES, K);

    // Print results
    size_t offset = 0;
    for (size_t q = 0; q < results->num_queries; q++) {
        printf("Query %zu results:\n", q);
        for (size_t i = 0; i < results->results_per_query[q]; i++) {
            printf("  [%zu] id=%zu, distance=%.4f\n", 
                   i, results->indices[offset + i], results->distances[offset + i]);
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

    return ret;
}