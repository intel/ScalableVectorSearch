// required for nftw
#define _XOPEN_SOURCE 500
// required for mkdtemp
#define _GNU_SOURCE

#include "svs/c_api/svs_c.h"
#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define NUM_VECTORS 10000
#define NUM_QUERIES 1
#define DIMENSION 128
#define K 10

void generate_random_data(float* data, size_t count, size_t dim) {
    for (size_t i = 0; i < count * dim; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int nftw_callback(
    const char* fpath, const struct stat* sb, int typeflag, struct FTW* ftwbuf
) {
    if (typeflag == FTW_DP) {
        // directory, remove it
        return rmdir(fpath);
    } else {
        // file, remove it
        return unlink(fpath);
    }
}

int remove_directory_recursive(const char* path) {
    // remove the directory and its contents using function nftw()
    return nftw(path, nftw_callback, 64, FTW_DEPTH | FTW_PHYS);
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
    svs_search_results_t results = NULL;
    char tmp_dir_template[] = "svs_index_XXXXXX";
    char* tmp_dir = NULL;
    svs_search_results_t loaded_results = NULL;

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

    // Search
    printf("Searching %d queries for top-%d neighbors...\n", NUM_QUERIES, K);
    results =
        svs_index_search(index, queries, NUM_QUERIES, K, NULL /* search_params */, error);
    if (!results) {
        fprintf(stderr, "Failed to search index: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }
    printf("Search completed successfully!\n");

    // Create temporary directory for saving the index
    tmp_dir = mkdtemp(tmp_dir_template);
    if (!tmp_dir) {
        fprintf(stderr, "Failed to create temporary directory\n");
        ret = 1;
        goto cleanup;
    }

    printf("Saving index to directory: %s\n", tmp_dir);
    // Save the index to disk
    if (!svs_index_save(index, tmp_dir, error)) {
        fprintf(stderr, "Failed to save index: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }
    printf("Index saved successfully!\n");

    svs_index_free(index);
    index = NULL;
    // Load the index from disk
    printf("Loading index from directory: %s\n", tmp_dir);
    index = svs_index_load(builder, tmp_dir, error);
    if (!index) {
        fprintf(stderr, "Failed to load index: %s\n", svs_error_get_message(error));
        ret = 1;
        goto cleanup;
    }
    printf("Index loaded successfully!\n");

    // Search the loaded index
    printf(
        "Searching loaded index for %d queries for top-%d neighbors...\n", NUM_QUERIES, K
    );
    loaded_results =
        svs_index_search(index, queries, NUM_QUERIES, K, NULL /* search_params */, error);
    if (!loaded_results) {
        fprintf(
            stderr, "Failed to search loaded index: %s\n", svs_error_get_message(error)
        );
        ret = 1;
        goto cleanup;
    }
    printf("Search on loaded index completed successfully!\n");

    // Compare results
    if (results->num_queries != loaded_results->num_queries) {
        fprintf(
            stderr, "Mismatch in number of queries between original and loaded results\n"
        );
        ret = 1;
        goto cleanup;
    }

    size_t offset = 0;
    for (size_t q = 0; q < results->num_queries; q++) {
        if (results->results_per_query[q] != loaded_results->results_per_query[q]) {
            fprintf(stderr, "Mismatch in number of results for query %zu\n", q);
            ret = 1;
            goto cleanup;
        }
        printf("Query %zu results:\n", q);
        for (size_t i = 0; i < results->results_per_query[q]; i++) {
            if (results->indices[offset + i] != loaded_results->indices[offset + i]) {
                fprintf(
                    stderr, "Mismatch in neighbor indices for query %zu, result %zu\n", q, i
                );
                ret = 1;
                goto cleanup;
            }
            printf(
                "  [%zu] id=%zu, distance=%.4f, diff=%.4f\n",
                i,
                results->indices[offset + i],
                results->distances[offset + i],
                results->distances[offset + i] - loaded_results->distances[offset + i]
            );
        }
        offset += results->results_per_query[q];
    }

    printf("Done!\n");

cleanup:
    // Cleanup
    if (tmp_dir) {
        // remove the temporary directory and its contents
        remove_directory_recursive(tmp_dir);
    }
    svs_search_results_free(results);
    svs_search_results_free(loaded_results);
    svs_index_free(index);
    svs_index_builder_free(builder);
    svs_storage_free(storage);
    svs_algorithm_free(algorithm);
    free(data);
    free(queries);
    svs_error_free(error);

    return ret;
}
