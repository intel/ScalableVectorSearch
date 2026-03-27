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
#include "svs/c_api/svs_c.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// Standard library
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Helper function to generate test data
void generate_test_data(std::vector<float>& data, size_t num_vectors, size_t dimension) {
    data.resize(num_vectors * dimension);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((i * 7) % 100) / 100.0f;
    }
}

// Helper to calculate Euclidean distance
float euclidean_distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Sequential threadpool for testing
size_t sequential_tp_size(void* /*self*/) { return 1; }

void sequential_tp_parallel_for(
    void* /*self*/, void (*func)(void*, size_t), void* svs_param, size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        func(svs_param, i);
    }
}

} // namespace

CATCH_TEST_CASE("C API Index Build and Search", "[c_api][index][build][search]") {
    const size_t NUM_VECTORS = 100;
    const size_t NUM_QUERIES = 5;
    const size_t DIMENSION = 32;
    const size_t K = 10;

    std::vector<float> data;
    std::vector<float> queries;
    generate_test_data(data, NUM_VECTORS, DIMENSION);
    generate_test_data(queries, NUM_QUERIES, DIMENSION);

    CATCH_SECTION("Basic Index Build and Search") {
        svs_error_h error = svs_error_create();

        // Create algorithm
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Create builder
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Build index with default threadpool
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Create search parameters
        svs_search_params_h search_params = svs_search_params_create_vamana(50, error);
        CATCH_REQUIRE(search_params != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Perform search
        svs_search_results_t results =
            svs_index_search(index, queries.data(), NUM_QUERIES, K, search_params, error);
        CATCH_REQUIRE(results != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Validate results structure
        CATCH_REQUIRE(results->num_queries == NUM_QUERIES);
        CATCH_REQUIRE(results->results_per_query != nullptr);
        CATCH_REQUIRE(results->indices != nullptr);
        CATCH_REQUIRE(results->distances != nullptr);

        // Check that each query returned K results
        for (size_t i = 0; i < NUM_QUERIES; ++i) {
            CATCH_REQUIRE(results->results_per_query[i] == K);
        }

        // Check that indices are within valid range
        for (size_t i = 0; i < NUM_QUERIES * K; ++i) {
            CATCH_REQUIRE(results->indices[i] < NUM_VECTORS);
        }

        // Check that distances are non-negative
        for (size_t i = 0; i < NUM_QUERIES * K; ++i) {
            CATCH_REQUIRE(results->distances[i] >= 0.0f);
        }

        // Cleanup
        svs_search_results_free(results);
        svs_search_params_free(search_params);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Search without Search Parameters") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Search without explicit search parameters (uses defaults)
        svs_search_results_t results =
            svs_index_search(index, queries.data(), NUM_QUERIES, K, nullptr, error);
        CATCH_REQUIRE(results != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results->num_queries == NUM_QUERIES);

        svs_search_results_free(results);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index with Different Storage Types") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);

        // Test with Float16 storage
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);

        svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);
        CATCH_REQUIRE(storage != nullptr);

        bool success = svs_index_builder_set_storage(builder, storage, error);
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        svs_search_results_t results =
            svs_index_search(index, queries.data(), NUM_QUERIES, K, nullptr, error);
        CATCH_REQUIRE(results != nullptr);
        CATCH_REQUIRE(results->num_queries == NUM_QUERIES);

        svs_search_results_free(results);
        svs_index_free(index);
        svs_storage_free(storage);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index with Custom Threadpool") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        // Set custom threadpool
        struct svs_threadpool_interface custom_pool = {
            {sequential_tp_size, sequential_tp_parallel_for}, nullptr};
        bool success =
            svs_index_builder_set_threadpool_custom(builder, &custom_pool, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify index works with custom threadpool
        svs_search_results_t results =
            svs_index_search(index, queries.data(), NUM_QUERIES, K, nullptr, error);
        CATCH_REQUIRE(results != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_results_free(results);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Get Distance") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Get distance from first vector to first query
        float distance = -1.0f;
        bool success = svs_index_get_distance(index, 0, queries.data(), &distance, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(distance >= 0.0f);

        // Verify distance is approximately correct
        float expected_distance =
            euclidean_distance(data.data(), queries.data(), DIMENSION);
        CATCH_REQUIRE(std::abs(distance - expected_distance) < 0.1f);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Reconstruct") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Reconstruct first 3 vectors
        size_t ids[] = {0, 5, 10};
        size_t num_ids = 3;
        std::vector<float> reconstructed(num_ids * DIMENSION);

        bool success = svs_index_reconstruct(
            index, ids, num_ids, reconstructed.data(), DIMENSION, error
        );
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify reconstructed data is close to original
        for (size_t i = 0; i < num_ids; ++i) {
            size_t id = ids[i];
            const float* original = &data[id * DIMENSION];
            const float* recon = &reconstructed[i * DIMENSION];

            float distance = euclidean_distance(original, recon, DIMENSION);
            CATCH_REQUIRE(distance < 1.0f); // Allow some reconstruction error
        }

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Search with Different K Values") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Test with different K values
        size_t k_values[] = {1, 5, 10, 20};
        for (size_t i = 0; i < sizeof(k_values) / sizeof(k_values[0]); ++i) {
            size_t k = k_values[i];
            svs_search_results_t results =
                svs_index_search(index, queries.data(), NUM_QUERIES, k, nullptr, error);
            CATCH_REQUIRE(results != nullptr);
            CATCH_REQUIRE(results->num_queries == NUM_QUERIES);

            for (size_t q = 0; q < NUM_QUERIES; ++q) {
                CATCH_REQUIRE(results->results_per_query[q] == k);
            }

            svs_search_results_free(results);
        }

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Multiple Searches on Same Index") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Perform multiple searches
        for (size_t i = 0; i < 3; ++i) {
            svs_search_results_t results =
                svs_index_search(index, queries.data(), NUM_QUERIES, K, nullptr, error);
            CATCH_REQUIRE(results != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));
            CATCH_REQUIRE(results->num_queries == NUM_QUERIES);
            svs_search_results_free(results);
        }

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}

CATCH_TEST_CASE("C API Threadpool Management", "[c_api][index][threadpool]") {
    const size_t NUM_VECTORS = 100;
    const size_t DIMENSION = 32;

    std::vector<float> data;
    generate_test_data(data, NUM_VECTORS, DIMENSION);

    CATCH_SECTION("Native Threadpool Get/Set Num Threads") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        // Set native threadpool
        bool success =
            svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_NATIVE, 2, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Get current number of threads
        size_t num_threads = 0;
        success = svs_index_get_num_threads(index, &num_threads, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(num_threads == 2);

        // Set to different number of threads
        success = svs_index_set_num_threads(index, 4, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify the change
        success = svs_index_get_num_threads(index, &num_threads, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(num_threads == 4);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("OMP Threadpool Get/Set Num Threads") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        // Set OMP threadpool
        bool success =
            svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_OMP, 3, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Get current number of threads
        size_t num_threads = 0;
        success = svs_index_get_num_threads(index, &num_threads, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(num_threads == 3);

        // Set to different number of threads
        success = svs_index_set_num_threads(index, 5, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify the change
        success = svs_index_get_num_threads(index, &num_threads, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(num_threads == 5);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Custom Threadpool Get/Set Num Threads") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        // Set custom threadpool
        struct svs_threadpool_interface custom_pool = {
            {sequential_tp_size, sequential_tp_parallel_for}, nullptr};
        bool success =
            svs_index_builder_set_threadpool_custom(builder, &custom_pool, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Get number of threads from custom threadpool
        size_t num_threads = 0;
        success = svs_index_get_num_threads(index, &num_threads, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(num_threads == 1); // Sequential threadpool reports size 1

        // Setting num_threads on custom threadpool should fail with
        // SVS_ERROR_INVALID_OPERATION
        success = svs_index_set_num_threads(index, 2, error);
        CATCH_REQUIRE_FALSE(success);
        CATCH_REQUIRE_FALSE(svs_error_ok(error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_OPERATION);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Single Thread Threadpool") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        // Set single thread threadpool
        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_SINGLE_THREAD, 1, error
        );
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Get number of threads
        size_t num_threads = 0;
        success = svs_index_get_num_threads(index, &num_threads, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(num_threads == 1);

        // Try to set number of threads (should fail with SVS_ERROR_INVALID_OPERATION since
        // it's single thread)
        success = svs_index_set_num_threads(index, 2, error);
        CATCH_REQUIRE_FALSE(success);
        CATCH_REQUIRE_FALSE(svs_error_ok(error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_OPERATION);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Default Threadpool") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        // Don't set any threadpool - use default
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Get number of threads from default threadpool
        size_t num_threads = 0;
        bool success = svs_index_get_num_threads(index, &num_threads, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(num_threads > 0); // Should have at least 1 thread

        // Try to set number of threads
        success = svs_index_set_num_threads(index, 2, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify the change
        success = svs_index_get_num_threads(index, &num_threads, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(num_threads == 2);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Invalid Set Num Threads") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Try to set to 0 threads (invalid) - should fail with SVS_ERROR_INVALID_ARGUMENT
        bool success = svs_index_set_num_threads(index, 0, error);
        CATCH_REQUIRE(success == false);
        CATCH_REQUIRE(svs_error_ok(error) == false);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}
