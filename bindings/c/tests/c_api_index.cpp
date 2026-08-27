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
#include <algorithm>
#include <vector>

CATCH_TEST_CASE("C API Index Build and Search", "[c_api][index][build][search]") {
    const size_t NUM_VECTORS = 100;
    const size_t NUM_QUERIES = 5;
    const size_t DIMENSION = 32;
    const size_t K = 10;
    const size_t NUM_THREADS = 4;

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

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);
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
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index, queries.data(), NUM_QUERIES, K, &results, search_params, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));

        // Validate results structure
        CATCH_REQUIRE(results.num_queries == NUM_QUERIES);
        CATCH_REQUIRE(results.offsets != nullptr);
        CATCH_REQUIRE(results.indices != nullptr);
        CATCH_REQUIRE(results.distances != nullptr);

        // Check that each query returned K results
        for (size_t i = 0; i < NUM_QUERIES; ++i) {
            CATCH_REQUIRE(results.offsets[i + 1] - results.offsets[i] == K);
        }

        // Check that indices are within valid range
        for (size_t i = 0; i < NUM_QUERIES * K; ++i) {
            CATCH_REQUIRE(results.indices[i] < NUM_VECTORS);
        }

        // Check that distances are non-negative
        for (size_t i = 0; i < NUM_QUERIES * K; ++i) {
            CATCH_REQUIRE(results.distances[i] >= 0.0f);
        }

        // Cleanup
        svs_search_results_free(&results);
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

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Search without explicit search parameters (uses defaults)
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index, queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results.num_queries == NUM_QUERIES);

        svs_search_results_free(&results);
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

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);
        CATCH_REQUIRE(storage != nullptr);

        success = svs_index_builder_set_storage(builder, storage, error);
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index, queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(results.num_queries == NUM_QUERIES);

        svs_search_results_free(&results);
        svs_index_free(index);
        svs_storage_free(storage);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Basic Build and Search with Quantized Storages") {
        svs_error_h error = svs_error_create();

        auto run_build_and_search = [&](svs_storage_h storage) {
            CATCH_REQUIRE(storage != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
            CATCH_REQUIRE(algorithm != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_index_builder_h builder = svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
            );
            CATCH_REQUIRE(builder != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            bool success = svs_index_builder_set_threadpool(
                builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
            );
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(svs_error_ok(error));

            success = svs_index_builder_set_storage(builder, storage, error);
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
            CATCH_REQUIRE(index != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
            CATCH_REQUIRE(svs_index_search_topk(
                index, queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, error
            ));
            CATCH_REQUIRE(svs_error_ok(error));
            CATCH_REQUIRE(results.num_queries == NUM_QUERIES);

            for (size_t i = 0; i < NUM_QUERIES; ++i) {
                CATCH_REQUIRE(results.offsets[i + 1] - results.offsets[i] == K);
            }

            svs_search_results_free(&results);
            svs_index_free(index);
            svs_index_builder_free(builder);
            svs_algorithm_free(algorithm);
            svs_storage_free(storage);
        };

        // LeanVec: leanvec_dims = DIMENSION / 2, primary = int4, secondary = int8
        svs_storage_h storage = svs_storage_create_leanvec(
            DIMENSION / 2, SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, error
        );
        CATCH_REQUIRE(check_storage_support(storage, error) == true);
        if (storage_usable(storage)) {
            run_build_and_search(storage);
        }

        // LVQ: primary = int4, residual = int8
        storage = svs_storage_create_lvq(SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, error);
        CATCH_REQUIRE(check_storage_support(storage, error) == true);
        if (storage_usable(storage)) {
            run_build_and_search(storage);
        }

        // Scalar Quantization is available in every build - require it to work.
        storage = svs_storage_create_sq(SVS_DATA_TYPE_INT8, error);
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        run_build_and_search(storage);

        svs_error_free(error);
    }

    CATCH_SECTION("Index with Custom Threadpool") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        // Set custom threadpool
        struct svs_threadpool_interface_ops custom_ops =
            SVS_INIT_THREADPOOL_OPS(sequential_tp_size, sequential_tp_parallel_for);
        struct svs_threadpool_interface custom_pool =
            SVS_MAKE_INTERFACE(nullptr, custom_ops);
        bool success =
            svs_index_builder_set_threadpool_custom(builder, &custom_pool, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify index works with custom threadpool
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index, queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_results_free(&results);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Deprecated svs_index_search wrapper") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Intentionally exercise the deprecated API to ensure the wrapper still delegates
        // correctly to svs_index_search_topk.
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        CATCH_REQUIRE(svs_index_search(
            index, queries.data(), NUM_QUERIES, K, &results, nullptr, error
        ));
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results.num_queries == NUM_QUERIES);

        svs_search_results_free(&results);
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

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Get distance from first vector to first query
        float distance = -1.0f;
        success = svs_index_get_distance(index, 0, queries.data(), &distance, error);
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

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Reconstruct first 3 vectors
        size_t ids[] = {0, 5, 10};
        size_t num_ids = 3;
        std::vector<float> reconstructed(num_ids * DIMENSION);

        success = svs_index_reconstruct(
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

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Test with different K values
        size_t k_values[] = {1, 5, 10, 20};
        for (size_t i = 0; i < sizeof(k_values) / sizeof(k_values[0]); ++i) {
            size_t k = k_values[i];
            svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
            CATCH_REQUIRE(svs_index_search_topk(
                index, queries.data(), NUM_QUERIES, k, &results, nullptr, nullptr, error
            ));
            CATCH_REQUIRE(results.num_queries == NUM_QUERIES);

            for (size_t q = 0; q < NUM_QUERIES; ++q) {
                CATCH_REQUIRE(results.offsets[q + 1] - results.offsets[q] == k);
            }

            svs_search_results_free(&results);
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

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);

        // Perform multiple searches
        for (size_t i = 0; i < 3; ++i) {
            svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
            CATCH_REQUIRE(svs_index_search_topk(
                index, queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, error
            ));
            CATCH_REQUIRE(svs_error_ok(error));
            CATCH_REQUIRE(results.num_queries == NUM_QUERIES);
            svs_search_results_free(&results);
        }

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Save and Load") {
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

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Build index with default threadpool
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Create temporary directory for saving index
        TempDir temp_dir;
        auto temp_path = temp_dir.path();

        // Save the index to disk
        const char* directory = temp_path.c_str();
        success = svs_index_save(index, directory, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Load the index back
        svs_index_h loaded_index = svs_index_load(builder, directory, error);
        CATCH_REQUIRE(loaded_index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Perform search on loaded index
        std::vector<float> queries;
        generate_test_data(queries, 2, DIMENSION);

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            loaded_index, queries.data(), 2, K, &results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results.num_queries == 2);

        // Cleanup
        svs_search_results_free(&results);
        svs_index_free(loaded_index);
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
        struct svs_threadpool_interface_ops custom_ops =
            SVS_INIT_THREADPOOL_OPS(sequential_tp_size, sequential_tp_parallel_for);
        struct svs_threadpool_interface custom_pool =
            SVS_MAKE_INTERFACE(nullptr, custom_ops);
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

CATCH_TEST_CASE("C API Index Memory Management", "[c_api][index][memory]") {
    const size_t NUM_VECTORS = 100;
    const size_t DIMENSION = 32;

    std::vector<float> data;
    generate_test_data(data, NUM_VECTORS, DIMENSION);

    CATCH_SECTION("Memory Accounting Functions") {
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

        bool success =
            svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_NATIVE, 4, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Build index
        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Test get_memory_usage
        size_t memory_usage = 0;
        success = svs_index_get_memory_usage(index, &memory_usage, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(memory_usage > 0);

        // Test get_memory_breakdown
        svs_memory_breakdown_t breakdown = SVS_INIT_MEMORY_BREAKDOWN();
        success = svs_index_get_memory_breakdown(index, &breakdown, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(breakdown.graph_bytes > 0);
        CATCH_REQUIRE(breakdown.data_bytes > 0);
        CATCH_REQUIRE(breakdown.metadata_bytes >= 0);

        // Verify that breakdown.total() == memory_usage
        size_t total =
            breakdown.graph_bytes + breakdown.data_bytes + breakdown.metadata_bytes;
        CATCH_REQUIRE(total == memory_usage);

        // Test null-arg handling for get_memory_usage
        svs_error_h error2 = svs_error_create();
        success = svs_index_get_memory_usage(nullptr, &memory_usage, error2);
        CATCH_REQUIRE(success == false);
        svs_error_free(error2);

        error2 = svs_error_create();
        success = svs_index_get_memory_usage(index, nullptr, error2);
        CATCH_REQUIRE(success == false);
        svs_error_free(error2);

        // Test null-arg handling for get_memory_breakdown
        error2 = svs_error_create();
        success = svs_index_get_memory_breakdown(nullptr, &breakdown, error2);
        CATCH_REQUIRE(success == false);
        svs_error_free(error2);

        error2 = svs_error_create();
        success = svs_index_get_memory_breakdown(index, nullptr, error2);
        CATCH_REQUIRE(success == false);
        svs_error_free(error2);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Estimate Memory vs Actual Breakdown") {
        svs_error_h error = svs_error_create();

        // Build an index and compare its actual memory breakdown against the
        // pre-build estimate produced by svs_index_builder_estimate_memory().
        // `storage` may be nullptr to exercise the default (simple float32) storage.
        auto estimate_and_verify = [&](svs_storage_h storage) {
            svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
            CATCH_REQUIRE(algorithm != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_index_builder_h builder = svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
            );
            CATCH_REQUIRE(builder != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            bool success = svs_index_builder_set_threadpool(
                builder, SVS_THREADPOOL_KIND_NATIVE, 4, error
            );
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(svs_error_ok(error));

            if (storage != nullptr) {
                success = svs_index_builder_set_storage(builder, storage, error);
                CATCH_REQUIRE(success);
                CATCH_REQUIRE(svs_error_ok(error));
            }

            // Estimate before build.
            svs_memory_breakdown_t estimated{};
            success =
                svs_index_builder_estimate_memory(builder, NUM_VECTORS, &estimated, error);
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(svs_error_ok(error));
            CATCH_REQUIRE(estimated.graph_bytes > 0);
            CATCH_REQUIRE(estimated.data_bytes > 0);
            CATCH_REQUIRE(estimated.metadata_bytes > 0);

            // Build the index and query the actual breakdown.
            svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
            CATCH_REQUIRE(index != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_memory_breakdown_t actual{};
            success = svs_index_get_memory_breakdown(index, &actual, error);
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(svs_error_ok(error));

            // Allow up to 1% deviation between the pre-build estimate and the
            // actual allocation (compressed storages may add small per-dataset
            // overhead not accounted for by the estimator, and vice versa).
            auto within_1pct = [](size_t estimate, size_t actual_val) {
                if (estimate == actual_val) {
                    return true;
                }
                const auto [smaller, larger] = std::minmax(estimate, actual_val);
                return (larger - smaller) * 100 <= larger;
            };
            CATCH_REQUIRE(within_1pct(estimated.graph_bytes, actual.graph_bytes));
            CATCH_REQUIRE(estimated.data_bytes == actual.data_bytes);
            CATCH_REQUIRE(within_1pct(estimated.data_bytes, actual.data_bytes));
            CATCH_REQUIRE(within_1pct(estimated.metadata_bytes, actual.metadata_bytes));

            svs_index_free(index);
            svs_index_builder_free(builder);
            svs_algorithm_free(algorithm);
        };

        // Default storage (simple float32).
        estimate_and_verify(nullptr);

        // Simple float16 storage.
        {
            svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);
            CATCH_REQUIRE(check_storage_support(storage, error) == true);
            if (storage != nullptr) {
                estimate_and_verify(storage);
                svs_storage_free(storage);
            }
        }

        // Scalar quantization storage
        {
            svs_storage_h storage = svs_storage_create_sq(SVS_DATA_TYPE_INT8, error);
            CATCH_REQUIRE(check_storage_support(storage, error) == true);
            if (storage != nullptr) {
                estimate_and_verify(storage);
                svs_storage_free(storage);
            }
        }

        // LVQ: primary = int4, residual = int8.
        {
            svs_storage_h storage =
                svs_storage_create_lvq(SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, error);
            CATCH_REQUIRE(check_storage_support(storage, error) == true);
            if (storage != nullptr) {
                estimate_and_verify(storage);
                svs_storage_free(storage);
            }
        }

        // LeanVec: leanvec_dims = DIMENSION / 2, primary = int4, secondary = int8.
        {
            svs_storage_h storage = svs_storage_create_leanvec(
                DIMENSION / 2, SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, error
            );
            CATCH_REQUIRE(check_storage_support(storage, error) == true);
            if (storage != nullptr) {
                estimate_and_verify(storage);
                svs_storage_free(storage);
            }
        }

        // LeanVec: leanvec_dims = DIMENSION / 2, primary = int4, secondary = int4.
        {
            svs_storage_h storage = svs_storage_create_leanvec(
                DIMENSION / 2, SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT4, error
            );
            CATCH_REQUIRE(check_storage_support(storage, error) == true);
            if (storage != nullptr) {
                estimate_and_verify(storage);
                svs_storage_free(storage);
            }
        }

        svs_error_free(error);
    }

    CATCH_SECTION("Estimate Search Memory") {
        const size_t NUM_QUERIES = 5;
        const size_t K = 10;
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Basic estimate using the builder's default search parameters.
        size_t default_size = 0;
        bool success = svs_index_builder_estimate_search_memory(
            builder, NUM_QUERIES, K, nullptr, &default_size, error
        );
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(default_size > 0);

        // The estimate scales linearly with the number of queries.
        size_t double_queries_size = 0;
        success = svs_index_builder_estimate_search_memory(
            builder, NUM_QUERIES * 2, K, nullptr, &double_queries_size, error
        );
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(double_queries_size == default_size * 2);

        // Explicit search parameters yield a valid estimate.
        svs_search_params_h search_params = svs_search_params_create_vamana(50, error);
        CATCH_REQUIRE(search_params != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        size_t params_size = 0;
        success = svs_index_builder_estimate_search_memory(
            builder, NUM_QUERIES, K, search_params, &params_size, error
        );
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(params_size > 0);

        // A larger search window size requires at least as much memory.
        svs_search_params_h large_params = svs_search_params_create_vamana(100, error);
        CATCH_REQUIRE(large_params != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        size_t large_params_size = 0;
        success = svs_index_builder_estimate_search_memory(
            builder, NUM_QUERIES, K, large_params, &large_params_size, error
        );
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(large_params_size >= params_size);

        // Requesting more neighbors than the search window size grows the estimate.
        size_t many_neighbors_size = 0;
        success = svs_index_builder_estimate_search_memory(
            builder, NUM_QUERIES, 200, search_params, &many_neighbors_size, error
        );
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(many_neighbors_size >= params_size);

        // Null-argument handling.
        size_t out_size = 0;
        CATCH_REQUIRE(
            svs_index_builder_estimate_search_memory(
                nullptr, NUM_QUERIES, K, nullptr, &out_size, error
            ) == false
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        CATCH_REQUIRE(
            svs_index_builder_estimate_search_memory(
                builder, NUM_QUERIES, K, nullptr, nullptr, error
            ) == false
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        CATCH_REQUIRE(
            svs_index_builder_estimate_search_memory(
                builder, 0, K, nullptr, &out_size, error
            ) == false
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        CATCH_REQUIRE(
            svs_index_builder_estimate_search_memory(
                builder, NUM_QUERIES, 0, nullptr, &out_size, error
            ) == false
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        svs_search_params_free(large_params);
        svs_search_params_free(search_params);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}

namespace {

// ID filter callback: accepts odd IDs only (~50% selectivity).
bool filter_is_odd(void* /*self*/, size_t id) { return (id % 2) == 1; }

// ID filter callback: accepts IDs strictly below the threshold stored in `self`.
// Used to model a restrictive, low-selectivity filter.
bool filter_below_threshold(void* self, size_t id) {
    return id < *static_cast<const size_t*>(self);
}

// Estimated selectivity callbacks (conservative estimates below true rates).
float filter_rate_odd(void* /*self*/) { return 0.4f; }
float filter_rate_low(void* /*self*/) { return 0.05f; }

} // namespace

CATCH_TEST_CASE("C API Filtered Search topK", "[c_api][index][search][filter]") {
    const size_t NUM_VECTORS = 1000;
    const size_t NUM_QUERIES = 5;
    const size_t DIMENSION = 32;
    const size_t K = 10;
    const size_t NUM_THREADS = 4;

    std::vector<float> data;
    std::vector<float> queries;
    generate_test_data(data, NUM_VECTORS, DIMENSION);
    generate_test_data(queries, NUM_QUERIES, DIMENSION);

    CATCH_SECTION("Normal filter for odd IDs") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_params_h search_params = svs_search_params_create_vamana(50, error);
        CATCH_REQUIRE(search_params != nullptr);

        // ~50% of the IDs pass the filter. Provide a conservative filter_rate estimate
        // (below the true selectivity) so the search is not short-circuited.
        svs_id_filter_interface_ops odd_ops =
            SVS_INIT_ID_FILTER_OPS(filter_is_odd, filter_rate_odd);
        svs_id_filter_interface id_filter = SVS_MAKE_INTERFACE(nullptr, odd_ops);

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index,
            queries.data(),
            NUM_QUERIES,
            K,
            &results,
            search_params,
            &id_filter,
            error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results.num_queries == NUM_QUERIES);

        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            CATCH_REQUIRE(results.offsets[q + 1] - results.offsets[q] == K);
            for (size_t j = 0; j < K; ++j) {
                size_t idx = results.indices[q * K + j];
                // Every neighbor must be a valid, in-range odd ID.
                CATCH_REQUIRE(idx != static_cast<size_t>(-1));
                CATCH_REQUIRE(idx < NUM_VECTORS);
                CATCH_REQUIRE((idx % 2) == 1);
                // Distances must be finite and non-decreasing.
                CATCH_REQUIRE(std::isfinite(results.distances[q * K + j]));
                if (j > 0) {
                    CATCH_REQUIRE(
                        results.distances[q * K + j] >= results.distances[q * K + j - 1]
                    );
                }
            }
        }

        svs_search_results_free(&results);
        svs_search_params_free(search_params);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Low-rate (restrictive) filter") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);

        bool success = svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        );
        CATCH_REQUIRE(success);

        svs_index_h index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_params_h search_params = svs_search_params_create_vamana(100, error);
        CATCH_REQUIRE(search_params != nullptr);

        // Only the lowest 10% of the IDs pass the filter. Provide a conservative
        // filter_rate (below the true selectivity) so the search keeps iterating instead
        // of giving up early.
        size_t max_valid_id = NUM_VECTORS / 10;
        svs_id_filter_interface_ops low_ops =
            SVS_INIT_ID_FILTER_OPS(filter_below_threshold, filter_rate_low);
        svs_id_filter_interface id_filter = SVS_MAKE_INTERFACE(&max_valid_id, low_ops);

        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            index,
            queries.data(),
            NUM_QUERIES,
            K,
            &results,
            search_params,
            &id_filter,
            error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results.num_queries == NUM_QUERIES);

        size_t total_found = 0;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            const size_t* indices = nullptr;
            const float* distances = nullptr;
            size_t num_neighbors = 0;
            svs_search_results_row(&results, q, &indices, &distances, &num_neighbors);
            CATCH_REQUIRE(num_neighbors > 0);
            CATCH_REQUIRE(num_neighbors <= K);
            for (size_t j = 0; j < num_neighbors; ++j) {
                size_t idx = indices[j];
                // Padding (unspecified) entries are allowed for a restrictive filter, but
                // any specified neighbor must pass the filter predicate.
                if (idx != static_cast<size_t>(-1)) {
                    CATCH_REQUIRE(idx < max_valid_id);
                    CATCH_REQUIRE(std::isfinite(distances[j]));
                    ++total_found;
                }
            }
        }
        // The restrictive filter still has plenty of matching vectors, so the search must
        // return at least some valid neighbors.
        CATCH_REQUIRE(total_found > 0);

        svs_search_results_free(&results);
        svs_search_params_free(search_params);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}
