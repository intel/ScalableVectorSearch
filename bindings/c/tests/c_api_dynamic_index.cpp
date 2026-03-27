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
#include <vector>

namespace {

// Helper function to generate test data
void generate_test_data(std::vector<float>& data, size_t num_vectors, size_t dimension) {
    data.resize(num_vectors * dimension);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((i * 13) % 100) / 100.0f;
    }
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

CATCH_TEST_CASE("C API Dynamic Index", "[c_api][index][dynamic]") {
    const size_t NUM_VECTORS = 50;
    const size_t DIMENSION = 32;
    const size_t K = 5;

    std::vector<float> data;
    std::vector<size_t> ids(NUM_VECTORS);
    generate_test_data(data, NUM_VECTORS, DIMENSION);

    // Generate sequential IDs
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        ids[i] = i;
    }

    CATCH_SECTION("Dynamic Index Build with IDs") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);

        // Build dynamic index with explicit IDs
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Build without IDs") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);

        // Build dynamic index without explicit IDs (auto-generated)
        svs_index_h index =
            svs_index_build_dynamic(builder, data.data(), nullptr, NUM_VECTORS, 0, error);
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Has ID") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Check for existing IDs
        for (size_t i = 0; i < 5; ++i) {
            bool has_id = false;
            bool success = svs_index_dynamic_has_id(index, ids[i], &has_id, error);
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(svs_error_ok(error));
            CATCH_REQUIRE(has_id == true);
        }

        // Check for non-existing ID
        bool has_id = false;
        bool success = svs_index_dynamic_has_id(index, NUM_VECTORS + 100, &has_id, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(has_id == false);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Add Points") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Add new points
        size_t num_new_points = 5;
        std::vector<float> new_data;
        std::vector<size_t> new_ids(num_new_points);
        generate_test_data(new_data, num_new_points, DIMENSION);

        for (size_t i = 0; i < num_new_points; ++i) {
            new_ids[i] = NUM_VECTORS + i;
        }

        size_t added_count = svs_index_dynamic_add_points(
            index, new_data.data(), new_ids.data(), num_new_points, error
        );
        CATCH_REQUIRE(added_count == num_new_points);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify new IDs exist
        for (size_t i = 0; i < num_new_points; ++i) {
            bool has_id = false;
            bool success = svs_index_dynamic_has_id(index, new_ids[i], &has_id, error);
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(has_id == true);
        }

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Delete Points") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Delete some points
        size_t ids_to_delete[] = {0, 5, 10};
        size_t num_to_delete = 3;

        size_t deleted_count =
            svs_index_dynamic_delete_points(index, ids_to_delete, num_to_delete, error);
        CATCH_REQUIRE(deleted_count == num_to_delete);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify deleted IDs don't exist
        for (size_t i = 0; i < num_to_delete; ++i) {
            bool has_id = false;
            bool success =
                svs_index_dynamic_has_id(index, ids_to_delete[i], &has_id, error);
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(has_id == false);
        }

        // Verify other IDs still exist
        bool has_id = false;
        bool success = svs_index_dynamic_has_id(index, 1, &has_id, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(has_id == true);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Add and Delete") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Delete some points
        size_t ids_to_delete[] = {0, 1};
        svs_index_dynamic_delete_points(index, ids_to_delete, 2, error);
        CATCH_REQUIRE(svs_error_ok(error));

        // Add new points with the deleted IDs
        std::vector<float> new_data;
        generate_test_data(new_data, 2, DIMENSION);

        size_t added_count =
            svs_index_dynamic_add_points(index, new_data.data(), ids_to_delete, 2, error);
        CATCH_REQUIRE(added_count == 2);
        CATCH_REQUIRE(svs_error_ok(error));

        // Verify IDs exist again
        for (size_t i = 0; i < 2; ++i) {
            bool has_id = false;
            bool success =
                svs_index_dynamic_has_id(index, ids_to_delete[i], &has_id, error);
            CATCH_REQUIRE(success);
            CATCH_REQUIRE(has_id == true);
        }

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Consolidate") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Add and delete some points
        std::vector<float> new_data;
        std::vector<size_t> new_ids = {NUM_VECTORS, NUM_VECTORS + 1};
        generate_test_data(new_data, 2, DIMENSION);

        svs_index_dynamic_add_points(index, new_data.data(), new_ids.data(), 2, error);

        size_t ids_to_delete[] = {0, 1};
        svs_index_dynamic_delete_points(index, ids_to_delete, 2, error);

        // Consolidate the index
        bool success = svs_index_dynamic_consolidate(index, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Compact") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Compact the index
        bool success = svs_index_dynamic_compact(index, 0, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Delete some points
        size_t ids_to_delete[] = {0, 1, 2};
        svs_index_dynamic_delete_points(index, ids_to_delete, 3, error);
        CATCH_REQUIRE(svs_error_ok(error));

        // Consolidate the index
        success = svs_index_dynamic_consolidate(index, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Compact the index
        success = svs_index_dynamic_compact(index, 0, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Search After Modifications") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Add some points
        std::vector<float> new_data;
        std::vector<size_t> new_ids = {NUM_VECTORS, NUM_VECTORS + 1, NUM_VECTORS + 2};
        generate_test_data(new_data, 3, DIMENSION);
        svs_index_dynamic_add_points(index, new_data.data(), new_ids.data(), 3, error);

        // Delete some points
        size_t ids_to_delete[] = {0, 1};
        svs_index_dynamic_delete_points(index, ids_to_delete, 2, error);

        // Perform search
        std::vector<float> queries;
        generate_test_data(queries, 2, DIMENSION);

        svs_search_results_t results =
            svs_index_search(index, queries.data(), 2, K, nullptr, error);
        CATCH_REQUIRE(results != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results->num_queries == 2);

        // Verify deleted IDs don't appear in results
        for (size_t i = 0; i < results->num_queries * K; ++i) {
            size_t result_id = results->indices[i];
            CATCH_REQUIRE(result_id != 0);
            CATCH_REQUIRE(result_id != 1);
        }

        svs_search_results_free(results);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Dynamic Index Delete Non-existing ID") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, 0, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Try to delete non-existing ID
        size_t non_existing_id = NUM_VECTORS + 1000;
        size_t deleted_count =
            svs_index_dynamic_delete_points(index, &non_existing_id, 1, error);
        // Should return 0 for non-existing ID
        CATCH_REQUIRE(deleted_count == 0);

        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}
