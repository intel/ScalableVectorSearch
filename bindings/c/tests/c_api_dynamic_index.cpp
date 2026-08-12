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

// Test utilities
#include "c_api_test_utils.h"

// Standard library
#include <vector>

CATCH_TEST_CASE("C API Dynamic Index", "[c_api][index][dynamic]") {
    const size_t NUM_VECTORS = 50;
    const size_t DIMENSION = 32;
    const size_t K = 5;
    const size_t BLOCK_SIZE = 1024 * 1024; // 1 MB block size for testing

    std::vector<float> data;
    std::vector<size_t> ids(NUM_VECTORS);
    generate_test_data(data, NUM_VECTORS, DIMENSION);

    // Generate sequential IDs
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        ids[i] = i;
    }

    svs_error_h error = svs_error_create();

    svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
    CATCH_REQUIRE(algorithm != nullptr);

    svs_index_builder_h builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
    );
    CATCH_REQUIRE(builder != nullptr);

    // Set single thread threadpool for testing
    bool success = svs_index_builder_set_threadpool(
        builder, SVS_THREADPOOL_KIND_SINGLE_THREAD, 1, error
    );
    CATCH_REQUIRE(success);
    CATCH_REQUIRE(svs_error_ok(error));

    CATCH_SECTION("Dynamic Index Build with IDs") {
        // Build dynamic index with explicit IDs
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_free(index);
    }

    CATCH_SECTION("Dynamic Index Build without IDs") {
        // Build dynamic index without explicit IDs (auto-generated)
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), nullptr, NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_index_free(index);
    }

    CATCH_SECTION("Dynamic Index Has ID") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
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
    }

    CATCH_SECTION("Dynamic Index Add Points") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
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
    }

    CATCH_SECTION("Dynamic Index Delete Points") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
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
    }

    CATCH_SECTION("Dynamic Index Add and Delete") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
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
    }

    CATCH_SECTION("Dynamic Index Consolidate") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
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
    }

    CATCH_SECTION("Dynamic Index Compact") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
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
    }

    CATCH_SECTION("Dynamic Index Search After Modifications") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
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
            svs_index_search_topK(index, queries.data(), 2, K, nullptr, nullptr, error);
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
    }

    CATCH_SECTION("Dynamic Index Delete Non-existing ID") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Try to delete non-existing ID
        size_t non_existing_id = NUM_VECTORS + 1000;
        size_t deleted_count =
            svs_index_dynamic_delete_points(index, &non_existing_id, 1, error);
        // Should return 0 for non-existing ID and no error
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(deleted_count == 0);

        // Try to delete mix of existing and non-existing IDs
        size_t ids_to_delete[] = {0, non_existing_id};
        deleted_count = svs_index_dynamic_delete_points(index, ids_to_delete, 2, error);
        // Should return 1 for the existing ID and no error
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(deleted_count == 1);

        svs_index_free(index);
    }

    CATCH_SECTION("Dynamic Index Save and Load") {
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(index != nullptr);

        // Create temporary directory for saving index
        TempDir temp_dir;
        auto temp_path = temp_dir.path();

        // Save the index to disk
        const char* directory = temp_path.c_str();
        bool success = svs_index_save(index, directory, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));

        // Load the index back
        svs_index_h loaded_index =
            svs_index_load_dynamic(builder, directory, BLOCK_SIZE, error);
        CATCH_REQUIRE(loaded_index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Perform search on loaded index
        std::vector<float> queries;
        generate_test_data(queries, 2, DIMENSION);

        svs_search_results_t results = svs_index_search_topK(
            loaded_index, queries.data(), 2, K, nullptr, nullptr, error
        );
        CATCH_REQUIRE(results != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results->num_queries == 2);

        svs_search_results_free(results);
        svs_index_free(loaded_index);
        svs_index_free(index);
    }
}

CATCH_TEST_CASE("C API Dynamic Index Memory", "[c_api][index][memory][dynamic]") {
    // TODO: fix the blocked memory breakdown reported by index for LVQ, LeanVec storages.
    // For now, we will:
    // * test only the default simple and SQ storages.
    // * Align graph and data sizes to BLOCK_SIZE to avoid test failures.
    const size_t BLOCK_SIZE = 8 * 1024; // 8 KB block size for testing
    const size_t DIMENSION = 32;
    const size_t GRAPH_DEGREE = 16;
    const size_t NUM_VECTORS = BLOCK_SIZE / DIMENSION; // full blocks of data
    const size_t K = 5;

    std::vector<float> data;
    std::vector<size_t> ids(NUM_VECTORS);
    generate_test_data(data, NUM_VECTORS, DIMENSION);

    // Generate sequential IDs
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        ids[i] = i;
    }

    svs_error_h error = svs_error_create();

    svs_algorithm_h algorithm = svs_algorithm_create_vamana(GRAPH_DEGREE, 100, 100, error);
    CATCH_REQUIRE(algorithm != nullptr);

    svs_index_builder_h builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
    );
    CATCH_REQUIRE(builder != nullptr);

    CATCH_SECTION("Memory Accounting Functions") {
        // Build dynamic index
        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // Test get_memory_usage
        size_t memory_usage = 0;
        bool success = svs_index_get_memory_usage(index, &memory_usage, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(memory_usage > 0);

        // Test get_memory_breakdown
        svs_memory_breakdown_t breakdown;
        success = svs_index_get_memory_breakdown(index, &breakdown, error);
        CATCH_REQUIRE(success);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(breakdown.graph_bytes > 0);
        CATCH_REQUIRE(breakdown.data_bytes > 0);
        CATCH_REQUIRE(breakdown.metadata_bytes > 0);

        // Verify that breakdown.total() == memory_usage
        size_t total =
            breakdown.graph_bytes + breakdown.data_bytes + breakdown.metadata_bytes;
        CATCH_REQUIRE(total == memory_usage);

        svs_index_free(index);
    }

    CATCH_SECTION("Estimate Memory vs Actual Breakdown") {
        // Build a dynamic index and compare its actual memory breakdown against
        // the pre-build estimate produced by
        // svs_index_builder_estimate_memory_dynamic(). `storage` may be nullptr
        // to exercise the default (simple float32) storage.
        auto estimate_and_verify = [&](svs_storage_h storage) {
            svs_algorithm_h local_algorithm =
                svs_algorithm_create_vamana(16, 32, 50, error);
            CATCH_REQUIRE(local_algorithm != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_index_builder_h local_builder = svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, local_algorithm, error
            );
            CATCH_REQUIRE(local_builder != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            bool ok = svs_index_builder_set_threadpool(
                local_builder, SVS_THREADPOOL_KIND_NATIVE, 4, error
            );
            CATCH_REQUIRE(ok);
            CATCH_REQUIRE(svs_error_ok(error));

            if (storage != nullptr) {
                ok = svs_index_builder_set_storage(local_builder, storage, error);
                CATCH_REQUIRE(ok);
                CATCH_REQUIRE(svs_error_ok(error));
            }

            // Estimate before build.
            svs_memory_breakdown_t estimated{};
            ok = svs_index_builder_estimate_memory_dynamic(
                local_builder, NUM_VECTORS, BLOCK_SIZE, &estimated, error
            );
            CATCH_REQUIRE(ok);
            CATCH_REQUIRE(svs_error_ok(error));
            CATCH_REQUIRE(estimated.graph_bytes > 0);
            CATCH_REQUIRE(estimated.data_bytes > 0);
            CATCH_REQUIRE(estimated.metadata_bytes > 0);

            // Build the dynamic index and query the actual breakdown.
            svs_index_h index = svs_index_build_dynamic(
                local_builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
            );
            CATCH_REQUIRE(index != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_memory_breakdown_t actual{};
            ok = svs_index_get_memory_breakdown(index, &actual, error);
            CATCH_REQUIRE(ok);
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
            CATCH_REQUIRE(within_1pct(estimated.data_bytes, actual.data_bytes));
            CATCH_REQUIRE(within_1pct(estimated.metadata_bytes, actual.metadata_bytes));

            svs_index_free(index);
            svs_index_builder_free(local_builder);
            svs_algorithm_free(local_algorithm);
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

        // Scalar quantization storage.
        {
            svs_storage_h storage = svs_storage_create_sq(SVS_DATA_TYPE_INT8, error);
            CATCH_REQUIRE(check_storage_support(storage, error) == true);
            if (storage != nullptr) {
                estimate_and_verify(storage);
                svs_storage_free(storage);
            }
        }
    }

    svs_index_builder_free(builder);
    svs_algorithm_free(algorithm);
    svs_error_free(error);
}

namespace {

// ID filter callback: accepts odd IDs only (~50% selectivity).
bool filter_is_odd(void* /*self*/, size_t id) { return (id % 2) == 1; }

// ID filter callback: accepts IDs strictly below the threshold stored in `self`.
// Used to model a restrictive, low-selectivity filter.
bool filter_below_threshold(void* self, size_t id) {
    return id < *static_cast<const size_t*>(self);
}

} // namespace

CATCH_TEST_CASE(
    "C API Dynamic Filtered Search topK", "[c_api][index][dynamic][search][filter]"
) {
    const size_t NUM_VECTORS = 1000;
    const size_t NUM_QUERIES = 5;
    const size_t DIMENSION = 32;
    const size_t K = 10;
    const size_t NUM_THREADS = 4;
    const size_t BLOCK_SIZE = 1024 * 1024; // 1 MB block size for testing

    std::vector<float> data;
    std::vector<float> queries;
    std::vector<size_t> ids(NUM_VECTORS);
    generate_test_data(data, NUM_VECTORS, DIMENSION);
    generate_test_data(queries, NUM_QUERIES, DIMENSION);
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        ids[i] = i;
    }

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

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_params_h search_params = svs_search_params_create_vamana(50, error);
        CATCH_REQUIRE(search_params != nullptr);

        // ~50% of the IDs pass the filter. Provide a conservative filter_rate estimate
        // (below the true selectivity) so the search is not short-circuited.
        svs_id_filter_interface id_filter{};
        id_filter.ops.is_member = &filter_is_odd;
        id_filter.self = nullptr;
        id_filter.filter_rate = 0.4f;

        svs_search_results_t results = svs_index_search_topK(
            index, queries.data(), NUM_QUERIES, K, search_params, &id_filter, error
        );
        CATCH_REQUIRE(results != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results->num_queries == NUM_QUERIES);

        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            CATCH_REQUIRE(results->results_per_query[q] == K);
            for (size_t j = 0; j < K; ++j) {
                size_t idx = results->indices[q * K + j];
                // Every neighbor must be a valid, in-range odd ID.
                CATCH_REQUIRE(idx != static_cast<size_t>(-1));
                CATCH_REQUIRE(idx < NUM_VECTORS);
                CATCH_REQUIRE((idx % 2) == 1);
                // Distances must be finite and non-decreasing.
                CATCH_REQUIRE(std::isfinite(results->distances[q * K + j]));
                if (j > 0) {
                    CATCH_REQUIRE(
                        results->distances[q * K + j] >= results->distances[q * K + j - 1]
                    );
                }
            }
        }

        svs_search_results_free(results);
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

        svs_index_h index = svs_index_build_dynamic(
            builder, data.data(), ids.data(), NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        svs_search_params_h search_params = svs_search_params_create_vamana(100, error);
        CATCH_REQUIRE(search_params != nullptr);

        // Only the lowest 10% of the IDs pass the filter. Provide a conservative
        // filter_rate (below the true selectivity) so the search keeps iterating instead
        // of giving up early.
        size_t max_valid_id = NUM_VECTORS / 10;
        svs_id_filter_interface id_filter{};
        id_filter.ops.is_member = &filter_below_threshold;
        id_filter.self = &max_valid_id;
        id_filter.filter_rate = 0.05f;

        svs_search_results_t results = svs_index_search_topK(
            index, queries.data(), NUM_QUERIES, K, search_params, &id_filter, error
        );
        CATCH_REQUIRE(results != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(results->num_queries == NUM_QUERIES);

        size_t total_found = 0;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            CATCH_REQUIRE(results->results_per_query[q] == K);
            for (size_t j = 0; j < K; ++j) {
                size_t idx = results->indices[q * K + j];
                // Padding (unspecified) entries are allowed for a restrictive filter, but
                // any specified neighbor must pass the filter predicate.
                if (idx != static_cast<size_t>(-1)) {
                    CATCH_REQUIRE(idx < max_valid_id);
                    CATCH_REQUIRE(std::isfinite(results->distances[q * K + j]));
                    ++total_found;
                }
            }
        }
        // The restrictive filter still has plenty of matching vectors, so the search must
        // return at least some valid neighbors.
        CATCH_REQUIRE(total_found > 0);

        svs_search_results_free(results);
        svs_search_params_free(search_params);
        svs_index_free(index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}
