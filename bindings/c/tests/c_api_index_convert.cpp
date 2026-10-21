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

CATCH_TEST_CASE("C API Index Conversion", "[c_api][index][convert]") {
    const size_t NUM_VECTORS = 100;
    const size_t NUM_QUERIES = 5;
    const size_t DIMENSION = 32;
    const size_t K = 10;
    const size_t NUM_THREADS = 4;

    std::vector<float> data;
    std::vector<float> queries;
    generate_test_data(data, NUM_VECTORS, DIMENSION);
    generate_test_data(queries, NUM_QUERIES, DIMENSION);

    CATCH_SECTION("Vamana Convert") {
        svs_error_h error = svs_error_create();

        svs_search_params_h search_params = svs_search_params_create_vamana(50, error);
        CATCH_REQUIRE(search_params != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        auto simple_fp32 = [&]() {
            svs_storage_h s = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error);
            CATCH_REQUIRE(s != nullptr);
            return s;
        };
        auto simple_fp16 = [&]() {
            svs_storage_h s = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);
            CATCH_REQUIRE(s != nullptr);
            return s;
        };
        auto scalar_int8 = [&]() {
            svs_storage_h s = svs_storage_create_sq(SVS_DATA_TYPE_INT8, error);
            CATCH_REQUIRE(s != nullptr);
            return s;
        };
        auto lvq_4x8 = [&]() {
            svs_storage_h s =
                svs_storage_create_lvq(SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, error);
            CATCH_REQUIRE(check_storage_support(s, error) == true);
            return s;
        };
        auto leanvec_4x8 = [&]() {
            svs_storage_h s = svs_storage_create_leanvec(
                DIMENSION / 2, SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, error
            );
            CATCH_REQUIRE(check_storage_support(s, error) == true);
            return s;
        };

        auto search_index = [&](svs_index_h index, svs_search_results_t& out) {
            CATCH_REQUIRE(svs_index_search_topk(
                index, queries.data(), NUM_QUERIES, K, &out, search_params, nullptr, error
            ));
            CATCH_REQUIRE(svs_error_ok(error));
        };

        // Mean per-query recall@K of the copy's neighbors against the source's.
        auto mean_recall = [&](const svs_search_results_t& src,
                               const svs_search_results_t& copy) {
            double total = 0.0;
            for (size_t q = 0; q < NUM_QUERIES; ++q) {
                size_t matches = 0;
                for (size_t i = src.offsets[q]; i < src.offsets[q + 1]; ++i) {
                    for (size_t j = copy.offsets[q]; j < copy.offsets[q + 1]; ++j) {
                        if (src.indices[i] == copy.indices[j]) {
                            ++matches;
                            break;
                        }
                    }
                }
                size_t k = src.offsets[q + 1] - src.offsets[q];
                total += (k == 0) ? 1.0 : static_cast<double>(matches) / k;
            }
            return total / NUM_QUERIES;
        };

        // Builds a source index using `src_storage`, copies it into a fresh builder
        // configured with `dst_storage`, then checks the copy reproduces the source's
        // neighbors to within `min_recall`. Takes ownership of both storage handles.
        auto run_copy_case = [&](svs_storage_h src_storage,
                                 svs_storage_h dst_storage,
                                 double min_recall) {
            // Skip the test case if either the source or destination storage is not usable.
            // E.g. LVQ/Leanvec is not available on this platform
            if (!storage_usable(src_storage) || !storage_usable(dst_storage)) {
                svs_storage_free(dst_storage);
                svs_storage_free(src_storage);
                return;
            }
            svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
            CATCH_REQUIRE(algorithm != nullptr);

            svs_index_builder_h builder = svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
            );
            CATCH_REQUIRE(builder != nullptr);
            CATCH_REQUIRE(svs_index_builder_set_threadpool(
                builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
            ));
            CATCH_REQUIRE(svs_index_builder_set_storage(builder, src_storage, error));

            svs_index_h src_index =
                svs_index_build(builder, data.data(), NUM_VECTORS, error);
            CATCH_REQUIRE(src_index != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            // Reuse the same builder for the copy, swapping in the destination storage.
            CATCH_REQUIRE(svs_index_builder_set_storage(builder, dst_storage, error));
            svs_index_h copy_index = svs_index_convert(builder, src_index, error);
            CATCH_REQUIRE(copy_index != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_search_results_t src_results = SVS_INIT_SEARCH_RESULTS();
            svs_search_results_t copy_results = SVS_INIT_SEARCH_RESULTS();
            search_index(src_index, src_results);
            search_index(copy_index, copy_results);

            CATCH_REQUIRE(copy_results.num_queries == src_results.num_queries);
            for (size_t q = 0; q < NUM_QUERIES; ++q) {
                CATCH_REQUIRE(copy_results.offsets[q + 1] - copy_results.offsets[q] == K);
            }
            CATCH_REQUIRE(mean_recall(src_results, copy_results) >= min_recall);

            svs_search_results_free(&copy_results);
            svs_search_results_free(&src_results);
            svs_index_free(copy_index);
            svs_index_free(src_index);
            svs_index_builder_free(builder);
            svs_storage_free(dst_storage);
            svs_storage_free(src_storage);
            svs_algorithm_free(algorithm);
        };

        // fp32 <-> fp16. Half-precision is near-lossless here, so recall stays high.
        run_copy_case(simple_fp32(), simple_fp16(), 0.8);
        run_copy_case(simple_fp16(), simple_fp32(), 0.8);

        // simple <-> scalar. Int8 quantization perturbs distances, so require a
        // moderate recall rather than an exact match.
        run_copy_case(simple_fp32(), scalar_int8(), 0.5);
        run_copy_case(scalar_int8(), simple_fp32(), 0.5);

        // simple <-> lvq
        run_copy_case(simple_fp32(), lvq_4x8(), 0.8);
        run_copy_case(lvq_4x8(), simple_fp32(), 0.8);

        // simple <-> leanvec
        run_copy_case(simple_fp32(), leanvec_4x8(), 0.8);
        run_copy_case(leanvec_4x8(), simple_fp32(), 0.8);

        // TODO: compressed <-> compressed cases are not supported

        svs_search_params_free(search_params);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Convert Failures") {
        svs_error_h error = svs_error_create();

        // Build a valid source index to convert from.
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);
        CATCH_REQUIRE(svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        ));
        svs_index_h src_index = svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(src_index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // NULL builder is rejected.
        CATCH_REQUIRE(svs_index_convert(nullptr, src_index, error) == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        // NULL source index is rejected.
        CATCH_REQUIRE(svs_index_convert(builder, nullptr, error) == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        // A dynamic source index is rejected.
        svs_index_h dynamic_index =
            svs_index_build_dynamic(builder, data.data(), nullptr, NUM_VECTORS, 0, error);
        CATCH_REQUIRE(dynamic_index != nullptr);
        CATCH_REQUIRE(svs_index_convert(builder, dynamic_index, error) == nullptr);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        svs_index_free(dynamic_index);

        // Builds a destination builder that differs from the source in exactly one
        // aspect, so each conversion must fail with `expected_code`.
        auto expect_convert_failure = [&](svs_index_builder_h dst_builder,
                                          svs_error_code_t expected_code) {
            CATCH_REQUIRE(dst_builder != nullptr);
            CATCH_REQUIRE(svs_index_builder_set_threadpool(
                dst_builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
            ));
            CATCH_REQUIRE(svs_index_convert(dst_builder, src_index, error) == nullptr);
            CATCH_REQUIRE(svs_error_get_code(error) == expected_code);
            svs_index_builder_free(dst_builder);
        };

        // Distance metric mismatch is not supported.
        expect_convert_failure(
            svs_index_builder_create(
                SVS_DISTANCE_METRIC_COSINE, DIMENSION, algorithm, error
            ),
            SVS_ERROR_NOT_IMPLEMENTED
        );

        // Dimensionality mismatch is an invalid operation.
        expect_convert_failure(
            svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION * 2, algorithm, error
            ),
            SVS_ERROR_INVALID_OPERATION
        );

        // Build-parameter mismatch (graph degree) is not supported.
        svs_algorithm_h other_algorithm = svs_algorithm_create_vamana(32, 32, 50, error);
        CATCH_REQUIRE(other_algorithm != nullptr);
        expect_convert_failure(
            svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, other_algorithm, error
            ),
            SVS_ERROR_NOT_IMPLEMENTED
        );
        svs_algorithm_free(other_algorithm);

        svs_index_free(src_index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}

CATCH_TEST_CASE("C API Dynamic Index Conversion", "[c_api][index][dynamic][convert]") {
    const size_t NUM_VECTORS = 100;
    const size_t NUM_QUERIES = 5;
    const size_t DIMENSION = 32;
    const size_t K = 10;
    const size_t NUM_THREADS = 4;
    const size_t BLOCK_SIZE = 1024 * 1024; // 1 MB block size for testing

    std::vector<float> data;
    std::vector<float> queries;
    generate_test_data(data, NUM_VECTORS, DIMENSION);
    generate_test_data(queries, NUM_QUERIES, DIMENSION);

    CATCH_SECTION("Vamana Convert") {
        svs_error_h error = svs_error_create();

        svs_search_params_h search_params = svs_search_params_create_vamana(50, error);
        CATCH_REQUIRE(search_params != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        auto simple_fp32 = [&]() {
            svs_storage_h s = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error);
            CATCH_REQUIRE(s != nullptr);
            return s;
        };
        auto simple_fp16 = [&]() {
            svs_storage_h s = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);
            CATCH_REQUIRE(s != nullptr);
            return s;
        };
        auto scalar_int8 = [&]() {
            svs_storage_h s = svs_storage_create_sq(SVS_DATA_TYPE_INT8, error);
            CATCH_REQUIRE(s != nullptr);
            return s;
        };
        auto lvq_4x8 = [&]() {
            svs_storage_h s =
                svs_storage_create_lvq(SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, error);
            CATCH_REQUIRE(check_storage_support(s, error) == true);
            return s;
        };
        auto leanvec_4x8 = [&]() {
            svs_storage_h s = svs_storage_create_leanvec(
                DIMENSION / 2, SVS_DATA_TYPE_INT4, SVS_DATA_TYPE_INT8, error
            );
            CATCH_REQUIRE(check_storage_support(s, error) == true);
            return s;
        };

        auto search_index = [&](svs_index_h index, svs_search_results_t& out) {
            CATCH_REQUIRE(svs_index_search_topk(
                index, queries.data(), NUM_QUERIES, K, &out, search_params, nullptr, error
            ));
            CATCH_REQUIRE(svs_error_ok(error));
        };

        // Mean per-query recall@K of the copy's neighbors against the source's.
        auto mean_recall = [&](const svs_search_results_t& src,
                               const svs_search_results_t& copy) {
            double total = 0.0;
            for (size_t q = 0; q < NUM_QUERIES; ++q) {
                size_t matches = 0;
                for (size_t i = src.offsets[q]; i < src.offsets[q + 1]; ++i) {
                    for (size_t j = copy.offsets[q]; j < copy.offsets[q + 1]; ++j) {
                        if (src.indices[i] == copy.indices[j]) {
                            ++matches;
                            break;
                        }
                    }
                }
                size_t k = src.offsets[q + 1] - src.offsets[q];
                total += (k == 0) ? 1.0 : static_cast<double>(matches) / k;
            }
            return total / NUM_QUERIES;
        };

        // Builds a source index using `src_storage`, copies it into a fresh builder
        // configured with `dst_storage`, then checks the copy reproduces the source's
        // neighbors to within `min_recall`. Takes ownership of both storage handles.
        auto run_copy_case = [&](svs_storage_h src_storage,
                                 svs_storage_h dst_storage,
                                 double min_recall) {
            // Skip the test case if either the source or destination storage is not usable.
            // E.g. LVQ/Leanvec is not available on this platform
            if (!storage_usable(src_storage) || !storage_usable(dst_storage)) {
                svs_storage_free(dst_storage);
                svs_storage_free(src_storage);
                return;
            }
            svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
            CATCH_REQUIRE(algorithm != nullptr);

            svs_index_builder_h builder = svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
            );
            CATCH_REQUIRE(builder != nullptr);
            CATCH_REQUIRE(svs_index_builder_set_threadpool(
                builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
            ));
            CATCH_REQUIRE(svs_index_builder_set_storage(builder, src_storage, error));

            svs_index_h src_index = svs_index_build_dynamic(
                builder, data.data(), nullptr, NUM_VECTORS, BLOCK_SIZE, error
            );
            CATCH_REQUIRE(src_index != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            // Reuse the same builder for the copy, swapping in the destination storage.
            CATCH_REQUIRE(svs_index_builder_set_storage(builder, dst_storage, error));
            svs_index_h copy_index =
                svs_index_convert_dynamic(builder, src_index, BLOCK_SIZE, error);
            CATCH_REQUIRE(copy_index != nullptr);
            CATCH_REQUIRE(svs_error_ok(error));

            svs_search_results_t src_results = SVS_INIT_SEARCH_RESULTS();
            svs_search_results_t copy_results = SVS_INIT_SEARCH_RESULTS();
            search_index(src_index, src_results);
            search_index(copy_index, copy_results);

            CATCH_REQUIRE(copy_results.num_queries == src_results.num_queries);
            for (size_t q = 0; q < NUM_QUERIES; ++q) {
                CATCH_REQUIRE(copy_results.offsets[q + 1] - copy_results.offsets[q] == K);
            }
            CATCH_REQUIRE(mean_recall(src_results, copy_results) >= min_recall);

            svs_search_results_free(&copy_results);
            svs_search_results_free(&src_results);
            svs_index_free(copy_index);
            svs_index_free(src_index);
            svs_index_builder_free(builder);
            svs_storage_free(dst_storage);
            svs_storage_free(src_storage);
            svs_algorithm_free(algorithm);
        };

        // fp32 <-> fp16. Half-precision is near-lossless here, so recall stays high.
        run_copy_case(simple_fp32(), simple_fp16(), 0.8);
        run_copy_case(simple_fp16(), simple_fp32(), 0.8);

        // simple <-> scalar. Int8 quantization perturbs distances, so require a
        // moderate recall rather than an exact match.
        run_copy_case(simple_fp32(), scalar_int8(), 0.5);
        run_copy_case(scalar_int8(), simple_fp32(), 0.5);

        // simple <-> lvq
        run_copy_case(simple_fp32(), lvq_4x8(), 0.8);
        run_copy_case(lvq_4x8(), simple_fp32(), 0.8);

        // simple <-> leanvec
        run_copy_case(simple_fp32(), leanvec_4x8(), 0.8);
        run_copy_case(leanvec_4x8(), simple_fp32(), 0.8);

        // TODO: compressed <-> compressed cases are not supported

        svs_search_params_free(search_params);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Convert Failures") {
        svs_error_h error = svs_error_create();

        // Build a valid source index to convert from.
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);
        CATCH_REQUIRE(svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        ));
        svs_index_h src_index = svs_index_build_dynamic(
            builder, data.data(), nullptr, NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(src_index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        // NULL builder is rejected.
        CATCH_REQUIRE(
            svs_index_convert_dynamic(nullptr, src_index, BLOCK_SIZE, error) == nullptr
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        // NULL source index is rejected.
        CATCH_REQUIRE(
            svs_index_convert_dynamic(builder, nullptr, BLOCK_SIZE, error) == nullptr
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        // A static source index is rejected.
        svs_index_h static_index =
            svs_index_build(builder, data.data(), NUM_VECTORS, error);
        CATCH_REQUIRE(static_index != nullptr);
        CATCH_REQUIRE(
            svs_index_convert_dynamic(builder, static_index, BLOCK_SIZE, error) == nullptr
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        svs_index_free(static_index);

        // Builds a destination builder that differs from the source in exactly one
        // aspect, so each conversion must fail with `expected_code`.
        auto expect_convert_failure = [&](svs_index_builder_h dst_builder,
                                          svs_error_code_t expected_code) {
            CATCH_REQUIRE(dst_builder != nullptr);
            CATCH_REQUIRE(svs_index_builder_set_threadpool(
                dst_builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
            ));
            CATCH_REQUIRE(
                svs_index_convert_dynamic(dst_builder, src_index, BLOCK_SIZE, error) ==
                nullptr
            );
            CATCH_REQUIRE(svs_error_get_code(error) == expected_code);
            svs_index_builder_free(dst_builder);
        };

        // Distance metric mismatch is not supported.
        expect_convert_failure(
            svs_index_builder_create(
                SVS_DISTANCE_METRIC_COSINE, DIMENSION, algorithm, error
            ),
            SVS_ERROR_NOT_IMPLEMENTED
        );

        // Dimensionality mismatch is an invalid operation.
        expect_convert_failure(
            svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION * 2, algorithm, error
            ),
            SVS_ERROR_INVALID_OPERATION
        );

        // Build-parameter mismatch (graph degree) is not supported.
        svs_algorithm_h other_algorithm = svs_algorithm_create_vamana(32, 32, 50, error);
        CATCH_REQUIRE(other_algorithm != nullptr);
        expect_convert_failure(
            svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, other_algorithm, error
            ),
            SVS_ERROR_NOT_IMPLEMENTED
        );
        svs_algorithm_free(other_algorithm);

        svs_index_free(src_index);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    const size_t DELETE_STRIDE = 5;
    auto is_deleted = [&](size_t id) { return id % DELETE_STRIDE == 0; };

    // Builds a dynamic source index with `src_storage`, deletes every DELETE_STRIDE-th
    // vector (optionally consolidating), converts it to `dst_storage`, and checks the
    // copy keeps only the surviving IDs. Takes ownership of both storage handles.
    auto run_modified_copy_case = [&](svs_error_h error,
                                      svs_storage_h src_storage,
                                      svs_storage_h dst_storage,
                                      bool consolidate,
                                      double min_recall) {
        CATCH_REQUIRE(src_storage != nullptr);
        CATCH_REQUIRE(dst_storage != nullptr);

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);
        svs_search_params_h search_params = svs_search_params_create_vamana(50, error);
        CATCH_REQUIRE(search_params != nullptr);

        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);
        CATCH_REQUIRE(svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        ));
        CATCH_REQUIRE(svs_index_builder_set_storage(builder, src_storage, error));

        svs_index_h src_index = svs_index_build_dynamic(
            builder, data.data(), nullptr, NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(src_index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        std::vector<size_t> ids_to_delete;
        for (size_t id = 0; id < NUM_VECTORS; ++id) {
            if (is_deleted(id)) {
                ids_to_delete.push_back(id);
            }
        }
        size_t deleted_count = 0;
        CATCH_REQUIRE(svs_index_dynamic_delete_points(
            src_index, ids_to_delete.data(), ids_to_delete.size(), &deleted_count, error
        ));
        CATCH_REQUIRE(deleted_count == ids_to_delete.size());

        if (consolidate) {
            CATCH_REQUIRE(svs_index_dynamic_consolidate(src_index, error));
            CATCH_REQUIRE(svs_error_ok(error));
        }

        CATCH_REQUIRE(svs_index_builder_set_storage(builder, dst_storage, error));
        svs_index_h copy_index =
            svs_index_convert_dynamic(builder, src_index, BLOCK_SIZE, error);
        CATCH_REQUIRE(copy_index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        const size_t expected_size = NUM_VECTORS - ids_to_delete.size();
        size_t src_size = 0;
        size_t copy_size = 0;
        CATCH_REQUIRE(svs_index_get_size(src_index, &src_size, error));
        CATCH_REQUIRE(svs_index_get_size(copy_index, &copy_size, error));
        CATCH_REQUIRE(src_size == expected_size);
        CATCH_REQUIRE(copy_size == expected_size);

        for (size_t id = 0; id < NUM_VECTORS; ++id) {
            bool has_id = false;
            CATCH_REQUIRE(svs_index_dynamic_has_id(copy_index, id, &has_id, error));
            CATCH_REQUIRE(has_id == !is_deleted(id));
        }

        svs_search_results_t src_results = SVS_INIT_SEARCH_RESULTS();
        svs_search_results_t copy_results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            src_index,
            queries.data(),
            NUM_QUERIES,
            K,
            &src_results,
            search_params,
            nullptr,
            error
        ));
        CATCH_REQUIRE(svs_index_search_topk(
            copy_index,
            queries.data(),
            NUM_QUERIES,
            K,
            &copy_results,
            search_params,
            nullptr,
            error
        ));
        CATCH_REQUIRE(svs_error_ok(error));

        CATCH_REQUIRE(copy_results.num_queries == NUM_QUERIES);
        double total_recall = 0.0;
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            CATCH_REQUIRE(copy_results.offsets[q + 1] - copy_results.offsets[q] == K);
            size_t matches = 0;
            for (size_t j = copy_results.offsets[q]; j < copy_results.offsets[q + 1]; ++j) {
                CATCH_REQUIRE(copy_results.indices[j] < NUM_VECTORS);
                CATCH_REQUIRE_FALSE(is_deleted(copy_results.indices[j]));
                for (size_t i = src_results.offsets[q]; i < src_results.offsets[q + 1];
                     ++i) {
                    if (src_results.indices[i] == copy_results.indices[j]) {
                        ++matches;
                        break;
                    }
                }
            }
            total_recall += static_cast<double>(matches) / K;
        }
        CATCH_REQUIRE(total_recall / NUM_QUERIES >= min_recall);

        svs_search_results_free(&copy_results);
        svs_search_results_free(&src_results);
        svs_index_free(copy_index);
        svs_index_free(src_index);
        svs_index_builder_free(builder);
        svs_search_params_free(search_params);
        svs_algorithm_free(algorithm);
        svs_storage_free(dst_storage);
        svs_storage_free(src_storage);
    };

    CATCH_SECTION("Vamana Convert with Deleted Vectors") {
        svs_error_h error = svs_error_create();
        run_modified_copy_case(
            error,
            svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error),
            svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error),
            false,
            0.8
        );
        run_modified_copy_case(
            error,
            svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error),
            svs_storage_create_sq(SVS_DATA_TYPE_INT8, error),
            false,
            0.5
        );
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Convert after Consolidation") {
        svs_error_h error = svs_error_create();
        run_modified_copy_case(
            error,
            svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error),
            svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error),
            true,
            0.8
        );
        run_modified_copy_case(
            error,
            svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error),
            svs_storage_create_sq(SVS_DATA_TYPE_INT8, error),
            true,
            0.5
        );
        svs_error_free(error);
    }

    // Exercises the mutation paths that use the build parameters carried over by the
    // conversion (pruning during insertion and consolidation).
    CATCH_SECTION("Modify Converted Index") {
        svs_error_h error = svs_error_create();
        const size_t NUM_ADDED = 20;

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(16, 32, 50, error);
        CATCH_REQUIRE(algorithm != nullptr);
        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, DIMENSION, algorithm, error
        );
        CATCH_REQUIRE(builder != nullptr);
        CATCH_REQUIRE(svs_index_builder_set_threadpool(
            builder, SVS_THREADPOOL_KIND_NATIVE, NUM_THREADS, error
        ));

        svs_index_h src_index = svs_index_build_dynamic(
            builder, data.data(), nullptr, NUM_VECTORS, BLOCK_SIZE, error
        );
        CATCH_REQUIRE(src_index != nullptr);

        std::vector<size_t> initial_deletes;
        for (size_t id = 0; id < NUM_VECTORS; id += DELETE_STRIDE) {
            initial_deletes.push_back(id);
        }
        CATCH_REQUIRE(svs_index_dynamic_delete_points(
            src_index, initial_deletes.data(), initial_deletes.size(), nullptr, error
        ));

        svs_storage_h dst_storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);
        CATCH_REQUIRE(dst_storage != nullptr);
        CATCH_REQUIRE(svs_index_builder_set_storage(builder, dst_storage, error));
        svs_index_h copy_index =
            svs_index_convert_dynamic(builder, src_index, BLOCK_SIZE, error);
        CATCH_REQUIRE(copy_index != nullptr);
        CATCH_REQUIRE(svs_error_ok(error));

        auto expect_size = [&](svs_index_h index, size_t expected) {
            size_t size = 0;
            CATCH_REQUIRE(svs_index_get_size(index, &size, error));
            CATCH_REQUIRE(size == expected);
        };
        auto expect_has_id = [&](size_t id, bool expected) {
            bool has_id = !expected;
            CATCH_REQUIRE(svs_index_dynamic_has_id(copy_index, id, &has_id, error));
            CATCH_REQUIRE(has_id == expected);
        };

        size_t expected_size = NUM_VECTORS - initial_deletes.size();
        expect_size(copy_index, expected_size);

        // Add new points with fresh IDs.
        std::vector<float> new_data;
        generate_test_data(new_data, NUM_ADDED, DIMENSION);
        // Shift away from the original data, which the deterministic generator repeats.
        for (auto& v : new_data) {
            v += 10.0f;
        }
        std::vector<size_t> new_ids(NUM_ADDED);
        for (size_t i = 0; i < NUM_ADDED; ++i) {
            new_ids[i] = NUM_VECTORS + i;
        }
        size_t added_count = 0;
        CATCH_REQUIRE(svs_index_dynamic_add_points(
            copy_index, new_data.data(), new_ids.data(), NUM_ADDED, &added_count, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(added_count == NUM_ADDED);
        expected_size += NUM_ADDED;
        expect_size(copy_index, expected_size);
        for (auto id : new_ids) {
            expect_has_id(id, true);
        }

        // Each added vector must be retrievable as its own nearest neighbor.
        svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();
        CATCH_REQUIRE(svs_index_search_topk(
            copy_index, new_data.data(), NUM_ADDED, 1, &results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        size_t self_hits = 0;
        for (size_t q = 0; q < NUM_ADDED; ++q) {
            CATCH_REQUIRE(results.offsets[q + 1] - results.offsets[q] == 1);
            self_hits += results.indices[results.offsets[q]] == new_ids[q] ? 1 : 0;
        }
        CATCH_REQUIRE(self_hits >= NUM_ADDED * 9 / 10);

        // Delete a mix of original and newly added IDs, then consolidate and compact.
        std::vector<size_t> more_deletes = {1, 2, 3, new_ids[0], new_ids[1]};
        size_t deleted_count = 0;
        CATCH_REQUIRE(svs_index_dynamic_delete_points(
            copy_index, more_deletes.data(), more_deletes.size(), &deleted_count, error
        ));
        CATCH_REQUIRE(deleted_count == more_deletes.size());
        expected_size -= more_deletes.size();
        expect_size(copy_index, expected_size);

        CATCH_REQUIRE(svs_index_dynamic_consolidate(copy_index, error));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(svs_index_dynamic_compact(copy_index, 0, error));
        CATCH_REQUIRE(svs_error_ok(error));
        expect_size(copy_index, expected_size);

        auto is_live = [&](size_t id) {
            if (std::find(more_deletes.begin(), more_deletes.end(), id) !=
                more_deletes.end()) {
                return false;
            }
            return id >= NUM_VECTORS || !is_deleted(id);
        };
        for (size_t id = 0; id < NUM_VECTORS + NUM_ADDED; ++id) {
            expect_has_id(id, is_live(id));
        }

        CATCH_REQUIRE(svs_index_search_topk(
            copy_index, queries.data(), NUM_QUERIES, K, &results, nullptr, nullptr, error
        ));
        CATCH_REQUIRE(svs_error_ok(error));
        for (size_t q = 0; q < NUM_QUERIES; ++q) {
            CATCH_REQUIRE(results.offsets[q + 1] - results.offsets[q] == K);
            for (size_t j = results.offsets[q]; j < results.offsets[q + 1]; ++j) {
                CATCH_REQUIRE(is_live(results.indices[j]));
            }
        }

        // The source index must be unaffected by mutations of the copy.
        expect_size(src_index, NUM_VECTORS - initial_deletes.size());

        svs_search_results_free(&results);
        svs_index_free(copy_index);
        svs_index_free(src_index);
        svs_storage_free(dst_storage);
        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}
