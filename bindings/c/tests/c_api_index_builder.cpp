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
#include <vector>

namespace {

// Helper function to generate random test data
void generate_test_data(std::vector<float>& data, size_t num_vectors, size_t dimension) {
    data.resize(num_vectors * dimension);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }
}

// Sequential threadpool implementation for testing
size_t sequential_tp_size(void* /*self*/) { return 1; }

void sequential_tp_parallel_for(
    void* /*self*/, void (*func)(void*, size_t), void* svs_param, size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        func(svs_param, i);
    }
}

} // namespace

CATCH_TEST_CASE("C API Index Builder", "[c_api][index_builder]") {
    CATCH_SECTION("Index Builder Creation") {
        svs_error_h error = svs_error_create();
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder =
            svs_index_builder_create(SVS_DISTANCE_METRIC_EUCLIDEAN, 128, algorithm, error);
        CATCH_REQUIRE(builder != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Builder with Different Metrics") {
        svs_error_h error = svs_error_create();
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        // Euclidean
        svs_index_builder_h builder1 =
            svs_index_builder_create(SVS_DISTANCE_METRIC_EUCLIDEAN, 128, algorithm, error);
        CATCH_REQUIRE(builder1 != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        // Cosine
        svs_index_builder_h builder2 =
            svs_index_builder_create(SVS_DISTANCE_METRIC_COSINE, 128, algorithm, error);
        CATCH_REQUIRE(builder2 != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        // Dot Product
        svs_index_builder_h builder3 = svs_index_builder_create(
            SVS_DISTANCE_METRIC_DOT_PRODUCT, 128, algorithm, error
        );
        CATCH_REQUIRE(builder3 != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_index_builder_free(builder1);
        svs_index_builder_free(builder2);
        svs_index_builder_free(builder3);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Builder Set Storage") {
        svs_error_h error = svs_error_create();
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder =
            svs_index_builder_create(SVS_DISTANCE_METRIC_EUCLIDEAN, 128, algorithm, error);
        CATCH_REQUIRE(builder != nullptr);

        svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error);
        CATCH_REQUIRE(storage != nullptr);

        bool success = svs_index_builder_set_storage(builder, storage, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_index_builder_free(builder);
        svs_storage_free(storage);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Builder Set Threadpool Native") {
        svs_error_h error = svs_error_create();
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder =
            svs_index_builder_create(SVS_DISTANCE_METRIC_EUCLIDEAN, 128, algorithm, error);
        CATCH_REQUIRE(builder != nullptr);

        bool success =
            svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_NATIVE, 2, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Builder Set Threadpool OMP") {
        svs_error_h error = svs_error_create();
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder =
            svs_index_builder_create(SVS_DISTANCE_METRIC_EUCLIDEAN, 128, algorithm, error);
        CATCH_REQUIRE(builder != nullptr);

        bool success =
            svs_index_builder_set_threadpool(builder, SVS_THREADPOOL_KIND_OMP, 2, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Builder Set Custom Threadpool") {
        svs_error_h error = svs_error_create();
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder =
            svs_index_builder_create(SVS_DISTANCE_METRIC_EUCLIDEAN, 128, algorithm, error);
        CATCH_REQUIRE(builder != nullptr);

        struct svs_threadpool_interface custom_pool = {
            {sequential_tp_size, sequential_tp_parallel_for}, nullptr};

        bool success =
            svs_index_builder_set_threadpool_custom(builder, &custom_pool, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Builder with NULL Error") {
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, nullptr);
        CATCH_REQUIRE(algorithm != nullptr);

        svs_index_builder_h builder = svs_index_builder_create(
            SVS_DISTANCE_METRIC_EUCLIDEAN, 128, algorithm, nullptr
        );
        CATCH_REQUIRE(builder != nullptr);

        svs_index_builder_free(builder);
        svs_algorithm_free(algorithm);
    }

    CATCH_SECTION("Index Builder with Various Dimensions") {
        svs_error_h error = svs_error_create();
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        size_t dimensions[] = {32, 64, 128, 256, 384, 512, 768, 1024};
        for (size_t i = 0; i < sizeof(dimensions) / sizeof(dimensions[0]); ++i) {
            svs_index_builder_h builder = svs_index_builder_create(
                SVS_DISTANCE_METRIC_EUCLIDEAN, dimensions[i], algorithm, error
            );
            CATCH_REQUIRE(builder != nullptr);
            CATCH_REQUIRE(svs_error_ok(error) == true);
            svs_index_builder_free(builder);
        }

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Index Builder Invalid Parameters") {
        svs_error_h error = svs_error_create();
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        // Try to create with 0 dimension
        svs_index_builder_h builder =
            svs_index_builder_create(SVS_DISTANCE_METRIC_EUCLIDEAN, 0, algorithm, error);
        // Behavior depends on implementation
        if (builder != nullptr) {
            svs_index_builder_free(builder);
        }

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }
}
