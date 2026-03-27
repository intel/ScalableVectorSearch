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
#include <cmath>

CATCH_TEST_CASE("C API Vamana Algorithm", "[c_api][algorithm][vamana]") {
    CATCH_SECTION("Vamana Algorithm Creation") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Algorithm Get Graph Degree") {
        svs_error_h error = svs_error_create();
        size_t expected_degree = 64;

        svs_algorithm_h algorithm =
            svs_algorithm_create_vamana(expected_degree, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        size_t actual_degree = 0;
        bool success =
            svs_algorithm_vamana_get_graph_degree(algorithm, &actual_degree, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);
        CATCH_REQUIRE(actual_degree == expected_degree);

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Algorithm Set Graph Degree") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        size_t new_degree = 96;
        bool success = svs_algorithm_vamana_set_graph_degree(algorithm, new_degree, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        size_t actual_degree = 0;
        success = svs_algorithm_vamana_get_graph_degree(algorithm, &actual_degree, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(actual_degree == new_degree);

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Algorithm Get Build Window Size") {
        svs_error_h error = svs_error_create();
        size_t expected_window = 128;

        svs_algorithm_h algorithm =
            svs_algorithm_create_vamana(64, expected_window, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        size_t actual_window = 0;
        bool success =
            svs_algorithm_vamana_get_build_window_size(algorithm, &actual_window, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);
        CATCH_REQUIRE(actual_window == expected_window);

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Algorithm Set Build Window Size") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        size_t new_window = 256;
        bool success =
            svs_algorithm_vamana_set_build_window_size(algorithm, new_window, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        size_t actual_window = 0;
        success =
            svs_algorithm_vamana_get_build_window_size(algorithm, &actual_window, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(actual_window == new_window);

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Algorithm Get/Set Alpha") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        // Get default alpha
        float alpha = 0.0f;
        bool success = svs_algorithm_vamana_get_alpha(algorithm, &alpha, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);
        CATCH_REQUIRE(alpha > 0.0f);

        // Set new alpha
        float new_alpha = 1.5f;
        success = svs_algorithm_vamana_set_alpha(algorithm, new_alpha, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        // Verify the change
        float actual_alpha = 0.0f;
        success = svs_algorithm_vamana_get_alpha(algorithm, &actual_alpha, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(std::abs(actual_alpha - new_alpha) < 1e-6f);

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Algorithm Get/Set Search History") {
        svs_error_h error = svs_error_create();

        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);

        // Get default search history setting
        bool use_history = false;
        bool success =
            svs_algorithm_vamana_get_use_search_history(algorithm, &use_history, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        // Set search history
        bool new_value = !use_history;
        success = svs_algorithm_vamana_set_use_search_history(algorithm, new_value, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        // Verify the change
        bool actual_value = false;
        success =
            svs_algorithm_vamana_get_use_search_history(algorithm, &actual_value, error);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(actual_value == new_value);

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Algorithm with NULL Error") {
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, nullptr);
        CATCH_REQUIRE(algorithm != nullptr);

        size_t degree = 0;
        bool success = svs_algorithm_vamana_get_graph_degree(algorithm, &degree, nullptr);
        CATCH_REQUIRE(success == true);
        CATCH_REQUIRE(degree == 64);

        svs_algorithm_free(algorithm);
    }

    CATCH_SECTION("Vamana Algorithm Invalid Parameters") {
        svs_error_h error = svs_error_create();

        // Try to create with invalid parameters
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(0, 0, 0, error);
        CATCH_REQUIRE(algorithm == nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == false);

        svs_error_free(error);
    }
}
