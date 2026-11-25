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

CATCH_TEST_CASE("C API Search Parameters", "[c_api][search_params]") {
    CATCH_SECTION("Vamana Search Parameters Creation") {
        svs_error_h error = svs_error_create();

        size_t search_window_size = 100;
        svs_search_params_h params =
            svs_search_params_create_vamana(search_window_size, error);
        CATCH_REQUIRE(params != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_search_params_free(params);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Search Parameters Various Sizes") {
        svs_error_h error = svs_error_create();

        size_t sizes[] = {10, 50, 100, 200, 500, 1000};
        for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
            svs_search_params_h params = svs_search_params_create_vamana(sizes[i], error);
            CATCH_REQUIRE(params != nullptr);
            CATCH_REQUIRE(svs_error_ok(error) == true);
            svs_search_params_free(params);
        }

        svs_error_free(error);
    }

    CATCH_SECTION("Search Parameters with NULL Error") {
        svs_search_params_h params = svs_search_params_create_vamana(100, nullptr);
        CATCH_REQUIRE(params != nullptr);

        svs_search_params_free(params);
    }

    CATCH_SECTION("Multiple Search Parameters Handles") {
        svs_error_h error = svs_error_create();

        svs_search_params_h params1 = svs_search_params_create_vamana(50, error);
        svs_search_params_h params2 = svs_search_params_create_vamana(100, error);
        svs_search_params_h params3 = svs_search_params_create_vamana(200, error);

        CATCH_REQUIRE(params1 != nullptr);
        CATCH_REQUIRE(params2 != nullptr);
        CATCH_REQUIRE(params3 != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_search_params_free(params1);
        svs_search_params_free(params2);
        svs_search_params_free(params3);
        svs_error_free(error);
    }

    CATCH_SECTION("Search Parameters with Invalid Size") {
        svs_error_h error = svs_error_create();

        // Try to create with size 0
        svs_search_params_h params = svs_search_params_create_vamana(0, error);
        CATCH_REQUIRE(params == nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == false);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        const char* msg = svs_error_get_message(error);
        CATCH_REQUIRE(msg != nullptr);
        CATCH_REQUIRE(msg[0] != '\0');

        svs_error_free(error);
    }
    CATCH_SECTION("Vamana Search Parameters Buffer Capacity") {
        svs_error_h error = svs_error_create();
        svs_search_params_h params = svs_search_params_create_vamana(100, error);
        CATCH_REQUIRE(params != nullptr);

        size_t window_size = 0;
        size_t buffer_capacity = 0;
        CATCH_REQUIRE(svs_search_params_vamana_get_window_size(params, &window_size, error)
        );
        CATCH_REQUIRE(window_size == 100);
        CATCH_REQUIRE(
            svs_search_params_vamana_get_buffer_capacity(params, &buffer_capacity, error)
        );
        CATCH_REQUIRE(buffer_capacity == 100);

        CATCH_REQUIRE(svs_search_params_vamana_set_buffer_capacity(params, 250, error));
        CATCH_REQUIRE(svs_error_ok(error));
        CATCH_REQUIRE(
            svs_search_params_vamana_get_buffer_capacity(params, &buffer_capacity, error)
        );
        CATCH_REQUIRE(buffer_capacity == 250);
        CATCH_REQUIRE(svs_search_params_vamana_get_window_size(params, &window_size, error)
        );
        CATCH_REQUIRE(window_size == 100);

        // Equal to the window size is the lower bound and is accepted.
        CATCH_REQUIRE(svs_search_params_vamana_set_buffer_capacity(params, 100, error));
        CATCH_REQUIRE(
            svs_search_params_vamana_get_buffer_capacity(params, &buffer_capacity, error)
        );
        CATCH_REQUIRE(buffer_capacity == 100);

        // Below the window size is rejected and leaves the capacity unchanged.
        CATCH_REQUIRE_FALSE(svs_search_params_vamana_set_buffer_capacity(params, 99, error)
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE(
            svs_search_params_vamana_get_buffer_capacity(params, &buffer_capacity, nullptr)
        );
        CATCH_REQUIRE(buffer_capacity == 100);

        svs_search_params_free(params);
        svs_error_free(error);
    }

    CATCH_SECTION("Vamana Search Parameters Accessors with NULL Arguments") {
        svs_error_h error = svs_error_create();
        size_t value = 0;

        CATCH_REQUIRE_FALSE(svs_search_params_vamana_get_window_size(nullptr, &value, error)
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE_FALSE(
            svs_search_params_vamana_get_buffer_capacity(nullptr, &value, error)
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE_FALSE(svs_search_params_vamana_set_buffer_capacity(nullptr, 10, error)
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        svs_search_params_h params = svs_search_params_create_vamana(10, error);
        CATCH_REQUIRE(params != nullptr);
        CATCH_REQUIRE_FALSE(svs_search_params_vamana_get_window_size(params, nullptr, error)
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);
        CATCH_REQUIRE_FALSE(
            svs_search_params_vamana_get_buffer_capacity(params, nullptr, error)
        );
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        svs_search_params_free(params);
        svs_error_free(error);
    }
}
