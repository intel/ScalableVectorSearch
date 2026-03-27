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
        // Behavior depends on implementation - either nullptr or valid handle
        if (params != nullptr) {
            svs_search_params_free(params);
        }

        svs_error_free(error);
    }
}
