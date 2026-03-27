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

CATCH_TEST_CASE("C API Error Handling", "[c_api][error]") {
    CATCH_SECTION("Error Creation and Cleanup") {
        svs_error_h error = svs_error_create();
        CATCH_REQUIRE(error != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_OK);
        CATCH_REQUIRE(svs_error_get_message(error) != nullptr);
        svs_error_free(error);
    }

    CATCH_SECTION("Error State After API Call") {
        svs_error_h error = svs_error_create();
        CATCH_REQUIRE(error != nullptr);

        // Create a valid algorithm - should not set error
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, error);
        CATCH_REQUIRE(algorithm != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_OK);

        svs_algorithm_free(algorithm);
        svs_error_free(error);
    }

    CATCH_SECTION("Error State After Invalid API Call") {
        svs_error_h error = svs_error_create();
        CATCH_REQUIRE(error != nullptr);

        // Try to create algorithm with invalid parameters (e.g., 0 graph degree)
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(0, 0, 0, error);
        CATCH_REQUIRE(algorithm == nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == false);
        CATCH_REQUIRE(svs_error_get_code(error) != SVS_OK);
        CATCH_REQUIRE(svs_error_get_message(error) != nullptr);

        svs_error_free(error);
    }

    CATCH_SECTION("Multiple Error Handles") {
        svs_error_h error1 = svs_error_create();
        svs_error_h error2 = svs_error_create();

        CATCH_REQUIRE(error1 != nullptr);
        CATCH_REQUIRE(error2 != nullptr);
        CATCH_REQUIRE(error1 != error2);

        svs_error_free(error1);
        svs_error_free(error2);
    }

    CATCH_SECTION("NULL Error Handle") {
        // API calls should work with NULL error handle
        svs_algorithm_h algorithm = svs_algorithm_create_vamana(64, 128, 100, nullptr);
        CATCH_REQUIRE(algorithm != nullptr);
        svs_algorithm_free(algorithm);
    }
}
