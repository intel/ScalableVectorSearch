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

CATCH_TEST_CASE("C API Storage", "[c_api][storage]") {
    CATCH_SECTION("Simple Storage Float32") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error);
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("Simple Storage Float16") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("Simple Storage INT8") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_INT8, error);
        CATCH_REQUIRE(storage == nullptr);
        CATCH_REQUIRE_FALSE(svs_error_ok(error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("Simple Storage UINT8") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_UINT8, error);
        CATCH_REQUIRE(storage == nullptr);
        CATCH_REQUIRE_FALSE(svs_error_ok(error));
        CATCH_REQUIRE(svs_error_get_code(error) == SVS_ERROR_INVALID_ARGUMENT);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("LeanVec Storage") {
        svs_error_h error = svs_error_create();

        size_t leanvec_dims = 64;
        svs_storage_h storage = svs_storage_create_leanvec(
            leanvec_dims, SVS_DATA_TYPE_UINT8, SVS_DATA_TYPE_UINT8, error
        );
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("LeanVec Storage UINT4") {
        svs_error_h error = svs_error_create();

        size_t leanvec_dims = 64;
        svs_storage_h storage = svs_storage_create_leanvec(
            leanvec_dims, SVS_DATA_TYPE_UINT4, SVS_DATA_TYPE_UINT4, error
        );
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("LVQ Storage UINT4") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage =
            svs_storage_create_lvq(SVS_DATA_TYPE_UINT4, SVS_DATA_TYPE_VOID, error);
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("LVQ Storage UINT8") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage =
            svs_storage_create_lvq(SVS_DATA_TYPE_UINT8, SVS_DATA_TYPE_VOID, error);
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("LVQ Storage with Residual") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage =
            svs_storage_create_lvq(SVS_DATA_TYPE_UINT4, SVS_DATA_TYPE_UINT8, error);
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("Scalar Quantization Storage UINT8") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage = svs_storage_create_sq(SVS_DATA_TYPE_UINT8, error);
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("Scalar Quantization Storage INT8") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage = svs_storage_create_sq(SVS_DATA_TYPE_INT8, error);
        CATCH_REQUIRE(storage != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage);
        svs_error_free(error);
    }

    CATCH_SECTION("Storage with NULL Error") {
        svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, nullptr);
        CATCH_REQUIRE(storage != nullptr);

        svs_storage_free(storage);
    }

    CATCH_SECTION("Multiple Storage Handles") {
        svs_error_h error = svs_error_create();

        svs_storage_h storage1 = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, error);
        svs_storage_h storage2 = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT16, error);
        svs_storage_h storage3 =
            svs_storage_create_leanvec(64, SVS_DATA_TYPE_UINT8, SVS_DATA_TYPE_UINT8, error);

        CATCH_REQUIRE(storage1 != nullptr);
        CATCH_REQUIRE(storage2 != nullptr);
        CATCH_REQUIRE(storage3 != nullptr);
        CATCH_REQUIRE(svs_error_ok(error) == true);

        svs_storage_free(storage1);
        svs_storage_free(storage2);
        svs_storage_free(storage3);
        svs_error_free(error);
    }
}
