/*
 * Copyright 2023 Intel Corporation
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

// Header under test.
#include "svs/lib/file.h"

// test utils
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <filesystem>

CATCH_TEST_CASE("Filesystem Handling", "[lib][files]") {
    CATCH_SECTION("Errors") {
        // Check that if we try to open a path, we get an error.
        auto path = test_dataset::dataset_directory();
        CATCH_REQUIRE(std::filesystem::is_directory(path));
        CATCH_REQUIRE_THROWS_AS(
            svs::lib::check_file(path, std::ios_base::in), svs::ANNException
        );

        path = test_dataset::data_svs_file();
        // Runs without error.
        CATCH_REQUIRE(svs::lib::check_file(path, std::ios_base::in));
        // non-existent path
        CATCH_REQUIRE_THROWS_AS(
            svs::lib::check_file("hopefully/not/a/path", std::ios_base::in),
            svs::ANNException
        );

        // Writing should fail for non-existent directories.
        path = test_dataset::dataset_directory();
        CATCH_REQUIRE(svs::lib::check_file(path, std::ios_base::out));

        CATCH_REQUIRE_THROWS_AS(
            svs::lib::check_file("hopefully/not/a/path", std::ios_base::out),
            svs::ANNException
        );
    }
}
