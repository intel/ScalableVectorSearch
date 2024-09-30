/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
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
