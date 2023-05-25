/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
