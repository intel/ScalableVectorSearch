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

CATCH_TEST_CASE("DirectoryArchiver", "[lib][files]") {
    namespace fs = std::filesystem;
    using namespace svs::lib;

    auto tempdir = svs_test::prepare_temp_directory_v2();
    auto srcdir = tempdir / "src";
    auto dstdir = tempdir / "dst";

    // Create a source directory with some files in it.
    fs::create_directories(srcdir);
    std::ofstream(srcdir / "file1.txt") << "Hello, World!" << std::endl;
    fs::create_directories(srcdir / "subdir");
    std::ofstream(srcdir / "subdir/file2.txt") << "This is a test." << std::endl;

    CATCH_SECTION("Pack and Unpack") {
        // Pack the directory.
        std::stringstream ss;
        auto bytes_written = DirectoryArchiver::pack(srcdir, ss);
        CATCH_REQUIRE(bytes_written > 0);

        // Unpack the directory.
        fs::create_directories(dstdir);
        auto bytes_read = DirectoryArchiver::unpack(ss, dstdir);
        CATCH_REQUIRE(bytes_read == bytes_written);

        // Check that the files exist in the destination directory.
        CATCH_REQUIRE(fs::exists(dstdir / "file1.txt"));
        CATCH_REQUIRE(fs::exists(dstdir / "subdir/file2.txt"));

        // Check that the contents are correct.
        std::ifstream in1(dstdir / "file1.txt");
        std::string line1;
        std::getline(in1, line1);
        CATCH_REQUIRE(line1 == "Hello, World!");

        std::ifstream in2(dstdir / "subdir/file2.txt");
        std::string line2;
        std::getline(in2, line2);
        CATCH_REQUIRE(line2 == "This is a test.");
    }
}
