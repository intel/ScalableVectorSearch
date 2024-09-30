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
// corresponding header
#include "tests/utils/schemas.h"

// svs
#include "svs/lib/uuid.h"

// tests
#include "tests/utils/utils.h"

// stl
#include <filesystem>
#include <vector>

namespace test_schemas {
// Directory to the test schemas file
std::filesystem::path test_schema_directory() {
    return svs_test::data_directory() / "schemas";
}

// Full filepath to the V1 file
std::filesystem::path test_v1_file() { return test_schema_directory() / "test_v1.svs"; }

std::filesystem::path test_database_file() {
    return test_schema_directory() / "test_database.svs.db";
}

// Full filepath to the Vtest file
std::filesystem::path test_vtest_file() {
    return test_schema_directory() / "test_vtest.svs";
}

// Expected contents for vtest
std::vector<std::vector<float>> vtest_contents() {
    return std::vector<std::vector<float>>{
        {1.0, 2.0, 3.0, 4.0, 5.0}, {6.0, 7.0, 8.0, 9.0, 10.0}};
}

// Expected contents for v1
std::vector<std::vector<float>> v1_contents() {
    return std::vector<std::vector<float>>{
        {101.0, 102.0, 103.0, 104.0, 105.0}, {106.0, 107.0, 108.0, 109.0, 110.0}};
}
} // namespace test_schemas
