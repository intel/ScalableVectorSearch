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
