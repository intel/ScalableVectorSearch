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

#pragma once

// svs
#include "svs/lib/uuid.h"

// tests
#include "tests/utils/utils.h"

// stl
#include <filesystem>
#include <vector>

namespace test_schemas {

// Directory to the test schemas file
std::filesystem::path test_schema_directory();

// Full filepath to the V1 file
std::filesystem::path test_v1_file();

// Full filepath to the Vtest file
std::filesystem::path test_vtest_file();

// Full filepath to the Database Prototype file.
std::filesystem::path test_database_file();

// Expected UUID for vtest
constexpr inline svs::lib::UUID vtest_uuid() {
    constexpr auto uuid = svs::lib::UUID("99ad8b56-4cd4-4335-a669-e4c7cc721e2d");
    return uuid;
}

// Expected UUID for v1
constexpr inline svs::lib::UUID v1_uuid() {
    constexpr auto uuid = svs::lib::UUID("de3fe016-93c0-4c10-bef4-a5fb34791fe2");
    return uuid;
}

// Expected UUID for database
constexpr inline svs::lib::UUID database_uuid() {
    constexpr auto uuid = svs::lib::UUID("1ece4578-b6ad-4813-a9cb-22413860d64d");
    return uuid;
}

// Expected contents for vtest
std::vector<std::vector<float>> vtest_contents();

// Expected contents for v1
std::vector<std::vector<float>> v1_contents();

// Expected contents for database.
constexpr uint64_t database_kind() { return 0xc2fcdee1f5093720; }

constexpr svs::lib::Version database_version() { return svs::lib::Version(10, 20, 30); }

} // namespace test_schemas
