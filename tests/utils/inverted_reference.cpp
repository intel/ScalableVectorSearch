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

// svstest
#include "tests/utils/test_dataset.h"

// svsbenchmark
#include "svs-benchmark/inverted/memory/test.h"

// svs
#include "svs/core/distance.h"
#include "svs/lib/timing.h"
#include "svs/third-party/toml.h"

// stl
#include <filesystem>
#include <string_view>
#include <vector>

namespace test_dataset::inverted {

namespace {
std::filesystem::path reference_path() {
    return test_dataset::reference_directory() / "inverted_reference.toml";
}
} // namespace

const toml::table& parse_expected() {
    // Make the expected results a magic static variable.
    // This shaves off a bit of run time as we only need to parse the toml file once.
    static toml::table expected = toml::parse_file(reference_path().native());
    return expected;
}
} // namespace test_dataset::inverted
