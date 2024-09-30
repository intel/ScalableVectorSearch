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
// svstest
#include "tests/utils/test_dataset.h"

// svsbenchmark
#include "svs-benchmark/vamana/test.h"

// svs
#include "svs/core/distance.h"
#include "svs/lib/timing.h"
#include "svs/third-party/toml.h"

// stl
#include <filesystem>
#include <string_view>
#include <vector>

namespace test_dataset::vamana {
namespace {
std::filesystem::path reference_path() {
    return test_dataset::reference_directory() / "vamana_reference.toml";
}
} // namespace

const toml::table& parse_expected() {
    // Make the expected results a magic static variable.
    // This shaves off a bit of run time as we only need to parse the toml file once.
    static toml::table expected = toml::parse_file(reference_path().native());
    return expected;
}
} // namespace test_dataset::vamana
