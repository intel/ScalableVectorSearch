/*
 * Copyright 2024 Intel Corporation
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
