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

#pragma once

// svsbenchmark
#include "svs-benchmark/datasets.h"
#include "svs-benchmark/vamana/test.h"

// svs
#include "svs/core/distance.h"

// stl
#include <optional>
#include <string_view>
#include <vector>

namespace test_dataset::vamana {

// Implemented in CPP file.
const toml::table& parse_expected();

template <svsbenchmark::ValidDatasetSource T>
std::vector<svsbenchmark::vamana::ExpectedResult>
expected_results(std::string_view key, svs::DistanceType distance, const T& dataset) {
    const auto& table = parse_expected();
    auto v = svs::lib::load<std::vector<svsbenchmark::vamana::ExpectedResult>>(
        svs::lib::node_view_at(table, key), std::nullopt
    );
    auto output = std::vector<svsbenchmark::vamana::ExpectedResult>();
    for (const auto& i : v) {
        if ((i.distance_ == distance) && i.dataset_.match(dataset)) {
            output.push_back(i);
        }
    }
    return output;
}

/// Return the only reference build for the requested parameters.
/// Throws ANNException if the number of matches is not equal to one.
template <svsbenchmark::ValidDatasetSource T>
svsbenchmark::vamana::ExpectedResult
expected_build_results(svs::DistanceType distance, const T& dataset) {
    auto results = vamana::expected_results("vamana_test_build", distance, dataset);
    if (results.size() != 1) {
        throw ANNEXCEPTION("Got {} results when only one was expected!", results.size());
    }
    // Make sure the only result has build parameters.
    auto result = results[0];
    if (!result.build_parameters_.has_value()) {
        throw ANNEXCEPTION("Expected build result does not have build parameters!");
    }
    return result;
}

/// Return the only reference search for the requested parameters.
/// Throws ANNException if the number of dataset is not equal to one.
template <svsbenchmark::ValidDatasetSource T>
svsbenchmark::vamana::ExpectedResult
expected_search_results(svs::DistanceType distance, const T& dataset) {
    auto results = vamana::expected_results("vamana_test_search", distance, dataset);
    if (results.size() != 1) {
        throw ANNEXCEPTION("Got {} results when only one was expected!", results.size());
    }
    return results[0];
}

} // namespace test_dataset::vamana
