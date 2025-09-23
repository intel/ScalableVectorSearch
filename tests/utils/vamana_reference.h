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

// svs-test
#include "tests/utils/test_dataset.h"

// svsbenchmark
#include "svs-benchmark/datasets.h"
#include "svs-benchmark/vamana/test.h"
#include "svs/index/vamana/dynamic_index.h"

// svs
#include "svs/core/distance.h"

// stl
#include <optional>
#include <set>
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

template <typename DistanceFunctor, svsbenchmark::ValidDatasetSource T>
svsbenchmark::vamana::ExpectedResult
expected_build_results(DistanceFunctor, const T& dataset) {
    // Delegate to the DistanceType overload using the deduced distance type
    return expected_build_results(svs::distance_type_v<DistanceFunctor>, dataset);
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

/// Load a test index with default contained types.
svs::index::vamana::VamanaIndex<
    svs::graphs::SimpleGraph<uint32_t>,
    svs::data::SimpleData<float>,
    svs::DistanceL2>
load_test_index();

/// Load a test index with a custom distance functor.
template <typename Distance>
svs::index::vamana::
    VamanaIndex<svs::graphs::SimpleGraph<uint32_t>, svs::data::SimpleData<float>, Distance>
    load_test_index(const Distance& distance) {
    return svs::index::vamana::auto_assemble(
        test_dataset::vamana_config_file(),
        test_dataset::graph(),
        test_dataset::data_f32(),
        distance,
        1
    );
}

/// Load a test index with default contained types.
svs::index::vamana::MutableVamanaIndex<
    svs::graphs::SimpleGraph<uint32_t>,
    svs::data::SimpleData<float>,
    svs::DistanceL2>
load_dynamic_test_index();

template <typename Distance>
svs::index::vamana::MutableVamanaIndex<
    svs::graphs::SimpleGraph<uint32_t>,
    svs::data::SimpleData<float>,
    Distance>
load_dynamic_test_index(const Distance& distance) {
    return svs::index::vamana::auto_dynamic_assemble(
        test_dataset::vamana_config_file(),
        test_dataset::graph(),
        test_dataset::data_f32(),
        distance,
        1,
        true // debug_load_from_static
    );
}

// Return the set of distances that have reference build expectations for the
// uncompressed float32 dataset. Cached after first computation.
inline const std::set<svs::DistanceType>& available_build_distances() {
    static const std::set<svs::DistanceType> distances = []() {
        std::set<svs::DistanceType> ds;
        const auto dataset = svsbenchmark::Uncompressed(svs::DataType::float32);
        const auto& table = parse_expected();
        auto all_results =
            svs::lib::load<std::vector<svsbenchmark::vamana::ExpectedResult>>(
                svs::lib::node_view_at(table, "vamana_test_build"), std::nullopt
            );
        for (const auto& r : all_results) {
            if (r.dataset_.match(dataset)) {
                ds.insert(r.distance_);
            }
        }
        return ds;
    }();
    return distances;
}

} // namespace test_dataset::vamana
