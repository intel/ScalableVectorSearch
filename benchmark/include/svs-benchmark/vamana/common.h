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

// svs-benchmark
#include "svs-benchmark/benchmark.h"

// svs
#include "svs/core/distance.h"
#include "svs/index/vamana/search_params.h"

// stl
#include <initializer_list>

namespace svsbenchmark::vamana {

SVS_BENCHMARK_FOR_TESTS_ONLY inline float pick_alpha(svs::DistanceType distance) {
    switch (distance) {
        case svs::DistanceType::L2: {
            return 1.2;
        }
        case svs::DistanceType::MIP: {
            return 0.95;
        }
        case svs::DistanceType::Cosine: {
            return 0.95;
        }
    }
    throw ANNEXCEPTION("Unhandled distance type case!");
}

// Test Routines
SVS_BENCHMARK_FOR_TESTS_ONLY inline search::SearchParameters test_search_parameters() {
    return search::SearchParameters{10, {0.2, 0.5, 0.8, 0.9}};
}

SVS_BENCHMARK_FOR_TESTS_ONLY inline std::vector<svs::index::vamana::VamanaSearchParameters>
search_parameters_from_window_sizes(std::initializer_list<size_t> search_window_sizes) {
    auto v = std::vector<svs::index::vamana::VamanaSearchParameters>();
    for (auto i : search_window_sizes) {
        v.push_back({{i, i}, false, 1, 1});
    }
    return v;
}

SVS_BENCHMARK_FOR_TESTS_ONLY inline std::vector<svs::index::vamana::VamanaSearchParameters>
test_search_configs() {
    return std::vector<svs::index::vamana::VamanaSearchParameters>(
        {{{{10, 20}, false, 1, 1}, {{15, 15}, false, 1, 1}}}
    );
}

} // namespace svsbenchmark::vamana
