/*
 * Copyright (C) 2024 Intel Corporation
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
