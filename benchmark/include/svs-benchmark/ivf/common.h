/*
 * Copyright 2025 Intel Corporation
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
#include "svs/index/ivf/common.h"

// stl
#include <initializer_list>

namespace svsbenchmark::ivf {

// Test Routines
SVS_BENCHMARK_FOR_TESTS_ONLY inline search::SearchParameters test_search_parameters() {
    return search::SearchParameters{10, {0.5, 0.8, 0.9}};
}

SVS_BENCHMARK_FOR_TESTS_ONLY inline std::vector<svs::index::ivf::IVFSearchParameters>
test_search_configs() {
    return std::vector<svs::index::ivf::IVFSearchParameters>({{{10, 1.0}, {50, 1.0}}});
}

} // namespace svsbenchmark::ivf
