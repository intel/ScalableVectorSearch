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

#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/ivf/build.h"
#include "svs-benchmark/ivf/search.h"
#include "svs-benchmark/ivf/static_traits.h"
#include "svs-benchmark/test.h"

// svs
#include "svs/orchestrators/ivf.h"

// stl
#include <filesystem>
#include <functional>
#include <memory>
#include <string_view>

namespace svsbenchmark::ivf {

inline constexpr std::string_view test_benchmark_name() { return "ivf_test_generator"; }

// A benchmark that generates reference inputs for unit tests.
std::unique_ptr<Benchmark> test_generator();

///// Test Runner
struct IVFTest {
  public:
    std::vector<svsbenchmark::DistanceAndGroundtruth> groundtruths_;
    std::filesystem::path data_f32_;
    std::filesystem::path index_config_;
    std::filesystem::path graph_;
    std::filesystem::path queries_f32_;
    size_t queries_in_training_set_;
    // Runtime values
    size_t num_threads_;

  public:
    IVFTest(
        std::vector<svsbenchmark::DistanceAndGroundtruth> groundtruths,
        std::filesystem::path data_f32,
        std::filesystem::path index_config,
        std::filesystem::path graph,
        std::filesystem::path queries_f32,
        size_t queries_in_training_set,
        // Runtime values
        size_t num_threads
    )
        : groundtruths_{std::move(groundtruths)}
        , data_f32_{std::move(data_f32)}
        , index_config_{std::move(index_config)}
        , graph_{std::move(graph)}
        , queries_f32_{std::move(queries_f32)}
        , queries_in_training_set_{queries_in_training_set}
        , num_threads_{num_threads} {}

    static IVFTest example() {
        return IVFTest{
            {DistanceAndGroundtruth::example()}, // groundtruths
            "path/to/data_f32",                  // data_f32
            "path/to/config",                    // index_config
            "path/to/graph",                     // graph
            "path/to/queries_f32",               // queries_f32
            10000,                               // queries_in_training_set
            0,                                   // Num Threads (not-saved)
        };
    }

    const std::filesystem::path& groundtruth_for(svs::DistanceType distance) const {
        for (const auto& pair : groundtruths_) {
            if (pair.distance_ == distance) {
                return pair.path_;
            }
        }
        throw ANNEXCEPTION("Could not find a groundtruth for {} distance!", distance);
    }

    ///// Save/Load
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_ivf_test";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(groundtruths),
             SVS_LIST_SAVE_(data_f32),
             SVS_LIST_SAVE_(index_config),
             SVS_LIST_SAVE_(graph),
             SVS_LIST_SAVE_(queries_f32),
             SVS_LIST_SAVE_(queries_in_training_set)}
        );
    }

    static IVFTest load(
        const svs::lib::ContextFreeLoadTable& table,
        size_t num_threads,
        const std::optional<std::filesystem::path>& root = {}
    ) {
        return IVFTest{
            SVS_LOAD_MEMBER_AT_(table, groundtruths, root),
            svsbenchmark::extract_filename(table, "data_f32", root),
            svsbenchmark::extract_filename(table, "index_config", root),
            svsbenchmark::extract_filename(table, "graph", root),
            svsbenchmark::extract_filename(table, "queries_f32", root),
            SVS_LOAD_MEMBER_AT_(table, queries_in_training_set),
            num_threads};
    }
};

// Specialize ConfigAndResult for `svs::IVF`.
using ConfigAndResult =
    svsbenchmark::ConfigAndResultPrototype<svs::index::ivf::IVFSearchParameters>;

// Specialize ExpectedResult for `svs::IVF`.
using ExpectedResult = svsbenchmark::ExpectedResultPrototype<
    svs::index::ivf::IVFBuildParameters,
    svs::index::ivf::IVFSearchParameters>;

// Test functions take the test input and returns a `TestFunctionReturn` with the results.
using TestFunction = std::function<svsbenchmark::TestFunctionReturn(const IVFTest&)>;

} // namespace svsbenchmark::ivf
