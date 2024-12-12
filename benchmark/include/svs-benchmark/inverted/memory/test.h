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

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/inverted/memory/build.h"
#include "svs-benchmark/test.h"

// svs
#include "svs/index/inverted/memory_based.h"

namespace svsbenchmark::inverted::memory {

inline constexpr std::string_view test_benchmark_name() {
    return "inverted_test_generator";
}

// A benchmark that generates reference inputs for unit tests.
std::unique_ptr<Benchmark> test_generator();

///// Test Runner
struct InvertedTest {
  public:
    // Persistent Members
    std::vector<svsbenchmark::DistanceAndGroundtruth> groundtruths_;
    std::filesystem::path data_f32_;
    std::filesystem::path queries_f32_;
    size_t queries_in_training_set_;
    size_t num_threads_;

  public:
    InvertedTest(
        std::vector<svsbenchmark::DistanceAndGroundtruth> groundtruths,
        std::filesystem::path data_f32,
        std::filesystem::path queries_f32,
        size_t queries_in_training_set,
        size_t num_threads
    )
        : groundtruths_{std::move(groundtruths)}
        , data_f32_{std::move(data_f32)}
        , queries_f32_{std::move(queries_f32)}
        , queries_in_training_set_{queries_in_training_set}
        , num_threads_{num_threads} {
        if (num_threads == 0) {
            throw ANNEXCEPTION("Cannot construct an InvertedTest with 0 threads!");
        }
    }

    static InvertedTest example() {
        return InvertedTest{
            {DistanceAndGroundtruth::example()}, // groundtruths
            "path/to/data_f32",                  // data_f32
            "path/to/queries_f32",               // queries_f32
            1000,                                // queries_in_training_set
            1                                    // num_threads
        };
    }

    const std::filesystem::path& groundtruth_for(svs::DistanceType distance) const {
        for (auto& pair : groundtruths_) {
            if (pair.distance_ == distance) {
                return pair.path_;
            }
        }
        throw ANNEXCEPTION("Could not find a groundtruth for {} distance!", distance);
    }

    ///// Save/Load
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema =
        "benchmark_inverted_memory_test";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(groundtruths),
             SVS_LIST_SAVE_(data_f32),
             SVS_LIST_SAVE_(queries_f32),
             SVS_LIST_SAVE_(queries_in_training_set),
             SVS_LIST_SAVE_(num_threads)}
        );
    }

    // Require `num_threads` to be provided on a load.
    // This helps keep use of this class portable to machines with fewer threads.
    static InvertedTest load(
        const svs::lib::ContextFreeLoadTable& table,
        size_t num_threads,
        const std::optional<std::filesystem::path>& root = {}
    ) {
        return InvertedTest{
            SVS_LOAD_MEMBER_AT_(table, groundtruths, root),
            svsbenchmark::extract_filename(table, "data_f32", root),
            svsbenchmark::extract_filename(table, "queries_f32", root),
            SVS_LOAD_MEMBER_AT_(table, queries_in_training_set),
            num_threads};
    }
};

using ConfigAndResult =
    svsbenchmark::ConfigAndResultPrototype<svs::index::inverted::InvertedSearchParameters>;

// Specialize ExpectedResult for the inverted index.
using ExpectedResult = svsbenchmark::ExpectedResultPrototype<
    svs::index::inverted::InvertedBuildParameters,
    svs::index::inverted::InvertedSearchParameters>;

// Test functions take the test input and returns a `TestFunctionReturn` with the
using TestFunction = std::function<TestFunctionReturn(const InvertedTest&)>;

} // namespace svsbenchmark::inverted::memory
