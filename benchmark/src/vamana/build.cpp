/*
 * Copyright 2023 Intel Corporation
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
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/executable.h"
#include "svs-benchmark/vamana/uncompressed.h"

// svs
#include "svs/lib/saveload.h"
#include "svs/third-party/toml.h"

// third-party
#include "fmt/core.h"
#include "fmt/ranges.h"

// stl
#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

namespace svsbenchmark::vamana {
namespace {

/////
///// Dispatchers
/////

template <typename BenchmarkType> struct BuildDispatcher;

// Dispatcher for Uncompressed Data.
template <> struct BuildDispatcher<StaticBenchmark> {
    using type = vamana::StaticBuildDispatcher;
    static type dispatcher() {
        auto dispatcher = type{};
        vamana::register_uncompressed_static_build(dispatcher);
        return dispatcher;
    }
};

template <> struct BuildDispatcher<DynamicBenchmark> {
    using type = vamana::DynamicBuildDispatcher;
    static type dispatcher() {
        auto dispatcher = type{};
        vamana::register_uncompressed_dynamic_build(dispatcher);
        return dispatcher;
    }
};

/////
///// Executable
/////

constexpr std::string_view HELP_TEMPLATE = R"(
Run a {} benchmark for the Vamana index.

Usage:
    (1) src-file.toml (output-file.toml/--validate) [basename]
    (2) --help
    (3) --example

1. Run all the benchmarks in the global `{}` array in `src-file.toml`.
   All elements in the array must be parseable as a `{}`.

   Results will be saved to `output-file.toml`.

   If `--validate` is given as the second argument, then all pre-run checks will be
   performed on the input file and arguments but not benchmark will actually be run.

   Optional third argument `basename` will be used as the root for all file paths parsed.

2. Print this help message.

3. Display an example input TOML file to `stdout`.

Backend specializations are dispatched on the following fields of the input TOML file:
* build_type: The dataset type to use.
* query_type: The element type of the query dataset.
* data_type: The input type of the source dataset.
* distance: The distance function to use.
* ndims: The compile-time dimensionality.

Compiled specializations are listed below:
{{ build_type, query_type, data_type, distance, ndims }}
)";

template <typename BenchmarkType> struct Exe {
  public:
    // type alises
    using job_type = associated_job_t<BenchmarkType>;
    using dispatcher_type = typename BuildDispatcher<BenchmarkType>::type;

    static dispatcher_type dispatcher() {
        return BuildDispatcher<BenchmarkType>::dispatcher();
    }

    static std::string_view name() { return benchmark_name(BenchmarkType()); }

    static void print_help() {
        if constexpr (std::is_same_v<BenchmarkType, StaticBenchmark>) {
            fmt::print(
                HELP_TEMPLATE,
                "static build and search",
                name(),
                "svsbenchmark::Vamana::BuildJob"
            );
        } else if constexpr (std::is_same_v<BenchmarkType, DynamicBenchmark>) {
            fmt::print(
                HELP_TEMPLATE,
                "dynamic build, modification, and search",
                name(),
                "svsbenchmark::Vamana::DynamicBuildJob"
            );
        } else {
            throw ANNEXCEPTION("Unreachable");
        }
        auto f = dispatcher();
        for (size_t i = 0; i < f.size(); ++i) {
            auto dispatch_strings = std::array<std::string, 5>{
                f.description(i, 0),
                f.description(i, 1),
                f.description(i, 2),
                f.description(i, 3),
                f.description(i, 4),
            };
            fmt::print("{{ {} }}\n", fmt::join(dispatch_strings, ", "));
        }
    }

    static job_type example() { return job_type::example(); }

    template <typename F>
    static std::optional<std::vector<job_type>>
    parse_args_and_invoke(F&& f, std::span<const std::string_view> args) {
        auto root = std::optional<std::filesystem::path>();
        if (args.size() == 1) {
            root = std::filesystem::path(args[0]);
        }
        if constexpr (std::is_same_v<BenchmarkType, StaticBenchmark>) {
            return f(root, svsbenchmark::SaveDirectoryChecker());
        } else {
            return f(root);
        }
    }
};
} // namespace

// Return an executor for this benchmark.
std::unique_ptr<Benchmark> static_workflow() {
    return std::make_unique<JobBasedExecutable<Exe<StaticBenchmark>>>();
}
std::unique_ptr<Benchmark> dynamic_workflow() {
    return std::make_unique<JobBasedExecutable<Exe<DynamicBenchmark>>>();
}
} // namespace svsbenchmark::vamana
