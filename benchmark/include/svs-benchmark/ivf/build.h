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
#include "svs-benchmark/build.h"
#include "svs-benchmark/datasets.h"
#include "svs-benchmark/index_traits.h"
#include "svs-benchmark/ivf/search.h"
#include "svs-benchmark/search.h"

// svs
#include "svs/orchestrators/ivf.h"

// stl
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace svsbenchmark::ivf {

struct StaticBenchmark {};

// Forward declarations
struct BuildJob;

template <typename T> struct AssociatedJob;

template <> struct AssociatedJob<StaticBenchmark> {
    using type = BuildJob;
};

template <typename T> using associated_job_t = typename AssociatedJob<T>::type;

// Job names
inline constexpr std::string_view benchmark_name(StaticBenchmark) {
    return "ivf_static_build";
}

// Entry-point for registering the static index building executable.
std::unique_ptr<Benchmark> static_workflow();

// Shared struct between the static and dynamic paths.
struct BuildJobBase {
  public:
    // A descriptive name for this workload.
    std::string description_;

    // The dataset to load
    Dataset dataset_;

    // Paths
    std::filesystem::path data_;
    std::filesystem::path queries_;

    // The number of queries (taken form queries) to use in the training set.
    size_t queries_in_training_set_;

    // Dataset Parameters
    svs::DataType data_type_;
    svs::DataType query_type_;
    svs::DistanceType distance_;
    Extent ndims_;

    // Build Parameters
    svs::index::ivf::IVFBuildParameters build_parameters_;
    size_t num_threads_;

  public:
    ///// Contructor
    BuildJobBase(
        std::string_view description,
        svsbenchmark::Dataset dataset,
        std::filesystem::path data,
        std::filesystem::path queries,
        size_t queries_in_training_set,
        svs::DataType data_type,
        svs::DataType query_type,
        svs::DistanceType distance,
        size_t ndims,
        const svs::index::ivf::IVFBuildParameters& build_parameters,
        size_t num_threads
    )
        : description_{description}
        , dataset_{dataset}
        , data_{std::move(data)}
        , queries_{std::move(queries)}
        , queries_in_training_set_{queries_in_training_set}
        , data_type_{data_type}
        , query_type_{query_type}
        , distance_{distance}
        , ndims_{ndims}
        , build_parameters_{build_parameters}
        , num_threads_{num_threads} {}

    // Compatibility with `ExpectedResults`.
    const svs::index::ivf::IVFBuildParameters& get_build_parameters() const {
        return build_parameters_;
    }
    svs::DistanceType get_distance() const { return distance_; }

    // Return an example BuildJob that can be used to generate sample config files.
    static BuildJobBase example() {
        return BuildJobBase(
            "example index build",
            Dataset::example(),
            "data.fvecs",
            "queries.fvecs",
            5000,
            svs::DataType::float32,
            svs::DataType::float32,
            svs::DistanceType::L2,
            svs::Dynamic,
            svs::index::ivf::IVFBuildParameters(128, 10000, 10, false, 0.1),
            8
        );
    }

    svs::lib::SaveTable
    to_toml(std::string_view schema, const svs::lib::Version& version) const {
        return svs::lib::SaveTable(
            schema,
            version,
            {SVS_LIST_SAVE_(description),
             SVS_LIST_SAVE_(dataset),
             SVS_LIST_SAVE_(data),
             SVS_LIST_SAVE_(queries),
             SVS_LIST_SAVE_(queries_in_training_set),
             SVS_LIST_SAVE_(data_type),
             SVS_LIST_SAVE_(query_type),
             SVS_LIST_SAVE_(distance),
             SVS_LIST_SAVE_(ndims),
             SVS_LIST_SAVE_(build_parameters),
             SVS_LIST_SAVE_(num_threads)}
        );
    }

    static BuildJobBase from_toml(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root
    ) {
        namespace lib = svs::lib;
        return BuildJobBase(
            SVS_LOAD_MEMBER_AT_(table, description),
            SVS_LOAD_MEMBER_AT_(table, dataset, root),
            svsbenchmark::extract_filename(table, "data", root),
            svsbenchmark::extract_filename(table, "queries", root),
            SVS_LOAD_MEMBER_AT_(table, queries_in_training_set),
            SVS_LOAD_MEMBER_AT_(table, data_type),
            SVS_LOAD_MEMBER_AT_(table, query_type),
            SVS_LOAD_MEMBER_AT_(table, distance),
            SVS_LOAD_MEMBER_AT_(table, ndims),
            SVS_LOAD_MEMBER_AT_(table, build_parameters),
            SVS_LOAD_MEMBER_AT_(table, num_threads)
        );
    }
};

// Parsed setup for a static index build job.
struct BuildJob : public BuildJobBase {
  public:
    // Paths
    std::filesystem::path groundtruth_;
    // Preset search parameters
    std::vector<svs::index::ivf::IVFSearchParameters> preset_parameters_;
    // Post-build validation parameters.
    svsbenchmark::search::SearchParameters search_parameters_;
    // Directory to save the built index.
    // An empty optional implies no saving.
    std::optional<std::filesystem::path> save_directory_;

  public:
    template <typename... Args>
    BuildJob(
        std::filesystem::path groundtruth,
        std::vector<svs::index::ivf::IVFSearchParameters> preset_parameters,
        svsbenchmark::search::SearchParameters search_parameters,
        std::optional<std::filesystem::path> save_directory,
        Args&&... args
    )
        : BuildJobBase(std::forward<Args>(args)...)
        , groundtruth_{std::move(groundtruth)}
        , preset_parameters_{std::move(preset_parameters)}
        , search_parameters_{std::move(search_parameters)}
        , save_directory_{std::move(save_directory)} {}

    // Return an example BuildJob that can be used to generate sample config files.
    static BuildJob example() {
        return BuildJob(
            "groundtruth.ivecs",                               // groundtruth
            {{10, 1.0}, {10, 4.0}, {50, 1.0}},                 // preset_parameters
            svsbenchmark::search::SearchParameters::example(), // search_parameters
            std::nullopt,                                      // save_directory
            BuildJobBase::example()                            // base-class
        );
    }

    // Compatibility with abstract search-space.
    std::vector<svs::index::ivf::IVFSearchParameters> get_search_configs() const {
        return preset_parameters_;
    }
    const svsbenchmark::search::SearchParameters& get_search_parameters() const {
        return search_parameters_;
    }

    template <typename F>
    auto invoke(F&& f, const Checkpoint& SVS_UNUSED(checkpoint)) const {
        return f(dataset_, query_type_, data_type_, distance_, ndims_, *this);
    }

    // Save the index if the `save_directory` field is non-empty.
    template <typename Index> void maybe_save_index(Index& index) const {
        if (!save_directory_) {
            return;
        }
        const auto& root = save_directory_.value();
        svs::lib::save_to_disk(index, root / "clustering");
    }

    static constexpr svs::lib::Version save_version = svs::lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "benchmark_ivf_build_job";

    // Save the BuildJob to a TOML table.
    svs::lib::SaveTable save() const {
        // Get a base table.
        auto table = BuildJobBase::to_toml(serialization_schema, save_version);

        // Append the extra information needed by the static BuildJob.
        SVS_INSERT_SAVE_(table, groundtruth);
        SVS_INSERT_SAVE_(table, preset_parameters);
        SVS_INSERT_SAVE_(table, search_parameters);
        table.insert("save_directory", svs::lib::save(save_directory_.value_or("")));
        return table;
    }

    // Load a BuildJob from a TOML table.
    static BuildJob load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root,
        svsbenchmark::SaveDirectoryChecker& checker
    ) {
        return BuildJob(
            svsbenchmark::extract_filename(table, "groundtruth", root),
            SVS_LOAD_MEMBER_AT_(table, preset_parameters),
            SVS_LOAD_MEMBER_AT_(table, search_parameters),
            checker.extract(table.unwrap(), "save_directory"),
            BuildJobBase::from_toml(table, root)
        );
    }
};

// Dispatchers
using StaticBuildDispatcher = svs::lib::Dispatcher<
    toml::table,
    svsbenchmark::Dataset,
    svs::DataType,
    svs::DataType,
    svs::DistanceType,
    Extent,
    const BuildJob&>;

} // namespace svsbenchmark::ivf
