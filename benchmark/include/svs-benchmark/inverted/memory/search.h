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
#include "svs-benchmark/datasets.h"
#include "svs-benchmark/inverted/memory/build.h"
#include "svs-benchmark/search.h"

// svs
#include "svs/lib/saveload.h"

// stl
#include <filesystem>
#include <string_view>

namespace svsbenchmark::inverted::memory {

/////
///// Search setup using a piecemeal approach.
/////

struct PiecewiseAssembly {
  public:
    svsbenchmark::Dataset dataset_;
    svs::DataType query_type_;
    svs::DataType data_type_;
    Extent ndims_;
    svs::DistanceType distance_;
    ClusterStrategy strategy_; // strategy used by the backend clustering representation
    std::filesystem::path clustering_;
    std::filesystem::path primary_index_config_;
    std::filesystem::path primary_index_graph_;

  public:
    PiecewiseAssembly(
        svsbenchmark::Dataset dataset,
        svs::DataType query_type,
        svs::DataType data_type,
        Extent ndims,
        svs::DistanceType distance,
        ClusterStrategy strategy,
        std::filesystem::path clustering,
        std::filesystem::path primary_index_config,
        std::filesystem::path primary_index_graph
    )
        : dataset_{std::move(dataset)}
        , query_type_{query_type}
        , data_type_{data_type}
        , ndims_{ndims}
        , distance_{distance}
        , strategy_{strategy}
        , clustering_{std::move(clustering)}
        , primary_index_config_{std::move(primary_index_config)}
        , primary_index_graph_{std::move(primary_index_graph)} {}

    static PiecewiseAssembly example() {
        return PiecewiseAssembly(
            Dataset::example(),      // dataset
            svs::DataType::float32,  // query_type
            svs::DataType::float16,  // data_type
            Extent(svs::Dynamic),    // ndims
            svs::DistanceType::L2,   // distance
            ClusterStrategy::Sparse, // strategy
            "clustering_dir",        // clustering_dir
            "primary_config_dir",    // primary_config_dir
            "primary_graph_dir"      // primary_graph_dir
        );
    }

    ///// Save/Load
    /// Version History
    /// - v0.0.1: Added support datasets instead of build_type.
    static constexpr svs::lib::Version save_version{0, 0, 1};
    static constexpr std::string_view serialization_schema =
        "benchmark_inverted_memory_piecewise_assembly";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(dataset),
             SVS_LIST_SAVE_(query_type),
             SVS_LIST_SAVE_(data_type),
             SVS_LIST_SAVE_(ndims),
             SVS_LIST_SAVE_(distance),
             SVS_LIST_SAVE_(strategy),
             SVS_LIST_SAVE_(clustering),
             SVS_LIST_SAVE_(primary_index_config),
             SVS_LIST_SAVE_(primary_index_graph)}
        );
    }

    static PiecewiseAssembly load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root
    ) {
        return PiecewiseAssembly{
            SVS_LOAD_MEMBER_AT_(table, dataset, root),
            SVS_LOAD_MEMBER_AT_(table, query_type),
            SVS_LOAD_MEMBER_AT_(table, data_type),
            SVS_LOAD_MEMBER_AT_(table, ndims),
            SVS_LOAD_MEMBER_AT_(table, distance),
            SVS_LOAD_MEMBER_AT_(table, strategy),
            extract_filename(table, "clustering", root),
            extract_filename(table, "primary_index_config", root),
            extract_filename(table, "primary_index_graph", root)};
    }
};

struct MemorySearchJob {
  public:
    PiecewiseAssembly assembly_;
    size_t num_threads_;
    std::vector<svs::index::inverted::InvertedSearchParameters> search_configs_;
    svsbenchmark::search::SearchParameters search_targets_;
    std::filesystem::path original_data_;
    std::filesystem::path queries_;
    std::filesystem::path groundtruth_;

  public:
    MemorySearchJob(
        const PiecewiseAssembly& assembly,
        size_t num_threads,
        std::vector<svs::index::inverted::InvertedSearchParameters>&& search_configs,
        const svsbenchmark::search::SearchParameters& search_targets,
        std::filesystem::path original_data,
        std::filesystem::path queries,
        std::filesystem::path groundtruth
    )
        : assembly_{assembly}
        , num_threads_{num_threads}
        , search_configs_{std::move(search_configs)}
        , search_targets_{search_targets}
        , original_data_{std::move(original_data)}
        , queries_{std::move(queries)}
        , groundtruth_{std::move(groundtruth)} {}

    static MemorySearchJob example() {
        return MemorySearchJob{
            PiecewiseAssembly::example(),                      // assembly
            10,                                                // num_threads
            {},                                                // search_configs
            svsbenchmark::search::SearchParameters::example(), // search_targets
            "path/to/data",                                    // original_data
            "path/to/queries",                                 // queries
            "path/to/groundtruth"                              // groundtruth
        };
    }

    const svsbenchmark::search::SearchParameters& get_search_parameters() const {
        return search_targets_;
    }

    std::vector<svs::index::inverted::InvertedSearchParameters> get_search_configs() const {
        return search_configs_;
    }

    std::nullopt_t get_build_parameters() const { return std::nullopt; }

    template <typename F>
    auto invoke(F&& f, const Checkpoint& SVS_UNUSED(checkpoint)) const {
        return f(
            assembly_.dataset_,
            assembly_.query_type_,
            assembly_.data_type_,
            assembly_.distance_,
            assembly_.strategy_,
            assembly_.ndims_,
            *this
        );
    }

    ///// Save/Load
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema =
        "benchmark_inverted_memory_search_job";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(assembly),
             SVS_LIST_SAVE_(num_threads),
             SVS_LIST_SAVE_(search_configs),
             SVS_LIST_SAVE_(search_targets),
             SVS_LIST_SAVE_(original_data),
             SVS_LIST_SAVE_(queries),
             SVS_LIST_SAVE_(groundtruth)}
        );
    }

    static MemorySearchJob load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& data_root
    ) {
        return MemorySearchJob{
            SVS_LOAD_MEMBER_AT_(table, assembly, data_root),
            SVS_LOAD_MEMBER_AT_(table, num_threads),
            SVS_LOAD_MEMBER_AT_(table, search_configs),
            SVS_LOAD_MEMBER_AT_(table, search_targets),
            extract_filename(table, "original_data", data_root),
            extract_filename(table, "queries", data_root),
            extract_filename(table, "groundtruth", data_root)};
    }
};

/// Dispatcher Aliases
using MemorySearchDispatcher = svs::lib::Dispatcher<
    toml::table,
    svsbenchmark::Dataset,
    svs::DataType,
    svs::DataType,
    svs::DistanceType,
    svsbenchmark::inverted::memory::ClusterStrategy,
    Extent,
    const MemorySearchJob&>;

} // namespace svsbenchmark::inverted::memory
