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
#include "svs-benchmark/index_traits.h"
#include "svs-benchmark/search.h"

// svs
#include "svs/core/distance.h"
#include "svs/index/vamana/search_params.h"
#include "svs/lib/dispatcher.h"

// stl
#include <memory>

namespace svsbenchmark::vamana {

inline constexpr std::string_view search_benchmark_name() { return "vamana_static_search"; }

// Entry point for searching.
std::unique_ptr<Benchmark> search_static_workflow();

// The current state of the index.
struct VamanaState {
  public:
    svs::index::vamana::VamanaSearchParameters search_parameters_;
    size_t num_threads_;

  public:
    VamanaState(
        svs::index::vamana::VamanaSearchParameters search_parameters, size_t num_threads
    )
        : search_parameters_{search_parameters}
        , num_threads_{num_threads} {}

    template <typename Index>
    explicit VamanaState(const Index& index)
        : VamanaState(index.get_search_parameters(), index.get_num_threads()) {}

    // Saving
    // Version History
    // v0.0.0 - Initial Version:
    //   size_t search_window_size
    //   size_t num_threads
    //   bool visited_set_enabled
    // v0.0.1 - Refactor to use VamanaSearchParameters:
    //   svs::index::vamana::VamanaSearchParameters search_parameters_
    //   size_t num_threads_
    static constexpr svs::lib::Version save_version{0, 0, 1};
    static constexpr std::string_view serialization_schema = "benchmark_vamana_state";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(search_parameters), SVS_LIST_SAVE_(num_threads)}
        );
    }
};

struct SearchJob {
  public:
    std::string description_;
    svsbenchmark::Dataset dataset_;
    std::filesystem::path config_;
    std::filesystem::path graph_;
    std::filesystem::path data_;
    std::filesystem::path queries_;
    std::filesystem::path groundtruth_;
    size_t queries_in_training_set_;
    svs::DataType data_type_;
    svs::DataType query_type_;
    svs::DistanceType distance_;
    Extent ndims_;
    size_t num_threads_;
    svsbenchmark::search::SearchParameters search_parameters_;
    std::vector<svs::index::vamana::VamanaSearchParameters> preset_parameters_;

  public:
    SearchJob(
        std::string description,
        svsbenchmark::Dataset dataset,
        std::filesystem::path config,
        std::filesystem::path graph,
        std::filesystem::path data,
        std::filesystem::path queries,
        std::filesystem::path groundtruth,
        size_t queries_in_training_set,
        svs::DataType data_type,
        svs::DataType query_type,
        svs::DistanceType distance,
        Extent ndims,
        size_t num_threads,
        const svsbenchmark::search::SearchParameters& search_parameters,
        std::vector<svs::index::vamana::VamanaSearchParameters> preset_parameters
    )
        : description_{std::move(description)}
        , dataset_{std::move(dataset)}
        , config_{std::move(config)}
        , graph_{std::move(graph)}
        , data_{std::move(data)}
        , queries_{std::move(queries)}
        , groundtruth_{std::move(groundtruth)}
        , queries_in_training_set_{queries_in_training_set}
        , data_type_{data_type}
        , query_type_{query_type}
        , distance_{distance}
        , ndims_{ndims}
        , num_threads_{num_threads}
        , search_parameters_{search_parameters}
        , preset_parameters_{std::move(preset_parameters)} {}

    // Return the benchmark search parameters
    const svsbenchmark::search::SearchParameters& get_search_parameters() const {
        return search_parameters_;
    }

    // Compatbility with `ExpectedResults`
    static std::nullopt_t get_build_parameters() { return std::nullopt; }
    svs::DistanceType get_distance() const { return distance_; }

    // Return the preset search configurations.
    const std::vector<svs::index::vamana::VamanaSearchParameters>&
    get_search_configs() const {
        return preset_parameters_;
    }

    static SearchJob example() {
        return SearchJob{
            "index search",                                    // description
            Dataset::example(),                                // dataset
            "path/to/index/config",                            // config
            "path/to/graph",                                   // graph
            "path/to/data",                                    // data
            "path/to/queries",                                 // queries
            "path/to/groundtruth",                             // groundtruth
            5000,                                              // queries_in_training_set
            svs::DataType::float32,                            // data_type
            svs::DataType::float32,                            // query_type
            svs::DistanceType::L2,                             // distance
            Extent{svs::Dynamic},                              // ndims
            4,                                                 // num_threads
            search::SearchParameters::example(),               // search_parameters
            {{{10, 20}, false, 1, 1}, {{15, 15}, false, 1, 1}} // preset_parameters
        };
    }

    template <typename F>
    auto invoke(F&& f, const Checkpoint& SVS_UNUSED(checkpoiner)) const {
        return f(dataset_, query_type_, data_type_, distance_, ndims_, *this);
    }

    ///// Save/Load
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_vamana_search_job";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(description),
             SVS_LIST_SAVE_(dataset),
             SVS_LIST_SAVE_(config),
             SVS_LIST_SAVE_(graph),
             SVS_LIST_SAVE_(data),
             SVS_LIST_SAVE_(queries),
             SVS_LIST_SAVE_(groundtruth),
             SVS_LIST_SAVE_(queries_in_training_set),
             SVS_LIST_SAVE_(data_type),
             SVS_LIST_SAVE_(query_type),
             SVS_LIST_SAVE_(distance),
             SVS_LIST_SAVE_(ndims),
             SVS_LIST_SAVE_(num_threads),
             SVS_LIST_SAVE_(search_parameters),
             SVS_LIST_SAVE_(preset_parameters)}
        );
    }

    static SearchJob load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root = {}
    ) {
        return SearchJob{
            SVS_LOAD_MEMBER_AT_(table, description),
            SVS_LOAD_MEMBER_AT_(table, dataset, root),
            svsbenchmark::extract_filename(table, "config", root),
            svsbenchmark::extract_filename(table, "graph", root),
            svsbenchmark::extract_filename(table, "data", root),
            svsbenchmark::extract_filename(table, "queries", root),
            svsbenchmark::extract_filename(table, "groundtruth", root),
            SVS_LOAD_MEMBER_AT_(table, queries_in_training_set),
            SVS_LOAD_MEMBER_AT_(table, data_type),
            SVS_LOAD_MEMBER_AT_(table, query_type),
            SVS_LOAD_MEMBER_AT_(table, distance),
            SVS_LOAD_MEMBER_AT_(table, ndims),
            SVS_LOAD_MEMBER_AT_(table, num_threads),
            SVS_LOAD_MEMBER_AT_(table, search_parameters),
            SVS_LOAD_MEMBER_AT_(table, preset_parameters)};
    }
};

using StaticSearchDispatcher = svs::lib::Dispatcher<
    toml::table,
    Dataset,
    svs::DataType,
    svs::DataType,
    svs::DistanceType,
    Extent,
    const SearchJob&>;

} // namespace svsbenchmark::vamana
