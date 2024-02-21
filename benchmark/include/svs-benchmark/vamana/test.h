#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/test.h"
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/vamana/search.h"
#include "svs-benchmark/vamana/static_traits.h"

// svs
#include "svs/orchestrators/vamana.h"

// stl
#include <filesystem>
#include <functional>
#include <memory>
#include <string_view>

namespace svsbenchmark::vamana {

inline constexpr std::string_view test_benchmark_name() { return "vamana_test_generator"; }

// A benchmark that generates reference inputs for unit tests.
std::unique_ptr<Benchmark> test_generator();

///// Test Runner
struct VamanaTest {
  public:
    std::vector<svsbenchmark::DistanceAndGroundtruth> groundtruths_;
    std::filesystem::path data_f32_;
    std::filesystem::path index_config_;
    std::filesystem::path graph_;
    std::filesystem::path queries_f32_;
    size_t queries_in_training_set_;
    // Backend-specific members
    std::filesystem::path leanvec_data_matrix_;
    std::filesystem::path leanvec_query_matrix_;
    // Runtime values
    size_t num_threads_;

  public:
    VamanaTest(
        std::vector<svsbenchmark::DistanceAndGroundtruth> groundtruths,
        std::filesystem::path data_f32,
        std::filesystem::path index_config,
        std::filesystem::path graph,
        std::filesystem::path queries_f32,
        size_t queries_in_training_set,
        // backend-specific members
        std::filesystem::path leanvec_data_matrix,
        std::filesystem::path leanvec_query_matrix,
        // Runtime values
        size_t num_threads
    )
        : groundtruths_{std::move(groundtruths)}
        , data_f32_{std::move(data_f32)}
        , index_config_{std::move(index_config)}
        , graph_{std::move(graph)}
        , queries_f32_{std::move(queries_f32)}
        , queries_in_training_set_{queries_in_training_set}
        , leanvec_data_matrix_{std::move(leanvec_data_matrix)}
        , leanvec_query_matrix_{std::move(leanvec_query_matrix)}
        , num_threads_{num_threads} {}

    static VamanaTest example() {
        return VamanaTest{
            {DistanceAndGroundtruth::example()}, // groundtruths
            "path/to/data_f32",                  // data_f32
            "path/to/config",                    // index_config
            "path/to/graph",                     // graph
            "path/to/queries_f32",               // queries_f32
            10000,                               // queries_in_training_set
            "path/to/leanvec_data_matrix",       // LeanVec data matrix
            "path/to/leanvec_query_matrix",      // LeanVec query matrix
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
    // Version History
    // * v0.0.1 (Breaking): Added `leanvec_data_matrix` and `leanvec_query_matrix` file
    //   paths.
    //
    //   This is an incompatible change since generation and consumption of reference
    //   results is expected to be entirely internal to SVS.
    static constexpr svs::lib::Version save_version{0, 0, 1};
    static constexpr std::string_view serialization_schema = "benchmark_vamana_test";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(groundtruths),
             SVS_LIST_SAVE_(data_f32),
             SVS_LIST_SAVE_(index_config),
             SVS_LIST_SAVE_(graph),
             SVS_LIST_SAVE_(queries_f32),
             SVS_LIST_SAVE_(queries_in_training_set),
             SVS_LIST_SAVE_(leanvec_data_matrix),
             SVS_LIST_SAVE_(leanvec_query_matrix)}
        );
    }

    static VamanaTest load(
        const svs::lib::ContextFreeLoadTable& table,
        size_t num_threads,
        const std::optional<std::filesystem::path>& root = {}
    ) {
        return VamanaTest{
            SVS_LOAD_MEMBER_AT_(table, groundtruths, root),
            svsbenchmark::extract_filename(table, "data_f32", root),
            svsbenchmark::extract_filename(table, "index_config", root),
            svsbenchmark::extract_filename(table, "graph", root),
            svsbenchmark::extract_filename(table, "queries_f32", root),
            SVS_LOAD_MEMBER_AT_(table, queries_in_training_set),
            svsbenchmark::extract_filename(table, "leanvec_data_matrix", root),
            svsbenchmark::extract_filename(table, "leanvec_query_matrix", root),
            num_threads};
    }
};

// Specialize ConfigAndResult for `svs::Vamana`.
using ConfigAndResult =
    svsbenchmark::ConfigAndResultPrototype<svs::index::vamana::VamanaSearchParameters>;

// Specialize ExpectedResult for `svs::Vamana`.
using ExpectedResult = svsbenchmark::ExpectedResultPrototype<
    svs::index::vamana::VamanaBuildParameters,
    svs::index::vamana::VamanaSearchParameters>;

// Test functions take the test input and returns a `TestFunctionReturn` with the results.
using TestFunction = std::function<svsbenchmark::TestFunctionReturn(const VamanaTest&)>;

} // namespace svsbenchmark::vamana
