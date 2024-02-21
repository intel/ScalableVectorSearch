#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/build.h"
#include "svs-benchmark/datasets.h"
#include "svs-benchmark/index_traits.h"
#include "svs-benchmark/search.h"
#include "svs-benchmark/vamana/search.h"

// svs
#include "svs/index/vamana/dynamic_index.h"
#include "svs/orchestrators/vamana.h"

// stl
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace svsbenchmark::vamana {

struct StaticBenchmark {};
struct DynamicBenchmark {};

// Forward declarations
struct BuildJob;
struct DynamicBuildJob;

template <typename T> struct AssociatedJob;

template <> struct AssociatedJob<StaticBenchmark> {
    using type = BuildJob;
};

template <> struct AssociatedJob<DynamicBenchmark> {
    using type = DynamicBuildJob;
};

template <typename T> using associated_job_t = typename AssociatedJob<T>::type;

// Job names
inline constexpr std::string_view benchmark_name(StaticBenchmark) {
    return "vamana_static_build";
}

inline constexpr std::string_view benchmark_name(DynamicBenchmark) {
    return "vamana_dynamic_build";
}

// Entry-point for registering the static index building executable.
std::unique_ptr<Benchmark> static_workflow();
std::unique_ptr<Benchmark> dynamic_workflow();

// This enum controls the index tuning optimizations that happen at each step of the
// index modification.
enum class DynamicOptimizationLevel {
    // Only optimize the search window size on the test set.
    Minimal,
    // Optimize split-buffer only.
    SplitBufferOnTraining
};

inline constexpr std::string_view name(DynamicOptimizationLevel v) {
    switch (v) {
        using enum DynamicOptimizationLevel;
        case Minimal: {
            return "minimal";
        }
        case SplitBufferOnTraining: {
            return "split_buffer_on_training";
        }
    }
    throw ANNEXCEPTION("Unhandled enum value!");
}

inline DynamicOptimizationLevel parse_opt_level(std::string_view s) {
    using enum DynamicOptimizationLevel;
    if (constexpr auto c = name(Minimal); s == c) {
        return Minimal;
    } else if (constexpr auto c = name(SplitBufferOnTraining); s == c) {
        return SplitBufferOnTraining;
    }
    throw ANNEXCEPTION("Unparsable optimization level: {}", s);
}
} // namespace svsbenchmark::vamana

// Overload saving and loading.
template <> struct svs::lib::Saver<svsbenchmark::vamana::DynamicOptimizationLevel> {
    static SaveNode save(svsbenchmark::vamana::DynamicOptimizationLevel s) {
        return name(s);
    }
};

template <> struct svs::lib::Loader<svsbenchmark::vamana::DynamicOptimizationLevel> {
    using toml_type = toml::value<std::string>;
    static svsbenchmark::vamana::DynamicOptimizationLevel load(const toml_type& val) {
        return svsbenchmark::vamana::parse_opt_level(val.get());
    }
};

namespace svsbenchmark::vamana {

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
    svs::index::vamana::VamanaBuildParameters build_parameters_;
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
        const svs::index::vamana::VamanaBuildParameters& build_parameters,
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
    const svs::index::vamana::VamanaBuildParameters& get_build_parameters() const {
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
            svs::index::vamana::VamanaBuildParameters(1.2f, 64, 200, 750, 60, true),
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
    std::vector<size_t> search_window_sizes_;
    // Post-build validation parameters.
    svsbenchmark::search::SearchParameters search_parameters_;

  public:
    template <typename... Args>
    BuildJob(
        std::filesystem::path groundtruth,
        std::vector<size_t> search_window_sizes,
        svsbenchmark::search::SearchParameters search_parameters,
        Args&&... args
    )
        : BuildJobBase(std::forward<Args>(args)...)
        , groundtruth_{std::move(groundtruth)}
        , search_window_sizes_{std::move(search_window_sizes)}
        , search_parameters_{std::move(search_parameters)} {}

    // Return an example BuildJob that can be used to generate sample config files.
    static BuildJob example() {
        return BuildJob(
            "groundtruth.ivecs",
            {10, 20, 30, 40},
            svsbenchmark::search::SearchParameters::example(),
            BuildJobBase::example()
        );
    }

    // Compatibility with abstract search-space.
    std::vector<svs::index::vamana::VamanaSearchParameters> get_search_configs() const {
        auto results = std::vector<svs::index::vamana::VamanaSearchParameters>();
        for (size_t sws : search_window_sizes_) {
            results.push_back(
                svs::index::vamana::VamanaSearchParameters().buffer_config({sws, sws})
            );
        }
        return results;
    }
    const svsbenchmark::search::SearchParameters& get_search_parameters() const {
        return search_parameters_;
    }

    template <typename F>
    auto invoke(F&& f, const Checkpoint& SVS_UNUSED(checkpoint)) const {
        return f(dataset_, query_type_, data_type_, distance_, ndims_, *this);
    }

    // Versioning information for saving and reloading.
    // v0.0.2: Added `queries_in_training_set` field to divide the provided queries into
    //  a training set (for performance calibration) and a test set.
    // v0.0.3: Changed `build_type` to `dataset`, which is one of the variants defined by
    //  the `Datasets` class.
    static constexpr svs::lib::Version save_version = svs::lib::Version(0, 0, 3);
    static constexpr std::string_view serialization_schema = "benchmark_vamana_build_job";

    // Save the BuildJob to a TOML table.
    svs::lib::SaveTable save() const {
        // Get a base table.
        auto table = BuildJobBase::to_toml(serialization_schema, save_version);

        // Append the extra information needed by the static BuildJob.
        SVS_INSERT_SAVE_(table, groundtruth);
        SVS_INSERT_SAVE_(table, search_window_sizes);
        SVS_INSERT_SAVE_(table, search_parameters);
        return table;
    }

    // Load a BuildJob from a TOML table.
    static BuildJob load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root
    ) {
        return BuildJob{
            svsbenchmark::extract_filename(table, "groundtruth", root),
            SVS_LOAD_MEMBER_AT_(table, search_window_sizes),
            SVS_LOAD_MEMBER_AT_(table, search_parameters),
            BuildJobBase::from_toml(table, root)};
    }
};

struct DynamicBuildJob : public BuildJobBase {
  public:
    svsbenchmark::build::Schedule schedule_;
    DynamicOptimizationLevel dynamic_optimization_;
    svs::index::vamana::VamanaBuildParameters dynamic_parameters_;

  public:
    template <typename... Args>
    DynamicBuildJob(
        const svsbenchmark::build::Schedule& schedule,
        DynamicOptimizationLevel dynamic_optimization,
        const svs::index::vamana::VamanaBuildParameters& dynamic_parameters,
        Args&&... args
    )
        : BuildJobBase(std::forward<Args>(args)...)
        , schedule_{schedule}
        , dynamic_optimization_{dynamic_optimization}
        , dynamic_parameters_{dynamic_parameters} {}

    const svsbenchmark::build::Schedule& get_dynamic_schedule() const { return schedule_; }

    static DynamicBuildJob example() {
        return DynamicBuildJob(
            svsbenchmark::build::Schedule::example(),
            DynamicOptimizationLevel::Minimal,
            {1.2f, 64, 200, 750, 60, true},
            BuildJob::example()
        );
    }

    size_t queries_in_training_set() const { return queries_in_training_set_; }

    template <typename F> auto invoke(F&& f, const Checkpoint& checkpoint) const {
        return f(dataset_, query_type_, data_type_, distance_, ndims_, *this, checkpoint);
    }

    // Saving and Loading.

    // v0.0.2: Added `queries_in_training_set` field to divide the provided queries into
    //  a training set (for performance calibration) and a test set.
    //
    //  Also added `dynamic_optimization` taking values:
    //  - "minimal": Only tune search window size to achieve the desired recall on the
    //    test set.
    //  - "split_buffer_on_training": Tune the search buffer on the training set and the
    //    refine the search window size on the testing set.
    // v0.0.3: Switched to datasets as types rather than by string-matching.
    static constexpr svs::lib::Version save_version{0, 0, 3};
    static constexpr std::string_view serialization_schema =
        "benchmark_vamana_dynamic_build_job";
    svs::lib::SaveTable save() const {
        auto table = BuildJobBase::to_toml(serialization_schema, save_version);
        SVS_INSERT_SAVE_(table, schedule);
        SVS_INSERT_SAVE_(table, dynamic_optimization);
        SVS_INSERT_SAVE_(table, dynamic_parameters);
        return table;
    }

    static DynamicBuildJob load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root
    ) {
        return DynamicBuildJob(
            SVS_LOAD_MEMBER_AT_(table, schedule),
            SVS_LOAD_MEMBER_AT_(table, dynamic_optimization),
            SVS_LOAD_MEMBER_AT_(table, dynamic_parameters),
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

using DynamicBuildDispatcher = svs::lib::Dispatcher<
    toml::table,
    svsbenchmark::Dataset,
    svs::DataType,
    svs::DataType,
    svs::DistanceType,
    Extent,
    const DynamicBuildJob&,
    const Checkpoint&>;

} // namespace svsbenchmark::vamana
