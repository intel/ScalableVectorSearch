#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/build.h"

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

// Dynamic configuration for calibrating accuracies between runs.
struct VamanaConfig {
  public:
    size_t search_window_size_;

  public:
    explicit VamanaConfig(size_t search_window_size)
        : search_window_size_{search_window_size} {}

    // Saving and loading.
    static constexpr svs::lib::Version save_version{0, 0, 0};
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(save_version, {SVS_LIST_SAVE_(search_window_size)});
    }

    static VamanaConfig load(const toml::table& table, const svs::lib::Version& version) {
        if (version != save_version) {
            throw ANNEXCEPTION("Version mismatch!");
        }

        return VamanaConfig(SVS_LOAD_MEMBER_AT_(table, search_window_size));
    }
};

// Shared struct between the static and dynamic paths.
struct BuildJobBase {
  public:
    // A descriptive name for this workload.
    std::string description_;

    // The type of build
    std::string build_type_;

    // Paths
    std::filesystem::path data_;
    std::filesystem::path queries_;

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
        std::string_view build_type,
        std::filesystem::path data,
        std::filesystem::path queries,
        svs::DataType data_type,
        svs::DataType query_type,
        svs::DistanceType distance,
        size_t ndims,
        const svs::index::vamana::VamanaBuildParameters& build_parameters,
        size_t num_threads
    )
        : description_{description}
        , build_type_{build_type}
        , data_{std::move(data)}
        , queries_{std::move(queries)}
        , data_type_{data_type}
        , query_type_{query_type}
        , distance_{distance}
        , ndims_{ndims}
        , build_parameters_{build_parameters}
        , num_threads_{num_threads} {}

    // Return an example BuildJob that can be used to generate sample config files.
    static BuildJobBase example() {
        return BuildJobBase(
            "example index build",
            Uncompressed::name(),
            "data.fvecs",
            "queries.fvecs",
            svs::DataType::float32,
            svs::DataType::float32,
            svs::DistanceType::L2,
            svs::Dynamic,
            svs::index::vamana::VamanaBuildParameters(1.2f, 64, 200, 750, 60, true),
            8
        );
    }

    svs::lib::SaveTable to_toml(const svs::lib::Version& version) const {
        return svs::lib::SaveTable(
            version,
            {SVS_LIST_SAVE_(description),
             SVS_LIST_SAVE_(build_type),
             SVS_LIST_SAVE_(data),
             SVS_LIST_SAVE_(queries),
             SVS_LIST_SAVE_(data_type),
             SVS_LIST_SAVE_(query_type),
             SVS_LIST_SAVE_(distance),
             SVS_LIST_SAVE_(ndims),
             SVS_LIST_SAVE_(build_parameters),
             SVS_LIST_SAVE_(num_threads)}
        );
    }

    static BuildJobBase
    from_toml(const toml::table& table, const std::optional<std::filesystem::path>& root) {
        namespace lib = svs::lib;
        return BuildJobBase(
            SVS_LOAD_MEMBER_AT_(table, description),
            SVS_LOAD_MEMBER_AT_(table, build_type),
            svsbenchmark::extract_filename(table, "data", root),
            svsbenchmark::extract_filename(table, "queries", root),
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
    svsbenchmark::build::SearchParameters search_parameters_;

  public:
    template <typename... Args>
    BuildJob(
        std::filesystem::path groundtruth,
        std::vector<size_t> search_window_sizes,
        svsbenchmark::build::SearchParameters search_parameters,
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
            svsbenchmark::build::SearchParameters::example(),
            BuildJobBase::example()
        );
    }

    // Compatibility with abstract search-space.
    std::vector<VamanaConfig> get_search_configs() const {
        return std::vector<VamanaConfig>(
            search_window_sizes_.begin(), search_window_sizes_.end()
        );
    }
    const svsbenchmark::build::SearchParameters& get_search_parameters() const {
        return search_parameters_;
    }

    // Versioning information for saving and reloading.
    static constexpr svs::lib::Version save_version = svs::lib::Version(0, 0, 1);

    // Save the BuildJob to a TOML table.
    svs::lib::SaveTable save() const {
        // Get a base table.
        auto table = BuildJobBase::to_toml(save_version);

        // Append the extra information needed by the static BuildJob.
        SVS_INSERT_SAVE_(table, groundtruth);
        SVS_INSERT_SAVE_(table, search_window_sizes);
        SVS_INSERT_SAVE_(table, search_parameters);
        return table;
    }

    // Load a BuildJob from a TOML table.
    static BuildJob load(
        const toml::table& table,
        const svs::lib::Version& version,
        const std::optional<std::filesystem::path>& root
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Version mismatch!");
        }

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
    svs::index::vamana::VamanaBuildParameters dynamic_parameters_;

  public:
    template <typename... Args>
    DynamicBuildJob(
        const svsbenchmark::build::Schedule& schedule,
        const svs::index::vamana::VamanaBuildParameters& dynamic_parameters,
        Args&&... args
    )
        : BuildJobBase(std::forward<Args>(args)...)
        , schedule_{schedule}
        , dynamic_parameters_{dynamic_parameters} {}

    const svsbenchmark::build::Schedule& get_dynamic_schedule() const { return schedule_; }

    static DynamicBuildJob example() {
        return DynamicBuildJob(
            svsbenchmark::build::Schedule::example(),
            {1.2f, 64, 200, 750, 60, true},
            BuildJob::example()
        );
    }

    static constexpr svs::lib::Version save_version{0, 0, 1};
    svs::lib::SaveTable save() const {
        auto table = BuildJobBase::to_toml(save_version);
        SVS_INSERT_SAVE_(table, schedule);
        SVS_INSERT_SAVE_(table, dynamic_parameters);
        return table;
    }

    static DynamicBuildJob load(
        const toml::table& table,
        const svs::lib::Version& version,
        const std::optional<std::filesystem::path>& root
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Version mismatch!");
        }

        return DynamicBuildJob(
            SVS_LOAD_MEMBER_AT_(table, schedule),
            SVS_LOAD_MEMBER_AT_(table, dynamic_parameters),
            BuildJobBase::from_toml(table, root)
        );
    }
};

// // Struct recording performance of repeated searches of an index.
// struct RunReport {
//   public:
//     size_t num_queries_;
//     size_t search_window_size_;
//     bool visited_set_;
//     size_t num_threads_;
//     double recall_;
//     std::vector<double> latency_seconds_;
//
//   public:
//     RunReport(
//         size_t num_queries,
//         size_t search_window_size,
//         bool visited_set,
//         size_t num_threads,
//         double recall,
//         std::vector<double> latency_seconds
//     )
//         : num_queries_{num_queries}
//         , search_window_size_{search_window_size}
//         , visited_set_{visited_set}
//         , num_threads_{num_threads}
//         , recall_{recall}
//         , latency_seconds_{std::move(latency_seconds)} {}
//
//     // Default equality operator.
//     friend bool operator==(const RunReport&, const RunReport&) = default;
//
//     // Saving and reloading.
//     static constexpr svs::lib::Version save_version{0, 0, 0};
//     svs::lib::SaveTable save() const {
//         return svs::lib::SaveTable(
//             save_version,
//             {
//                 SVS_LIST_SAVE_(num_queries),
//                 SVS_LIST_SAVE_(search_window_size),
//                 SVS_LIST_SAVE_(visited_set),
//                 SVS_LIST_SAVE_(num_threads),
//                 SVS_LIST_SAVE_(recall),
//                 SVS_LIST_SAVE_(latency_seconds),
//             }
//         );
//     }
//
//     static RunReport load(const toml::table& table, const svs::lib::Version& version) {
//         if (version != save_version) {
//             throw ANNEXCEPTION("Version mismatch trying to reload a RunReport!");
//         }
//
//         return RunReport(
//             SVS_LOAD_MEMBER_AT_(table, num_queries),
//             SVS_LOAD_MEMBER_AT_(table, search_window_size),
//             SVS_LOAD_MEMBER_AT_(table, visited_set),
//             SVS_LOAD_MEMBER_AT_(table, num_threads),
//             SVS_LOAD_MEMBER_AT_(table, recall),
//             SVS_LOAD_MEMBER_AT_(table, latency_seconds)
//         );
//     }
// };
//
// struct DynamicOperation {
//   public:
//     DynamicOpKind op_kind_;
//     double op_time_;
//     double groundtruth_time_;
//     std::vector<RunReport> iso_recall_;
//     std::vector<RunReport> iso_windowsize_;
//
//   public:
//     DynamicOperation(
//         DynamicOpKind op_kind,
//         double op_time,
//         double groundtruth_time,
//         std::vector<RunReport> iso_recall,
//         std::vector<RunReport> iso_windowsize
//     )
//         : op_kind_{op_kind}
//         , op_time_{op_time}
//         , groundtruth_time_{groundtruth_time}
//         , iso_recall_{std::move(iso_recall)}
//         , iso_windowsize_{std::move(iso_windowsize)} {}
//
//     static constexpr svs::lib::Version save_version{0, 0, 0};
//     svs::lib::SaveTable save() const {
//         return svs::lib::SaveTable(
//             save_version,
//             {
//                 SVS_LIST_SAVE_(op_kind),
//                 SVS_LIST_SAVE_(op_time),
//                 SVS_LIST_SAVE_(groundtruth_time),
//                 SVS_LIST_SAVE_(iso_recall),
//                 SVS_LIST_SAVE_(iso_windowsize),
//             }
//         );
//     }
// };

struct VamanaState {
  public:
    size_t search_window_size_;
    size_t num_threads_;
    bool visited_set_;

  public:
    VamanaState(size_t search_window_size, size_t num_threads, bool visited_set)
        : search_window_size_{search_window_size}
        , num_threads_{num_threads}
        , visited_set_{visited_set} {}

    template <typename Index>
    explicit VamanaState(const Index& index)
        : VamanaState(
              index.get_search_window_size(),
              index.get_num_threads(),
              index.visited_set_enabled()
          ) {}

    // Saving
    static constexpr svs::lib::Version save_version{0, 0, 0};
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            save_version,
            {SVS_LIST_SAVE_(search_window_size),
             SVS_LIST_SAVE_(num_threads),
             SVS_LIST_SAVE_(visited_set)}
        );
    }
};

template <typename Index, svs::data::ImmutableMemoryDataset Queries, typename Groundtruth>
size_t calibrate(
    Index& index,
    const Queries& queries,
    const Groundtruth& groundtruth,
    size_t num_neighbors,
    double target_recall
) {
    auto range = svs::threads::UnitRange<size_t>(num_neighbors, 300);
    return *std::lower_bound(
        range.begin(),
        range.end(),
        target_recall,
        [&](size_t window_size, double recall) {
            index.set_search_window_size(window_size);
            auto result = index.search(queries, num_neighbors);
            auto this_recall = svs::k_recall_at_n(groundtruth, result);
            return this_recall < recall;
        }
    );
}

} // namespace svsbenchmark::vamana

// Bridge for the Dynamic Index.
namespace svsbenchmark::build {

template <typename Graph, typename Data, typename Dist>
struct IndexTraits<svs::index::vamana::MutableVamanaIndex<Graph, Data, Dist>> {
    using index_type = svs::index::vamana::MutableVamanaIndex<Graph, Data, Dist>;

    // Search window size.
    using config_type = svsbenchmark::vamana::VamanaConfig;
    using state_type = svsbenchmark::vamana::VamanaState;

    static std::string name() { return "dynamic vamana index"; }

    // Dynamic Operations
    template <svs::data::ImmutableMemoryDataset Points>
    static void
    add_points(index_type& index, const Points& points, const std::vector<size_t>& ids) {
        index.add_points(points, ids);
    }

    static void delete_points(index_type& index, const std::vector<size_t>& ids) {
        index.delete_entries(ids);
    }

    static void consolidate(index_type& index) {
        index.consolidate();
        index.compact();
    }

    // Configuration Space.
    static void apply_config(index_type& index, config_type config) {
        index.set_search_window_size(config.search_window_size_);
    }

    template <svs::data::ImmutableMemoryDataset Queries>
    static auto search(
        index_type& index, const Queries& queries, size_t num_neighbors, config_type config
    ) {
        apply_config(index, config);
        return index.search(queries, num_neighbors);
    }

    static state_type report_state(const index_type& index) { return state_type(index); }

    template <svs::data::ImmutableMemoryDataset Queries, typename Groundtruth>
    static config_type calibrate(
        index_type& index,
        const Queries& queries,
        const Groundtruth& groundtruth,
        size_t num_neighbors,
        double target_recall
    ) {
        return config_type(svsbenchmark::vamana::calibrate(
            index, queries, groundtruth, num_neighbors, target_recall
        ));
    }
};

template <> struct IndexTraits<svs::Vamana> {
    using config_type = svsbenchmark::vamana::VamanaConfig;
    using state_type = svsbenchmark::vamana::VamanaState;

    static std::string name() { return "static vamana index (type erased)"; }

    // Configuration Space.
    static void apply_config(svs::Vamana& index, config_type config) {
        index.set_search_window_size(config.search_window_size_);
    }

    template <svs::data::ImmutableMemoryDataset Queries>
    static auto search(
        svs::Vamana& index, const Queries& queries, size_t num_neighbors, config_type config
    ) {
        apply_config(index, config);
        return index.search(queries, num_neighbors);
    }

    static state_type report_state(const svs::Vamana& index) { return state_type(index); }

    template <svs::data::ImmutableMemoryDataset Queries, typename Groundtruth>
    static config_type calibrate(
        svs::Vamana& index,
        const Queries& queries,
        const Groundtruth& groundtruth,
        size_t num_neighbors,
        double target_recall
    ) {
        return config_type(svsbenchmark::vamana::calibrate(
            index, queries, groundtruth, num_neighbors, target_recall
        ));
    }
};

} // namespace svsbenchmark::build
