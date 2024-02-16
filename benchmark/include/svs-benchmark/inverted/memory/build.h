#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/build.h"
#include "svs-benchmark/datasets.h"
#include "svs-benchmark/inverted/inverted.h"
#include "svs-benchmark/inverted/memory/common.h"
#include "svs-benchmark/search.h"

// svs
#include "svs/index/inverted/memory_search_params.h"
#include "svs/index/vamana/build_params.h"
#include "svs/lib/saveload.h"
#include "svs/orchestrators/inverted.h"

// stl
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svsbenchmark::inverted::memory {

/////
///// State Configuration
/////

struct MemoryInvertedState {
  public:
    svs::index::inverted::InvertedSearchParameters search_parameters_;
    size_t num_threads_;

  public:
    MemoryInvertedState(
        const svs::index::inverted::InvertedSearchParameters& search_parameters,
        size_t num_threads
    )
        : search_parameters_{search_parameters}
        , num_threads_{num_threads} {}

    // Saving.
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema =
        "benchmark_inverted_memory_state";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(search_parameters), SVS_LIST_SAVE_(num_threads)}
        );
    }
};

/////
///// Job definitions
/////

struct MemoryBuildJob {
  public:
    // A descriptive name for this workload.
    std::string description_;

    // The type of the build.
    svsbenchmark::Dataset dataset_;

    // Paths
    std::filesystem::path data_;
    std::filesystem::path queries_;
    std::filesystem::path groundtruth_;

    // Dataset Parameters
    svs::DataType data_type_;
    svs::DataType query_type_;
    svs::DistanceType distance_;
    Extent ndims_;

    // Build Parameters
    svs::index::vamana::VamanaBuildParameters primary_build_parameters_;
    svs::index::inverted::ClusteringParameters clustering_parameters_;
    std::optional<std::filesystem::path> centroids_directory_; // predefined centroids
    size_t num_build_threads_;

    // Backend Parameters
    ClusterStrategy strategy_;

    // Determine if we want to save an intermediate state of the index.
    // If not given, assume that saving is not desired.
    std::optional<std::filesystem::path> save_directory_;

    // Post-build
    std::vector<svs::index::inverted::InvertedSearchParameters> search_configs_;
    svsbenchmark::search::SearchParameters search_parameters_;

  public:
    MemoryBuildJob(
        std::string description,
        svsbenchmark::Dataset dataset,
        // Paths
        std::filesystem::path data,
        std::filesystem::path queries,
        std::filesystem::path groundtruth,
        // Dataset
        svs::DataType data_type,
        svs::DataType query_type,
        svs::DistanceType distance,
        Extent ndims,
        // Build
        const svs::index::vamana::VamanaBuildParameters& primary_build_parameters,
        const svs::index::inverted::ClusteringParameters& clustering_parameters,
        std::optional<std::filesystem::path> centroids_directory,
        size_t num_build_threads,

        // Backend
        ClusterStrategy strategy,
        std::optional<std::filesystem::path> save_directory,
        // Search
        std::vector<svs::index::inverted::InvertedSearchParameters> search_configs,
        const svsbenchmark::search::SearchParameters& search_parameters
    )
        : description_{std::move(description)}
        , dataset_{std::move(dataset)}
        , data_{std::move(data)}
        , queries_{std::move(queries)}
        , groundtruth_{std::move(groundtruth)}
        , data_type_{data_type}
        , query_type_{query_type}
        , distance_{distance}
        , ndims_{ndims}
        , primary_build_parameters_{primary_build_parameters}
        , clustering_parameters_{clustering_parameters}
        , centroids_directory_{std::move(centroids_directory)}
        , num_build_threads_{num_build_threads}
        , strategy_{strategy}
        , save_directory_{std::move(save_directory)}
        , search_configs_{std::move(search_configs)}
        , search_parameters_{search_parameters} {}

    static MemoryBuildJob example() {
        return MemoryBuildJob(
            "example index build",
            Dataset::example(),
            "data",
            "queries",
            "groundtruth",
            svs::DataType::float32,
            svs::DataType::float32,
            svs::DistanceType::L2,
            Extent(svs::Dynamic),
            {1.2f, 64, 200, 750, 60, true},
            {},
            std::nullopt,
            10,
            ClusterStrategy::Sparse,
            {},
            std::vector<svs::index::inverted::InvertedSearchParameters>({{}}),
            svsbenchmark::search::SearchParameters::example()
        );
    }

    // Accessor functions.
    svs::DistanceType get_distance() const { return distance_; }
    svs::index::inverted::InvertedBuildParameters get_build_parameters() const {
        return svs::index::inverted::InvertedBuildParameters{
            clustering_parameters_, primary_build_parameters_};
    }

    std::vector<svs::index::inverted::InvertedSearchParameters> get_search_configs() const {
        return search_configs_;
    }

    svsbenchmark::search::SearchParameters get_search_parameters() const {
        return search_parameters_;
    }

    template <typename F>
    auto invoke(F&& f, const Checkpoint& SVS_UNUSED(checkpoint)) const {
        return f(dataset_, query_type_, data_type_, distance_, strategy_, ndims_, *this);
    }

    ///// Save/Load
    /// Version History
    /// - v0.0.1: Added support for datasets rather than build_type
    static constexpr svs::lib::Version save_version{0, 0, 1};
    static constexpr std::string_view serialization_schema =
        "benchmark_inverted_memory_build_job";
    svs::lib::SaveTable save() const {
        auto centroids_directory = centroids_directory_.value_or("");
        auto save_directory = save_directory_.value_or("");
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(description),
             SVS_LIST_SAVE_(dataset),
             SVS_LIST_SAVE_(data),
             SVS_LIST_SAVE_(queries),
             SVS_LIST_SAVE_(groundtruth),
             SVS_LIST_SAVE_(data_type),
             SVS_LIST_SAVE_(query_type),
             SVS_LIST_SAVE_(distance),
             SVS_LIST_SAVE_(ndims),
             SVS_LIST_SAVE_(primary_build_parameters),
             SVS_LIST_SAVE_(clustering_parameters),
             {"centroids_directory", svs::lib::save(centroids_directory)},
             SVS_LIST_SAVE_(num_build_threads),
             SVS_LIST_SAVE_(strategy),
             {"save_directory", svs::lib::save(save_directory)},
             SVS_LIST_SAVE_(search_configs),
             SVS_LIST_SAVE_(search_parameters)}
        );
    }

    void make_save_directory() const {
        if (save_directory_.has_value()) {
            auto& dir = *save_directory_;
            if (!std::filesystem::is_directory(dir)) {
                std::filesystem::create_directory(dir);
            }
        }
    }

    static MemoryBuildJob load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root,
        svsbenchmark::SaveDirectoryChecker& checker
    ) {
        auto centroids_directory_file =
            svs::lib::load_at<std::filesystem::path>(table, "centroids_directory");
        auto centroids_directory = std::optional<std::filesystem::path>();
        if (!centroids_directory_file.empty()) {
            centroids_directory.emplace(std::move(centroids_directory_file));
        }

        return MemoryBuildJob(
            SVS_LOAD_MEMBER_AT_(table, description),
            SVS_LOAD_MEMBER_AT_(table, dataset, root),
            svsbenchmark::extract_filename(table, "data", root),
            svsbenchmark::extract_filename(table, "queries", root),
            svsbenchmark::extract_filename(table, "groundtruth", root),
            SVS_LOAD_MEMBER_AT_(table, data_type),
            SVS_LOAD_MEMBER_AT_(table, query_type),
            SVS_LOAD_MEMBER_AT_(table, distance),
            SVS_LOAD_MEMBER_AT_(table, ndims),
            SVS_LOAD_MEMBER_AT_(table, primary_build_parameters),
            SVS_LOAD_MEMBER_AT_(table, clustering_parameters),
            std::move(centroids_directory),
            SVS_LOAD_MEMBER_AT_(table, num_build_threads),
            SVS_LOAD_MEMBER_AT_(table, strategy),
            checker.extract(table.unwrap(), "save_directory"),
            SVS_LOAD_MEMBER_AT_(table, search_configs),
            SVS_LOAD_MEMBER_AT_(table, search_parameters)
        );
    }
};

struct CentroidSelector {
  public:
    template <svs::data::ImmutableMemoryDataset Data, std::integral I = uint32_t>
    std::vector<I, svs::lib::Allocator<I>> operator()(
        const Data& data,
        const svs::index::inverted::ClusteringParameters& clustering_parameters,
        size_t num_threads,
        svs::lib::Type<I> integer_type = {}
    ) const {
        if (directory_.has_value()) {
            std::vector<I, svs::lib::Allocator<I>> centroids = svs::lib::load_from_disk<
                svs::lib::BinaryBlobLoader<I, svs::lib::Allocator<I>>>(directory_.value());
            inverted::validate_external_centroids<I>(centroids, data.size());
            return centroids;
        }

        return svs::index::inverted::pick_centroids_randomly(
            data, clustering_parameters, num_threads, integer_type
        );
    }

  public:
    std::optional<std::filesystem::path> directory_;
};

struct ClusteringSaver {
    // Methods
  public:
    template <std::integral I>
    void operator()(const svs::index::inverted::Clustering<I>& clustering) {
        if (directory_.has_value()) {
            svs::lib::save_to_disk(clustering, directory_.value());
        }
    }

    // Construct with an optional root and a sub-directory.
    // If the root contains a value, then the constructed inner directory will as well.
    ClusteringSaver(
        const std::optional<std::filesystem::path>& root, std::string_view subdir
    )
        : directory_{std::nullopt} {
        if (root.has_value()) {
            directory_ = (*root) / subdir;
        }
    }

    // Members
  public:
    std::optional<std::filesystem::path> directory_{};
};

///// Shared Functions
// Memory-based clustering/saving/building
template <typename QueryType, typename Loader, typename Distance, typename ClusterStrategy>
auto build(
    const MemoryBuildJob& job,
    const Loader& loader,
    Distance distance,
    ClusterStrategy strategy
) {
    auto cluster_save_path = std::optional<std::filesystem::path>();
    if (job.save_directory_.has_value()) {
        job.make_save_directory();
    }

    auto index = svs::Inverted::build<QueryType>(
        job.get_build_parameters(),
        loader,
        distance,
        job.num_build_threads_,
        strategy,
        CentroidSelector{job.centroids_directory_},
        ClusteringSaver{job.save_directory_, "clustering"}
    );

    if (job.save_directory_.has_value()) {
        auto root = *(job.save_directory_);
        index.save_primary_index(
            root / "vamana_config", root / "vamana_graph", root / "vamana_data"
        );
    }

    return index;
}

///// Dispatcher Aliases
using MemoryBuildDispatcher = svs::lib::Dispatcher<
    toml::table,
    svsbenchmark::Dataset,
    svs::DataType,
    svs::DataType,
    svs::DistanceType,
    svsbenchmark::inverted::memory::ClusterStrategy,
    Extent,
    const MemoryBuildJob&>;

} // namespace svsbenchmark::inverted::memory
