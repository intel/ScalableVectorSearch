#pragma once

// Shared utilities for test reference generators.

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/datasets.h"
#include "svs-benchmark/index_traits.h"
#include "svs-benchmark/search.h"

// svs
#include "svs/core/distance.h"
#include "svs/lib/saveload.h"

// stl
#include <cmath>
#include <optional>

namespace svsbenchmark {

struct DistanceAndGroundtruth {
  public:
    svs::DistanceType distance_;
    std::filesystem::path path_;

  public:
    DistanceAndGroundtruth(const svs::DistanceType& distance, std::filesystem::path path)
        : distance_{distance}
        , path_{std::move(path)} {}

    static DistanceAndGroundtruth example() {
        return DistanceAndGroundtruth{
            svs::DistanceType::L2,     // distance
            "path/to/groundtruth/file" // path
        };
    }

    ///// Save/Load
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema =
        "benchmark_distance_and_groundtruth";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(distance), SVS_LIST_SAVE_(path)}
        );
    }

    static DistanceAndGroundtruth load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root = {}
    ) {
        return DistanceAndGroundtruth{
            SVS_LOAD_MEMBER_AT_(table, distance),
            svsbenchmark::extract_filename(table, "path", root)};
    }
};

// Simplified test results.
//
// By default - the benchmarking framework emits a very rich output.
// This isn't suitable for ingestion in the tests as it needs full type information
// of the serialized classes.
//
// Instead, these simplified representations are generated for use in testing.
template <typename SearchParameters> struct ConfigAndResultPrototype {
  public:
    SearchParameters search_parameters_;
    size_t num_neighbors_;
    size_t recall_k_;
    size_t num_queries_;
    double recall_;

  public:
    ConfigAndResultPrototype(
        const SearchParameters& search_parameters,
        size_t num_neighbors,
        size_t recall_k,
        size_t num_queries,
        double recall
    )
        : search_parameters_{search_parameters}
        , num_neighbors_{num_neighbors}
        , recall_k_{recall_k}
        , num_queries_{num_queries}
        , recall_{recall} {}

    // Construct from RunReport.
    template <typename Index>
        requires(std::is_same_v<svsbenchmark::config_type<Index>, SearchParameters>)
    explicit ConfigAndResultPrototype(const search::RunReport<Index>& report)
        : search_parameters_{report.config_}
        , num_neighbors_{report.num_neighbors_}
        , recall_k_{report.num_neighbors_}
        , num_queries_{report.num_queries_}
        , recall_{report.recall_} {}

    ///// Save/Load
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_config_and_result";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(search_parameters),
             SVS_LIST_SAVE_(num_neighbors),
             SVS_LIST_SAVE_(recall_k),
             SVS_LIST_SAVE_(num_queries),
             SVS_LIST_SAVE_(recall)}
        );
    }

    static ConfigAndResultPrototype load(const svs::lib::ContextFreeLoadTable& table) {
        return ConfigAndResultPrototype{
            SVS_LOAD_MEMBER_AT_(table, search_parameters),
            SVS_LOAD_MEMBER_AT_(table, num_neighbors),
            SVS_LOAD_MEMBER_AT_(table, recall_k),
            SVS_LOAD_MEMBER_AT_(table, num_queries),
            SVS_LOAD_MEMBER_AT_(table, recall)};
    }
};

template <typename BuildParameters, typename SearchParameters>
struct ExpectedResultPrototype {
  public:
    // Type Aliases
    using config_and_result_type = ConfigAndResultPrototype<SearchParameters>;

  public:
    // The kind of dataset used.
    svsbenchmark::Dataset dataset_;
    // The distance used for these results.
    svs::DistanceType distance_;
    // Build parameters. Left empty if used for search only.
    std::optional<BuildParameters> build_parameters_;
    // A list of configurations and the expected recall.
    std::vector<config_and_result_type> config_and_recall_;

  public:
    ExpectedResultPrototype(
        svsbenchmark::Dataset dataset,
        svs::DistanceType distance,
        std::optional<BuildParameters> build_parameters,
        std::vector<config_and_result_type> config_and_recall
    )
        : dataset_{std::move(dataset)}
        , distance_{distance}
        , build_parameters_{std::move(build_parameters)}
        , config_and_recall_{std::move(config_and_recall)} {}

    // Construct from a Search Report
    template <typename Job, typename Index, typename Mixin>
    ExpectedResultPrototype(
        svsbenchmark::Dataset dataset, const search::SearchReport<Job, Index, Mixin>& report
    )
        : dataset_{std::move(dataset)}
        , distance_(report.job_.get_distance())
        , build_parameters_{report.job_.get_build_parameters()}
        , config_and_recall_{} {
        for (const auto& x : report.target_configs_) {
            config_and_recall_.emplace_back(x);
        }
        for (const auto& x : report.target_recalls_) {
            config_and_recall_.emplace_back(x);
        }
    }

    ///// Save/Load
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_expected_result";
    svs::lib::SaveTable save() const {
        auto table = svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(dataset),
             SVS_LIST_SAVE_(distance),
             SVS_LIST_SAVE_(config_and_recall)}
        );
        if (build_parameters_.has_value()) {
            table.insert("build_parameters", svs::lib::save(*build_parameters_));
        }
        return table;
    }

    static ExpectedResultPrototype load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root
    ) {
        auto build_parameters = std::optional<BuildParameters>{std::nullopt};
        if (table.contains("build_parameters")) {
            build_parameters.emplace(
                svs::lib::load_at<BuildParameters>(table, "build_parameters")
            );
        }

        return ExpectedResultPrototype{
            SVS_LOAD_MEMBER_AT_(table, dataset, root),
            SVS_LOAD_MEMBER_AT_(table, distance),
            std::move(build_parameters),
            SVS_LOAD_MEMBER_AT_(table, config_and_recall)};
    }
};

struct TestFunctionReturn {
    std::string key_;
    toml::table results_;
};

/////
///// Dataset Transformation
/////

// Transformations on datasets.
namespace detail {
// TODO: Adjust dataset so narrowing doesn't throw.
inline uint8_t convert_to(svs::lib::Type<uint8_t>, float x) {
    return svs::lib::narrow<uint8_t>(std::clamp<float>(
        std::trunc(x),
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max()
    ));
}
inline int8_t convert_to(svs::lib::Type<int8_t>, float x) {
    return svs::lib::narrow<int8_t>(std::clamp<float>(
        std::trunc(x),
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max()
    ));
}
inline svs::Float16 convert_to(svs::lib::Type<svs::Float16>, float x) {
    return svs::Float16{x};
}
inline float convert_to(svs::lib::Type<float>, float x) { return x; }
} // namespace detail

template <typename To, size_t N, typename Allocator>
svs::data::SimpleData<To, N>
convert_data(svs::lib::Type<To> to, const svs::data::SimpleData<float, N, Allocator>& src) {
    auto dst = svs::data::SimpleData<To, N>(src.size(), src.dimensions());

    auto b = std::vector<To>(src.dimensions());
    for (size_t i = 0, imax = src.size(); i < imax; ++i) {
        const auto& d = src.get_datum(i);
        for (size_t j = 0; j < d.size(); ++j) {
            b.at(j) = detail::convert_to(to, d[j]);
        }
        dst.set_datum(i, b);
    }
    return dst;
}

} // namespace svsbenchmark
