/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */

#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/index_traits.h"
#include "svs-benchmark/search.h"

// svs
#include "svs/concepts/data.h"
#include "svs/core/recall.h"
#include "svs/lib/invoke.h"
#include "svs/lib/saveload.h"
#include "svs/misc/dynamic_helper.h"

// stl
#include <chrono>
#include <concepts>
#include <string>
#include <vector>

// Shared testing infrastructure for the mutable index.
namespace svsbenchmark::build {

template <std::integral I> I div(I i, float fraction) {
    return svs::lib::narrow<I>(std::floor(svs::lib::narrow<float>(i) * fraction));
}

// Setup for the the Dynamic schedule.
struct Schedule {
  public:
    // The fraction of the dataset to be used for initial construction.
    double initial_fraction_;
    // The fraction of the dataset to be added or deleted each iteration.
    double modify_fraction_;
    // The number of insertions and deletions to do before performing maintenance.
    size_t cycles_before_cleanup_;
    // The total number of iterations to run.
    size_t total_iterations_;
    // Config parameters for the dynamic helper.
    size_t bucket_divisor_;
    // The seed for the pseudo-random number generator used.
    uint64_t rng_seed_;
    // The search parameters.
    search::SearchParameters search_parameters_;

  public:
    Schedule(
        double initial_fraction,
        double modify_fraction,
        size_t cycles_before_cleanup,
        size_t total_iterations,
        size_t bucket_divisor,
        uint64_t rng_seed,
        search::SearchParameters search_parameters
    )
        : initial_fraction_{initial_fraction}
        , modify_fraction_{modify_fraction}
        , cycles_before_cleanup_{cycles_before_cleanup}
        , total_iterations_{total_iterations}
        , bucket_divisor_{bucket_divisor}
        , rng_seed_{rng_seed}
        , search_parameters_{std::move(search_parameters)} {}

    static Schedule example() {
        return Schedule(
            0.75, 0.01, 5, 20, 32, 0xc0ffee, search::SearchParameters::example()
        );
    }

    ///// Saving and Loading.
    // Version history
    // - v0.0.0: Initial version
    // - v0.0.1 (breaking): Added a `uint64_t rng_seed` field to initialize the random
    //      number generator used for additions and deletions.
    //
    //      Breaking to avoid legacy config entries from accidentally using a differenent
    //      seed.
    static constexpr svs::lib::Version save_version{0, 0, 1};
    static constexpr std::string_view serialization_schema = "benchmark_dynamic_schedule";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(initial_fraction),
             SVS_LIST_SAVE_(modify_fraction),
             SVS_LIST_SAVE_(cycles_before_cleanup),
             SVS_LIST_SAVE_(total_iterations),
             SVS_LIST_SAVE_(bucket_divisor),
             {"rng_seed", svs::lib::save(svs::lib::FullUnsigned(rng_seed_))},
             SVS_LIST_SAVE_(search_parameters)}
        );
    }

    static Schedule load(const svs::lib::ContextFreeLoadTable& table) {
        return Schedule(
            SVS_LOAD_MEMBER_AT_(table, initial_fraction),
            SVS_LOAD_MEMBER_AT_(table, modify_fraction),
            SVS_LOAD_MEMBER_AT_(table, cycles_before_cleanup),
            SVS_LOAD_MEMBER_AT_(table, total_iterations),
            SVS_LOAD_MEMBER_AT_(table, bucket_divisor),
            svs::lib::load_at<svs::lib::FullUnsigned>(table, "rng_seed"),
            SVS_LOAD_MEMBER_AT_(table, search_parameters)
        );
    }
};

enum class DynamicOpKind { Initial, Add, Delete, Consolidate };

inline std::string_view name(DynamicOpKind op) {
    switch (op) {
        case DynamicOpKind::Initial: {
            return "initial";
        }
        case DynamicOpKind::Add: {
            return "add";
        }
        case DynamicOpKind::Delete: {
            return "delete";
        }
        case DynamicOpKind::Consolidate: {
            return "consolidate";
        }
    }
    throw ANNEXCEPTION("Unknown op!");
}
} // namespace svsbenchmark::build

namespace svs::lib {
template <> struct Saver<svsbenchmark::build::DynamicOpKind> {
    static SaveNode save(svsbenchmark::build::DynamicOpKind val) {
        return SaveNode(name(val));
    }
};

} // namespace svs::lib

namespace svsbenchmark::build {

template <typename Index> struct DynamicOperation {
  public:
    DynamicOpKind kind_ = DynamicOpKind::Initial;
    double time_ = 0;
    double groundtruth_time_ = 0;
    std::vector<search::RunReport<Index>> iso_recall_{};
    std::vector<search::RunReport<Index>> iso_config_{};

  public:
    DynamicOperation() = default;

    DynamicOperation(
        DynamicOpKind kind,
        double time,
        double groundtruth_time,
        std::vector<search::RunReport<Index>> iso_recall,
        std::vector<search::RunReport<Index>> iso_config
    )
        : kind_{kind}
        , time_{time}
        , groundtruth_time_{groundtruth_time}
        , iso_recall_{std::move(iso_recall)}
        , iso_config_{std::move(iso_config)} {}

    // Saving
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_dynamic_operation";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {
                SVS_LIST_SAVE_(kind),
                SVS_LIST_SAVE_(time),
                SVS_LIST_SAVE_(groundtruth_time),
                SVS_LIST_SAVE_(iso_recall),
                SVS_LIST_SAVE_(iso_config),
            }
        );
    }
};

/////
///// High Level Reports
/////

template <typename Job, typename Index> struct StaticReport {
  public:
    double build_time_;
    std::chrono::time_point<std::chrono::system_clock> timestamp_;
    // The incoming job that these results are for.
    Job job_;
    // A descriptive name for the index.
    std::string index_description_;
    // Results for pre-generated configurations.
    std::vector<search::RunReport<Index>> target_configs_;
    std::vector<search::RunReport<Index>> target_recalls_;

  public:
    StaticReport(
        double build_time,
        const Job& job,
        std::string index_description,
        std::vector<search::RunReport<Index>> target_configs,
        std::vector<search::RunReport<Index>> target_recalls
    )
        : build_time_{build_time}
        , timestamp_{std::chrono::system_clock::now()}
        , job_{job}
        , index_description_{std::move(index_description)}
        , target_configs_{std::move(target_configs)}
        , target_recalls_{std::move(target_recalls)} {}

    // Saving.
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_static_report";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(build_time),
             SVS_LIST_SAVE_(timestamp),
             SVS_LIST_SAVE_(job),
             SVS_LIST_SAVE_(index_description),
             SVS_LIST_SAVE_(target_configs),
             SVS_LIST_SAVE_(target_recalls)}
        );
    }
};

template <typename Job, typename Index> struct DynamicReport {
  public:
    std::chrono::time_point<std::chrono::system_clock> timestamp_;
    Job job_;
    std::string index_description_;
    std::vector<DynamicOperation<Index>> results_;

  public:
    DynamicReport(const Job& job, std::string index_description)
        : timestamp_{std::chrono::system_clock::now()}
        , job_{job}
        , index_description_{std::move(index_description)}
        , results_{} {}

    void push_back(DynamicOperation<Index>&& op) { results_.push_back(std::move(op)); }
    void push_back(const DynamicOperation<Index>& op) { results_.push_back(op); }

    // Saving
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_dynamic_report";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(timestamp),
             SVS_LIST_SAVE_(job),
             SVS_LIST_SAVE_(index_description),
             SVS_LIST_SAVE_(results)}
        );
    }
};

// Main test pipeline.
template <typename Index, typename T, typename Q, typename Distance> struct Bundle {
  public:
    // Type aliases
    using index_type = Index;

  public:
    // Members
    Index index;
    svs::misc::ReferenceDataset<size_t, T, svs::Dynamic, Distance> reference;
    svs::data::SimpleData<Q> queries;
    double build_time;
};

// Initialization.
namespace detail {
template <typename F, typename T>
using deduce_index_type = std::
    invoke_result_t<const F&, const svs::data::SimpleData<T>&, const std::vector<size_t>&>;
}

template <typename T, typename Q, typename Distance, typename Init>
Bundle<detail::deduce_index_type<Init, T>, T, Q, Distance> initialize_dynamic(
    const std::filesystem::path& data_path,
    const std::filesystem::path& query_path,
    const Distance& distance,
    const Schedule& schedule,
    size_t num_threads,
    const Init& init
) {
    using Index = detail::deduce_index_type<Init, T>;
    auto queries = svs::data::SimpleData<Q>::load(query_path);
    auto data = svs::data::SimpleData<T>::load(data_path);
    size_t total_points = data.size();
    auto reference = svs::misc::ReferenceDataset<size_t, T, svs::Dynamic, Distance>(
        std::move(data),
        distance,
        num_threads,
        div(total_points, schedule.modify_fraction_ / schedule.bucket_divisor_),
        schedule.search_parameters_.num_neighbors_,
        queries,
        schedule.rng_seed_
    );

    auto num_initial_points = div(total_points, schedule.initial_fraction_);
    auto [vectors, indices] = reference.generate(num_initial_points);
    auto tic = svs::lib::now();
    auto bundle = Bundle<Index, T, Q, Distance>{
        .index = init(vectors, indices),
        .reference = std::move(reference),
        .queries = std::move(queries),
        .build_time = 0};
    bundle.build_time = svs::lib::time_difference(tic);
    return bundle;
}

template <
    typename Index,
    typename Reference,
    typename Queries,
    typename Extra = svsbenchmark::Placeholder>
DynamicOperation<Index> measure_state(
    Index& index,
    Reference& reference,
    const Queries& queries,
    size_t queries_in_training_set,
    DynamicOpKind op_kind,
    double op_time,
    const search::SearchParameters& parameters,
    const std::vector<config_type<Index>>& configurations,
    Extra&& extra = {}
) {
    auto tic = svs::lib::now();
    auto gt = svs::data::SimpleData{reference.groundtruth().indices()};
    double groundtruth_time = svs::lib::time_difference(tic);

    // Wait for groundtruth threads to go to sleep.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    size_t num_neighbors = parameters.num_neighbors_;
    auto iso_config =
        search::search_with_config(index, configurations, queries, gt, num_neighbors);

    auto query_set = search::QuerySet(queries, gt, queries_in_training_set);
    auto iso_recall = search::tune_and_search_with_hint(
        index,
        parameters,
        query_set,
        CalibrateContext::TrainingSetTune,
        configurations,
        extra
    );
    return DynamicOperation<Index>(
        op_kind, op_time, groundtruth_time, std::move(iso_recall), std::move(iso_config)
    );
}

template <
    typename Bundle,
    typename Job,
    typename DoCheckpoint,
    typename Extra = svsbenchmark::Placeholder>
toml::table dynamic_test_loop(
    Bundle& bundle, const Job& job, const DoCheckpoint& do_checkpoint, Extra&& extra = {}
) {
    using Index = typename Bundle::index_type;
    using Traits = IndexTraits<Index>;

    const auto& schedule = job.get_dynamic_schedule();
    const auto& search_parameters = schedule.search_parameters_;
    size_t num_neighbors = search_parameters.num_neighbors_;

    auto results = DynamicReport<Job, Index>(job, Traits::name());

    using config_type = typename Traits::config_type;
    auto configs = std::vector<config_type>();

    auto measure_and_checkpoint = [&](DynamicOpKind op_kind, double op_time) {
        auto op = measure_state(
            bundle.index,
            bundle.reference,
            bundle.queries,
            job.queries_in_training_set(),
            op_kind,
            op_time,
            search_parameters,
            configs,
            extra
        );
        results.push_back(std::move(op));
        do_checkpoint(svs::lib::save_to_table(results));
    };

    // Calibrate initial configurations.
    {
        auto gt = svs::data::SimpleData{bundle.reference.groundtruth().indices()};
        auto query_set = search::QuerySet(bundle.queries, gt, bundle.queries.size() / 2);
        for (auto target_recall : search_parameters.target_recalls_) {
            configs.push_back(Traits::calibrate(
                bundle.index,
                query_set.training_set_,
                query_set.training_set_groundtruth_,
                num_neighbors,
                target_recall,
                CalibrateContext::InitialTrainingSet,
                extra
            ));
        }

        measure_and_checkpoint(DynamicOpKind::Initial, bundle.build_time);
    }

    // Main test loop.
    size_t num_points = div(bundle.reference.size(), schedule.modify_fraction_);
    for (size_t i = 0; i < schedule.total_iterations_; ++i) {
        // Add points
        {
            auto [points, ids] = bundle.reference.generate(num_points);
            auto tic = svs::lib::now();
            Traits::add_points(bundle.index, points, ids);
            measure_and_checkpoint(DynamicOpKind::Add, svs::lib::time_difference(tic));
        }

        // Delete Points
        {
            auto points = bundle.reference.get_delete_points(num_points);
            auto tic = svs::lib::now();
            Traits::delete_points(bundle.index, points);
            measure_and_checkpoint(DynamicOpKind::Delete, svs::lib::time_difference(tic));
        }

        // Consolidate
        if ((i + 1) % schedule.cycles_before_cleanup_ == 0) {
            auto tic = svs::lib::now();
            Traits::consolidate(bundle.index);
            measure_and_checkpoint(
                DynamicOpKind::Consolidate, svs::lib::time_difference(tic)
            );
        }
    }
    return svs::lib::save_to_table(results);
}

} // namespace svsbenchmark::build
