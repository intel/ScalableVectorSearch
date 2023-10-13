// svs-benchmark
#include "benchmark.h"

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

// Static
struct SearchParameters {
  public:
    size_t num_neighbors_;
    std::vector<double> target_recalls_;

  public:
    SearchParameters(size_t num_neighbors, std::vector<double> target_recalls)
        : num_neighbors_{num_neighbors}
        , target_recalls_{std::move(target_recalls)} {}

    static SearchParameters example() { return SearchParameters(10, {0.80, 0.85, 0.90}); }

    // Saving and Loading
    static constexpr svs::lib::Version save_version{0, 0, 0};
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            save_version, {SVS_LIST_SAVE_(num_neighbors), SVS_LIST_SAVE_(target_recalls)}
        );
    }

    static SearchParameters
    load(const toml::table& table, const svs::lib::Version& version) {
        if (version != save_version) {
            throw ANNEXCEPTION("Mismatched Version!");
        }

        return SearchParameters(
            SVS_LOAD_MEMBER_AT_(table, num_neighbors),
            SVS_LOAD_MEMBER_AT_(table, target_recalls)
        );
    }
};

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
    SearchParameters search_parameters_;

  public:
    Schedule(
        double initial_fraction,
        double modify_fraction,
        size_t cycles_before_cleanup,
        size_t total_iterations,
        size_t bucket_divisor,
        uint64_t rng_seed,
        SearchParameters search_parameters
    )
        : initial_fraction_{initial_fraction}
        , modify_fraction_{modify_fraction}
        , cycles_before_cleanup_{cycles_before_cleanup}
        , total_iterations_{total_iterations}
        , bucket_divisor_{bucket_divisor}
        , rng_seed_{rng_seed}
        , search_parameters_{std::move(search_parameters)} {}

    static Schedule example() {
        return Schedule(0.75, 0.01, 5, 20, 32, 0xc0ffee, SearchParameters::example());
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
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
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

    static Schedule load(const toml::table& table, const svs::lib::Version& version) {
        if (version != save_version) {
            throw ANNEXCEPTION("Version mismatch when loading Schedule!");
        }

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

// Customize this data structure for the various data set implementations.
template <typename Index> struct IndexTraits;

template <typename Index> using config_type = typename IndexTraits<Index>::config_type;
template <typename Index> using state_type = typename IndexTraits<Index>::state_type;

template <typename Index> struct RunReport {
  public:
    config_type<Index> config_;
    state_type<Index> state_;
    double recall_;
    size_t num_queries_;
    size_t num_neighbors_;
    std::vector<double> latencies_;

  public:
    RunReport(
        const config_type<Index>& config,
        state_type<Index> state,
        double recall,
        size_t num_queries,
        size_t num_neighbors,
        std::vector<double> latencies
    )
        : config_{config}
        , state_{std::move(state)}
        , recall_{recall}
        , num_queries_{num_queries}
        , num_neighbors_{num_neighbors}
        , latencies_{std::move(latencies)} {}

    // Saving
    static constexpr svs::lib::Version save_version{0, 0, 0};
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            save_version,
            {
                SVS_LIST_SAVE_(config),
                SVS_LIST_SAVE_(state),
                SVS_LIST_SAVE_(recall),
                SVS_LIST_SAVE_(num_queries),
                SVS_LIST_SAVE_(num_neighbors),
                SVS_LIST_SAVE_(latencies),
            }
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
    std::vector<RunReport<Index>> iso_recall_{};
    std::vector<RunReport<Index>> iso_config_{};

  public:
    DynamicOperation() = default;

    DynamicOperation(
        DynamicOpKind kind,
        double time,
        double groundtruth_time,
        std::vector<RunReport<Index>> iso_recall,
        std::vector<RunReport<Index>> iso_config
    )
        : kind_{kind}
        , time_{time}
        , groundtruth_time_{groundtruth_time}
        , iso_recall_{std::move(iso_recall)}
        , iso_config_{std::move(iso_config)} {}

    // Saving
    static constexpr svs::lib::Version save_version{0, 0, 0};
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
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
    std::vector<RunReport<Index>> target_configs_;
    std::vector<RunReport<Index>> target_recalls_;

  public:
    StaticReport(
        double build_time,
        const Job& job,
        std::string index_description,
        std::vector<RunReport<Index>> target_configs,
        std::vector<RunReport<Index>> target_recalls
    )
        : build_time_{build_time}
        , timestamp_{std::chrono::system_clock::now()}
        , job_{job}
        , index_description_{std::move(index_description)}
        , target_configs_{std::move(target_configs)}
        , target_recalls_{std::move(target_recalls)} {}

    // Saving.
    static constexpr svs::lib::Version save_version{0, 0, 0};
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
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
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
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

template <typename Index, typename Queries, typename Groundtruth>
RunReport<Index> search_with_config(
    Index& index,
    const config_type<Index>& config,
    const Queries& queries,
    const Groundtruth& groundtruth,
    size_t num_neighbors
) {
    using Traits = IndexTraits<Index>;
    auto latencies = std::vector<double>();

    auto tic = svs::lib::now();
    auto results = Traits::search(index, queries, num_neighbors, config);
    latencies.push_back(svs::lib::time_difference(tic));

    for (size_t i = 0; i < 5; ++i) {
        tic = svs::lib::now();
        results = Traits::search(index, queries, num_neighbors, config);
        latencies.push_back(svs::lib::time_difference(tic));
    }
    double recall = svs::k_recall_at_n(groundtruth, results, num_neighbors, num_neighbors);
    return RunReport<Index>(
        config,
        Traits::report_state(index),
        recall,
        queries.size(),
        num_neighbors,
        std::move(latencies)
    );
}

template <typename Index, typename Queries, typename Groundtruth>
std::vector<RunReport<Index>> search_with_config(
    Index& index,
    const std::vector<config_type<Index>>& configs,
    const Queries& queries,
    const Groundtruth& groundtruth,
    size_t num_neighbors
) {
    auto reports = std::vector<RunReport<Index>>();
    for (const auto& config : configs) {
        reports.push_back(
            search_with_config(index, config, queries, groundtruth, num_neighbors)
        );
    }
    return reports;
}

template <typename Index, typename Queries, typename Groundtruth>
std::vector<RunReport<Index>> tune_and_search(
    Index& index,
    const SearchParameters& parameters,
    const Queries& queries,
    const Groundtruth& groundtruth
) {
    auto reports = std::vector<RunReport<Index>>();
    size_t num_neighbors = parameters.num_neighbors_;
    for (auto target_recall : parameters.target_recalls_) {
        auto config = IndexTraits<Index>::calibrate(
            index, queries, groundtruth, num_neighbors, target_recall
        );
        reports.push_back(
            search_with_config(index, config, queries, groundtruth, num_neighbors)
        );
    }
    return reports;
}

template <typename Index, typename Reference, typename Queries>
DynamicOperation<Index> measure_state(
    Index& index,
    Reference& reference,
    const Queries& queries,
    DynamicOpKind op_kind,
    double op_time,
    const SearchParameters& parameters,
    const std::vector<config_type<Index>>& configurations
) {
    auto tic = svs::lib::now();
    auto gt = reference.groundtruth();
    double groundtruth_time = svs::lib::time_difference(tic);

    // Wait for groundtruth threads to go to sleep.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    size_t num_neighbors = parameters.num_neighbors_;
    auto iso_config = search_with_config(index, configurations, queries, gt, num_neighbors);
    auto iso_recall = tune_and_search(index, parameters, queries, gt);
    return DynamicOperation<Index>(
        op_kind, op_time, groundtruth_time, std::move(iso_recall), std::move(iso_config)
    );
}

template <typename Bundle, typename Job, typename DoCheckpoint>
toml::table
dynamic_test_loop(Bundle& bundle, const Job& job, const DoCheckpoint& do_checkpoint) {
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
            op_kind,
            op_time,
            search_parameters,
            configs
        );
        results.push_back(std::move(op));
        do_checkpoint(svs::lib::save_to_table(results));
    };

    // Calibrate initial configurations.
    {
        auto gt = bundle.reference.groundtruth();
        for (auto target_recall : search_parameters.target_recalls_) {
            configs.push_back(Traits::calibrate(
                bundle.index, bundle.queries, gt, num_neighbors, target_recall
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

template <
    typename Index,
    typename Job,
    typename Queries,
    typename Groundtruth,
    typename DoCheckpoint>
toml::table static_test(
    Index& index,
    const Job& job,
    const Queries& queries,
    const Groundtruth& groundtruth,
    double build_time,
    const DoCheckpoint& do_checkpoint
) {
    const auto& search_parameters = job.get_search_parameters();
    auto target_configs = search_with_config(
        index,
        job.get_search_configs(),
        queries,
        groundtruth,
        search_parameters.num_neighbors_
    );

    auto target_recalls = tune_and_search(index, search_parameters, queries, groundtruth);
    auto results = StaticReport<Job, Index>(
        build_time,
        job,
        IndexTraits<Index>::name(),
        std::move(target_configs),
        std::move(target_recalls)
    );

    auto table = svs::lib::save(results);
    do_checkpoint(table);
    return table;
}
} // namespace svsbenchmark::build
