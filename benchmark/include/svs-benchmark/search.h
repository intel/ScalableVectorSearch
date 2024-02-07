#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/index_traits.h"

// svs
#include "svs/core/data.h"
#include "svs/core/data/view.h"
#include "svs/core/recall.h"
#include "svs/lib/saveload.h"

// stl
#include <vector>

namespace svsbenchmark::search {

/////
///// Classes
/////

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

template <typename Index> struct RunReport {
  public:
    using config_type = svsbenchmark::config_type<Index>;
    using state_type = svsbenchmark::state_type<Index>;

  public:
    config_type config_;
    state_type state_;
    double recall_;
    size_t num_queries_;
    size_t num_neighbors_;
    std::vector<double> latencies_;

  public:
    RunReport(
        const config_type& config,
        state_type state,
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

// Split Queries
template <typename T, std::integral I> struct QuerySet {
  public:
    using dataset_type = svs::data::SimpleData<T>;
    using groundtruth_type = svs::data::SimpleData<I>;

  public:
    dataset_type training_set_;
    groundtruth_type training_set_groundtruth_;
    dataset_type test_set_;
    groundtruth_type test_set_groundtruth_;

  public:
    // Construct from the individual components.
    QuerySet(
        dataset_type&& training_set,
        groundtruth_type&& training_set_groundtruth,
        dataset_type&& test_set,
        groundtruth_type&& test_set_groundtruth
    )
        : training_set_{std::move(training_set)}
        , training_set_groundtruth_{std::move(training_set_groundtruth)}
        , test_set_{std::move(test_set)}
        , test_set_groundtruth_{std::move(test_set_groundtruth)} {
        assert(training_set_.size() == training_set_groundtruth.size());
        assert(test_set_.size() == test_set_groundtruth.size());
        assert(training_set_.dimensions() == test_set_.dimensions());
    }

    QuerySet(
        const dataset_type& queries,
        const groundtruth_type& groundtruth,
        size_t number_of_training_elements
    ) {
        assert(queries.size() == groundtruth.size());
        if (number_of_training_elements >= queries.size()) {
            throw ANNEXCEPTION(
                "Number of elements to pull out into the training ({}) is greater than the "
                "total query set size of {}.",
                number_of_training_elements,
                queries.size()
            );
        }
        auto training_range = svs::threads::UnitRange(0, number_of_training_elements);
        training_set_ = dataset_type{training_range.size(), queries.dimensions()};
        training_set_groundtruth_ =
            groundtruth_type{training_range.size(), groundtruth.dimensions()};
        svs::data::copy(svs::data::make_view(queries, training_range), training_set_);
        svs::data::copy(
            svs::data::make_view(groundtruth, training_range), training_set_groundtruth_
        );

        auto test_range =
            svs::threads::UnitRange(number_of_training_elements, queries.size());
        test_set_ = dataset_type{test_range.size(), queries.dimensions()};
        test_set_groundtruth_ =
            groundtruth_type{test_range.size(), groundtruth.dimensions()};

        svs::data::copy(svs::data::make_view(queries, test_range), test_set_);
        svs::data::copy(
            svs::data::make_view(groundtruth, test_range), test_set_groundtruth_
        );
    }
};

template <
    typename Index,
    typename Queries,
    typename Groundtruth,
    typename Extra = svsbenchmark::Placeholder>
RunReport<Index> search_with_config(
    Index& index,
    const svsbenchmark::config_type<Index>& config,
    const Queries& queries,
    const Groundtruth& groundtruth,
    size_t num_neighbors
) {
    using Traits = svsbenchmark::IndexTraits<Index>;
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

template <
    typename Index,
    typename Queries,
    typename Groundtruth,
    typename Extra = svsbenchmark::Placeholder>
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

template <
    typename Index,
    typename QueryType,
    typename I,
    typename Extra = svsbenchmark::Placeholder>
std::vector<RunReport<Index>> tune_and_search(
    Index& index,
    const SearchParameters& parameters,
    const QuerySet<QueryType, I>& query_set,
    svsbenchmark::CalibrateContext context,
    Extra&& extra = {}
) {
    auto reports = std::vector<RunReport<Index>>();
    size_t num_neighbors = parameters.num_neighbors_;
    for (auto target_recall : parameters.target_recalls_) {
        // Do any necessary calibration on the training set.
        auto config = IndexTraits<Index>::calibrate(
            index,
            query_set.training_set_,
            query_set.training_set_groundtruth_,
            num_neighbors,
            target_recall,
            context,
            extra
        );

        // Refinement on the test set.
        // It is expected that the calibration routine does the minimal required to
        // achieve the desired recall on the test set.
        //
        // Feed forward the configuration derived on the training set.
        config = IndexTraits<Index>::calibrate_with_hint(
            index,
            query_set.test_set_,
            query_set.test_set_groundtruth_,
            num_neighbors,
            target_recall,
            svsbenchmark::CalibrateContext::TestSetTune,
            config,
            extra
        );

        reports.push_back(search_with_config(
            index,
            config,
            query_set.test_set_,
            query_set.test_set_groundtruth_,
            num_neighbors
        ));
    }
    return reports;
}

template <
    typename Index,
    typename QueryType,
    typename I,
    typename Extra = svsbenchmark::Placeholder>
std::vector<RunReport<Index>> tune_and_search_with_hint(
    Index& index,
    const SearchParameters& parameters,
    const QuerySet<QueryType, I>& query_set,
    svsbenchmark::CalibrateContext context,
    const std::vector<config_type<Index>>& configurations,
    Extra&& extra = {}
) {
    auto reports = std::vector<RunReport<Index>>();
    size_t num_neighbors = parameters.num_neighbors_;
    const auto& target_recalls = parameters.target_recalls_;
    if (target_recalls.size() != configurations.size()) {
        throw ANNEXCEPTION(
            "Number of target recalls and number of configurations do not match!"
        );
    }

    for (size_t i = 0, imax = target_recalls.size(); i < imax; ++i) {
        auto target_recall = target_recalls[i];
        const auto& hint = configurations.at(i);
        auto config = IndexTraits<Index>::calibrate_with_hint(
            index,
            query_set.training_set_,
            query_set.training_set_groundtruth_,
            num_neighbors,
            target_recall,
            context,
            hint,
            extra
        );

        // Refinement on the test set.
        // It is expected that the calibration routine does the minimal required to
        // achieve the desired recall on the test set.
        //
        // Feed forward the configuration derived on the training set.
        config = IndexTraits<Index>::calibrate_with_hint(
            index,
            query_set.test_set_,
            query_set.test_set_groundtruth_,
            num_neighbors,
            target_recall,
            svsbenchmark::CalibrateContext::TestSetTune,
            config,
            extra
        );

        reports.push_back(search_with_config(
            index,
            config,
            query_set.test_set_,
            query_set.test_set_groundtruth_,
            num_neighbors
        ));
    }
    return reports;
}

template <typename Job, typename Index, typename Mixin> struct SearchReport {
  public:
    Mixin additional_;
    std::chrono::time_point<std::chrono::system_clock> timestamp_;
    // The incoming job that these results are for.
    Job job_;
    // A descriptive name for the index.
    std::string index_description_;
    // Results for pre-generated configurations.
    std::vector<RunReport<Index>> target_configs_;
    std::vector<RunReport<Index>> target_recalls_;

  public:
    SearchReport(
        const Mixin& additional,
        const Job& job,
        std::string index_description,
        std::vector<search::RunReport<Index>> target_configs,
        std::vector<search::RunReport<Index>> target_recalls
    )
        : additional_{additional}
        , timestamp_{std::chrono::system_clock::now()}
        , job_{job}
        , index_description_{std::move(index_description)}
        , target_configs_{std::move(target_configs)}
        , target_recalls_{std::move(target_recalls)} {}

    /// Saving.
    // History
    // * v0.0.0: Initial Version
    // * v0.0.1: Replaced field `build_time_` parameterized `additional_`.
    //   Reason: Form of the class was general enough to be reused by both static build and
    //   pure search. Replacing this field is a better match for this new use case.
    static constexpr svs::lib::Version save_version{0, 0, 1};
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            save_version,
            {SVS_LIST_SAVE_(additional),
             SVS_LIST_SAVE_(timestamp),
             SVS_LIST_SAVE_(job),
             SVS_LIST_SAVE_(index_description),
             SVS_LIST_SAVE_(target_configs),
             SVS_LIST_SAVE_(target_recalls)}
        );
    }
};

template <
    typename Index,
    typename Job,
    typename QueryType,
    typename I,
    typename Mixin,
    typename Extra = svsbenchmark::Placeholder>
SearchReport<Job, Index, Mixin> run_search(
    Index& index,
    const Job& job,
    const QuerySet<QueryType, I>& query_set,
    const Mixin& additional,
    Extra&& extra = {}
) {
    const auto& search_parameters = job.get_search_parameters();
    auto target_configs = search::search_with_config(
        index,
        job.get_search_configs(),
        query_set.test_set_,
        query_set.test_set_groundtruth_,
        search_parameters.num_neighbors_
    );

    auto target_recalls = search::tune_and_search(
        index, search_parameters, query_set, CalibrateContext::InitialTrainingSet, extra
    );

    return SearchReport<Job, Index, Mixin>(
        additional,
        job,
        IndexTraits<Index>::name(),
        std::move(target_configs),
        std::move(target_recalls)
    );
}

} // namespace svsbenchmark::search
