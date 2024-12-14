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

#include "svs-benchmark/vamana/static_traits.h"

// svs
#include "svs/core/distance.h"
#include "svs/index/vamana/iterator.h"

namespace svsbenchmark::vamana {

/// Pre-configuration for the linear schedule.
struct LinearSchedulePrototype {
    size_t scale_search_window_;
    size_t scale_buffer_capacity_;
    int64_t enable_filter_after_;
    size_t batch_size_start_;
    size_t scale_batch_size_;
    // Whether search should be restarted on every iteration.
    bool restart_searches_;

    ///// Saving and Loading.
    static constexpr std::string_view serialization_schema =
        "svsbench_vamana_iter_schedule";
    static constexpr svs::lib::Version save_version{0, 0, 0};

    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable{
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(scale_search_window),
             SVS_LIST_SAVE_(scale_buffer_capacity),
             SVS_LIST_SAVE_(enable_filter_after),
             SVS_LIST_SAVE_(batch_size_start),
             SVS_LIST_SAVE_(scale_batch_size),
             SVS_LIST_SAVE_(restart_searches)}};
    }

    static LinearSchedulePrototype load(const svs::lib::ContextFreeLoadTable& table) {
        return LinearSchedulePrototype{
            SVS_LOAD_MEMBER_AT_(table, scale_search_window),
            SVS_LOAD_MEMBER_AT_(table, scale_buffer_capacity),
            SVS_LOAD_MEMBER_AT_(table, enable_filter_after),
            SVS_LOAD_MEMBER_AT_(table, batch_size_start),
            SVS_LOAD_MEMBER_AT_(table, scale_batch_size),
            SVS_LOAD_MEMBER_AT_(table, restart_searches)};
    }

    // Return several representative examples for the schedule.
    static std::vector<LinearSchedulePrototype> examples() {
        return {{10, 20, -1, 10, 0, false}, {10, 10, 3, 10, 5, false}};
    }

    // Should search be restarted from scratch every iteration.
    bool restart_every_iteration() const { return restart_searches_; }

    // Materialize an actual schedule given a set of base parameters.
    // NOTE: This does not propagate the `restart_searches_` flag.
    svs::index::vamana::LinearSchedule
    materialize(const svs::index::vamana::VamanaSearchParameters& sp) const {
        return svs::index::vamana::LinearSchedule{
            sp,
            svs::lib::narrow<uint16_t>(scale_search_window_),
            svs::lib::narrow<uint16_t>(scale_buffer_capacity_),
            svs::lib::narrow<int16_t>(enable_filter_after_),
            svs::lib::narrow<uint16_t>(batch_size_start_),
            svs::lib::narrow<uint16_t>(scale_batch_size_)};
    }
};

struct IteratorSearchParameters {
  public:
    ///// Members
    // The schedules to try.
    std::vector<LinearSchedulePrototype> schedules_;
    // target recalls relative to base number of neighbors.
    std::vector<svs::lib::Percent> target_recalls_;
    // The number of batches to yield.
    size_t num_batches_;
    // Since iterator search is performed on a single thread, this subsample parameter
    // provides a mechanism to operate on a reduced number of queries to reduce test time.
    size_t query_subsample_;

    ///// Saving and Loading.
    static constexpr std::string_view serialization_schema = "svsbenchamrk_isp";
    static constexpr svs::lib::Version save_version{0, 0, 0};

    static IteratorSearchParameters example() {
        return IteratorSearchParameters{
            .schedules_ = LinearSchedulePrototype::examples(),
            .target_recalls_ = {svs::lib::Percent(0.9)},
            .num_batches_ = 5,
            .query_subsample_ = 10,
        };
    }

    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable{
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(schedules),
             SVS_LIST_SAVE_(target_recalls),
             SVS_LIST_SAVE_(num_batches),
             SVS_LIST_SAVE_(query_subsample)}};
    }

    static IteratorSearchParameters load(const svs::lib::ContextFreeLoadTable& table) {
        return IteratorSearchParameters{
            SVS_LOAD_MEMBER_AT_(table, schedules),
            SVS_LOAD_MEMBER_AT_(table, target_recalls),
            SVS_LOAD_MEMBER_AT_(table, num_batches),
            SVS_LOAD_MEMBER_AT_(table, query_subsample)};
    }
};

inline constexpr std::string_view iterator_benchmark_name() { return "vamana_iterator_v1"; }

std::unique_ptr<Benchmark> iterator_benchmark();

// Perform a check size-reduction on the given queries and groundtruth.
//
// To maintain an appropriate split for training and test data, argument ``count`` must be
template <typename Q, typename I>
svsbenchmark::search::QuerySet<Q, I> subsample(
    const svs::data::ConstSimpleDataView<Q>& queries,
    const svs::data::ConstSimpleDataView<I>& groundtruth,
    size_t count
) {
    if (2 * count > queries.size()) {
        throw ANNEXCEPTION(
            "Subsample amount {} must be at most half of the total number of queries ({}) "
            "to provide an adequate training/test split.",
            count,
            queries.size()
        );
    }

    // TODO: Provide a method to resize views.
    return svsbenchmark::search::QuerySet<Q, I>{
        svs::data::ConstSimpleDataView<Q>{queries.data(), 2 * count, queries.dimensions()},
        svs::data::ConstSimpleDataView<I>{
            groundtruth.data(), 2 * count, groundtruth.dimensions()},
        count};
}

struct IteratorSearch {
  public:
    ///// Members
    svsbenchmark::Dataset dataset_;
    std::filesystem::path config_;
    std::filesystem::path graph_;
    std::filesystem::path data_;
    std::filesystem::path queries_;
    std::filesystem::path groundtruth_;
    svs::DistanceType distance_;
    IteratorSearchParameters parameters_;
    // Types of the queries and source datasets.
    svs::DataType query_type_;
    Extent ndims_;

    static IteratorSearch example() {
        return IteratorSearch{
            .dataset_ = Dataset::example(),
            .config_ = "path/to/index/config",
            .graph_ = "path/to/graph",
            .data_ = "path/to/data",
            .queries_ = "path/to/queries",
            .groundtruth_ = "path/to/groundtruth",
            .distance_ = svs::DistanceType::L2,
            .parameters_ = IteratorSearchParameters::example(),
            .query_type_ = svs::DataType::float32,
            .ndims_ = Extent{svs::Dynamic}};
    }

    // Dispatch invocation.
    template <typename F> auto invoke(F&& f, const Checkpoint& checkpointer) const {
        return f(dataset_, query_type_, distance_, ndims_, checkpointer, *this);
    }

    ///// Save/Load
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "svsbenchmark_vamana_iterator";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(dataset),
             SVS_LIST_SAVE_(config),
             SVS_LIST_SAVE_(graph),
             SVS_LIST_SAVE_(data),
             SVS_LIST_SAVE_(queries),
             SVS_LIST_SAVE_(groundtruth),
             SVS_LIST_SAVE_(distance),
             SVS_LIST_SAVE_(parameters),
             SVS_LIST_SAVE_(query_type),
             SVS_LIST_SAVE_(ndims)}
        );
    }

    static IteratorSearch load(
        const svs::lib::ContextFreeLoadTable& table,
        const std::optional<std::filesystem::path>& root = {}
    ) {
        return IteratorSearch{
            .dataset_ = SVS_LOAD_MEMBER_AT_(table, dataset, root),
            .config_ = svsbenchmark::extract_filename(table, "config", root),
            .graph_ = svsbenchmark::extract_filename(table, "graph", root),
            .data_ = svsbenchmark::extract_filename(table, "data", root),
            .queries_ = svsbenchmark::extract_filename(table, "queries", root),
            .groundtruth_ = svsbenchmark::extract_filename(table, "groundtruth", root),
            .distance_ = SVS_LOAD_MEMBER_AT_(table, distance),
            .parameters_ = SVS_LOAD_MEMBER_AT_(table, parameters),
            .query_type_ = SVS_LOAD_MEMBER_AT_(table, query_type),
            .ndims_ = SVS_LOAD_MEMBER_AT_(table, ndims)};
    }
};

using IteratorDispatcher = svs::lib::Dispatcher<
    toml::table,
    Dataset,
    svs::DataType,
    svs::DistanceType,
    Extent,
    const svsbenchmark::Checkpoint&,
    const IteratorSearch&>;

/////
///// Implementation
/////

struct YieldedResult {
    // The invocation number of this result.
    size_t iteration_;
    // The number of neighbors yielded for this result.
    size_t yielded_;
    // The number of results yielded so far in total.
    size_t total_yielded_;
    // The `total_yielded_` recall at `total_yielded_`.
    double total_recall_;
    // Execution time for the most recent batch of results.
    double execution_time_;

    ///// Saving.
    static constexpr std::string_view serialization_schema = "svsbenchmark_yielded_result";
    static constexpr svs::lib::Version save_version = svs::lib::Version(0, 0, 0);

    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable{
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(iteration),
             SVS_LIST_SAVE_(yielded),
             SVS_LIST_SAVE_(total_yielded),
             SVS_LIST_SAVE_(total_recall),
             SVS_LIST_SAVE_(execution_time)}};
    }
};

// TODO: Make the dependence on `Report` looser.
template <typename Index> struct QueryIteratorResult {
    LinearSchedulePrototype schedule_;
    size_t num_batches_;
    double target_recall_;
    search::RunReport<Index> report_;
    // The search parameters used for each iteration.
    // Must be the same for all queries in the batch.
    std::vector<svs::index::vamana::VamanaSearchParameters> iteration_parameters_;
    // Outer vector: Results for each query.
    // Inner vector: Results within a query.
    std::vector<std::vector<YieldedResult>> results_;

    ///// Constructor
    QueryIteratorResult(
        const LinearSchedulePrototype& schedule,
        double target_recall,
        search::RunReport<Index> report,
        std::vector<svs::index::vamana::VamanaSearchParameters> iteration_parameters,
        std::vector<std::vector<YieldedResult>> results
    )
        : schedule_{schedule}
        , num_batches_{iteration_parameters.size()}
        , target_recall_{target_recall}
        , report_{std::move(report)}
        , iteration_parameters_{std::move(iteration_parameters)}
        , results_{std::move(results)} {
        // Ensure all the yielded results have the correct size.
        for (size_t i = 0, imax = results_.size(); i < imax; ++i) {
            size_t these_batches = results_.at(i).size();
            if (results_.at(i).size() != num_batches_) {
                throw ANNEXCEPTION(
                    "Yielded result {} has {} batches when {} were expected.",
                    i,
                    these_batches,
                    num_batches_
                );
            }
        }
    }

    ///// Saving.
    static constexpr std::string_view serialization_schema = "svsbenchmark_iterator_result";
    static constexpr svs::lib::Version save_version = svs::lib::Version(0, 0, 0);

    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable{
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(schedule),
             SVS_LIST_SAVE_(num_batches),
             SVS_LIST_SAVE_(target_recall),
             SVS_LIST_SAVE_(report),
             SVS_LIST_SAVE_(iteration_parameters),
             SVS_LIST_SAVE_(results)}};
    }
};

template <
    typename Index,
    typename QueryType,
    typename I,
    typename MakeIterator,
    typename DoCheckpoint,
    typename Extra = svsbenchmark::Placeholder>
std::vector<QueryIteratorResult<Index>> tune_and_search_iterator(
    Index& index,
    const vamana::IteratorSearchParameters& parameters,
    const svsbenchmark::search::QuerySet<QueryType, I>& query_set,
    svsbenchmark::CalibrateContext context,
    MakeIterator&& make_iterator,
    const DoCheckpoint& do_checkpoint,
    Extra&& extra = {}
) {
    using traits = IndexTraits<Index>;

    const auto& query_test = query_set.test_set_;
    const auto& groundtruth_test = query_set.test_set_groundtruth_;

    const size_t nqueries = query_test.size();

    // Loop over each batchsize.
    auto query_iterator_results = std::vector<QueryIteratorResult<Index>>{};
    for (const auto& schedule : parameters.schedules_) {
        auto initial_batch_size = schedule.batch_size_start_;
        for (auto target_recall : parameters.target_recalls_) {
            // Calibrate the index for the given recall.
            auto config = traits::calibrate(
                index,
                query_set.training_set_,
                query_set.training_set_groundtruth_,
                initial_batch_size,
                target_recall.value(),
                context,
                extra
            );

            // Refine on the test set.
            config = traits::calibrate_with_hint(
                index,
                query_set.test_set_,
                query_set.test_set_groundtruth_,
                initial_batch_size,
                target_recall.value(),
                svsbenchmark::CalibrateContext::TestSetTune,
                config,
                extra
            );

            // Now we have a calibrated configuration - obtain a baseline report for
            // searching with this batchsize.
            auto report = svsbenchmark::search::search_with_config(
                index, config, query_test, groundtruth_test, initial_batch_size
            );

            // `resuilt_buffer`: All results that have been returned by the iterator.
            auto result_buffer = std::vector<size_t>();

            // Helper lambda to record statistics post-run.
            // This is needed since creation of the iterator starts graph search.
            //
            // TODO: It might be worth exposing an API on the iterator to disable eager
            //       searching in the constructor.
            auto tally = [&result_buffer, &groundtruth_test](
                             const auto& iterator,
                             size_t query_index,
                             size_t iteration,
                             double execution_time
                         ) {
                // Populate most recent results.
                auto iterator_results = iterator.results();
                std::transform(
                    iterator_results.begin(),
                    iterator_results.end(),
                    std::back_inserter(result_buffer),
                    [](auto neighbor) { return neighbor.id(); }
                );

                // Compute local recalls.
                size_t total_yielded = result_buffer.size();
                if (groundtruth_test.dimensions() < total_yielded) {
                    throw ANNEXCEPTION(
                        "Groundtruth with {} entries has insufficient entries to compute "
                        "recall for {} neighbors!",
                        groundtruth_test.dimensions(),
                        result_buffer.size()
                    );
                }
                auto count = svs::lib::count_intersect(
                    result_buffer,
                    groundtruth_test.get_datum(query_index).first(total_yielded)
                );

                double recall = svs::lib::narrow_cast<double>(count) /
                                svs::lib::narrow_cast<double>(total_yielded);

                return YieldedResult{
                    .iteration_ = iteration,
                    .yielded_ = iterator.size(),
                    .total_yielded_ = total_yielded,
                    .total_recall_ = recall,
                    .execution_time_ = execution_time};
            };

            // Now that we have the baseline, obtain iterator based results.
            auto iteration_parameters =
                std::vector<svs::index::vamana::VamanaSearchParameters>();
            auto yielded_results = std::vector<std::vector<YieldedResult>>();
            for (size_t i = 0; i < nqueries; ++i) {
                result_buffer.clear();
                auto& timings_for_this_query = yielded_results.emplace_back();
                auto&& query = query_test.get_datum(i);

                // The first call to `iterator` kick-starts graph search.
                auto tic = svs::lib::now();
                auto iterator = make_iterator(index, query, config, schedule);
                auto elapsed = svs::lib::time_difference(tic);
                if (i == 0) {
                    iteration_parameters.push_back(iterator.parameters_for_current_batch());
                }

                timings_for_this_query.push_back(tally(iterator, i, 0, elapsed));
                for (size_t j = 0; j < parameters.num_batches_; ++j) {
                    // If requested by the parent schedule, reset search for this
                    // iteration.
                    if (schedule.restart_every_iteration()) {
                        iterator.restart_next_search();
                    }

                    tic = svs::lib::now();
                    iterator.next();
                    elapsed = svs::lib::time_difference(tic);
                    timings_for_this_query.push_back(tally(iterator, i, j + 1, elapsed));
                    if (i == 0) {
                        iteration_parameters.push_back(
                            iterator.parameters_for_current_batch()
                        );
                    }
                }
            }

            // Finish up summarizing these results.
            query_iterator_results.emplace_back(
                schedule,
                target_recall.value(),
                std::move(report),
                std::move(iteration_parameters),
                std::move(yielded_results)
            );
            do_checkpoint(query_iterator_results);
        }
    }
    return query_iterator_results;
}

template <typename Index, typename QueryType, typename I>
toml::table tune_and_search_iterator(
    Index& index,
    const IteratorSearch& job,
    const svsbenchmark::search::QuerySet<QueryType, I>& query_set,
    const svsbenchmark::Checkpoint& checkpointer
) {
    // pre-lower the IteratorSearch for checkpointing purposes.
    auto toml_base = svs::lib::save_to_table(job);

    // Use a helper lambda to save the results.
    // This lambda can be reused when generating the final ``toml::table`` to ensure the
    // layout is the same.
    auto serialize_results = [&](const std::vector<QueryIteratorResult<Index>>&
                                     results_so_far) {
        return toml::table{{"job", toml_base}, {"results", svs::lib::save(results_so_far)}};
    };

    auto do_checkpoint = [&](const std::vector<QueryIteratorResult<Index>>& results_so_far
                         ) {
        checkpointer.checkpoint(
            serialize_results(results_so_far), iterator_benchmark_name()
        );
    };

    auto results = tune_and_search_iterator(
        index,
        job.parameters_,
        query_set,
        svsbenchmark::CalibrateContext::InitialTrainingSet,
        [](const auto& index, const auto& query, const auto& config, const auto& schedule) {
            return index.batch_iterator(query, schedule.materialize(config));
        },
        do_checkpoint,
        svsbenchmark::IndexTraits<Index>::regression_optimization()
    );
    return serialize_results(results);
}

} // namespace svsbenchmark::vamana
