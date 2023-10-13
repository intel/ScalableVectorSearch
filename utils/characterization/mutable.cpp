/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// svs
#include "svs/core/medioid.h"
#include "svs/core/recall.h"
#include "svs/index/flat/flat.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/lib/timing.h"

#include "svs/misc/dynamic_helper.h"

// svsmain
#include "svsmain.h"

// stl
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

using Idx = uint32_t;
using Eltype = svs::Float16;
using QueryEltype = float;
using Distance = svs::distance::DistanceL2;
const size_t N = 96;

const size_t NUM_NEIGHBORS = 10;
const double TARGET_RECALL = 0.95;
const float ALPHA = 1.2;

///
/// Utility Methods
///

template <std::integral I> I div(I i, float fraction) {
    return svs::lib::narrow<I>(std::floor(svs::lib::narrow<float>(i) * fraction));
}

template <typename... Args> std::string stringify(Args&&... args) {
    std::ostringstream stream{};
    ((stream << args), ...);
    return stream.str();
}

///
/// @brief Compute the window size required to achieve the desired recall.
///
template <typename MutableIndex, typename Groundtruth, typename Queries>
size_t find_windowsize(
    MutableIndex& index,
    const Groundtruth& groundtruth,
    const Queries& queries,
    double target_recall = TARGET_RECALL,
    size_t window_lower = NUM_NEIGHBORS,
    size_t window_upper = 1000
) {
    auto range = svs::threads::UnitRange<size_t>(window_lower, window_upper);
    return *std::lower_bound(
        range.begin(),
        range.end(),
        target_recall,
        [&](size_t window_size, double recall) {
            index.set_search_window_size(window_size);
            auto result = index.search(queries, NUM_NEIGHBORS);
            auto this_recall = svs::k_recall_at_n(groundtruth, result);
            return this_recall < recall;
        }
    );
}

///
/// @brief A report regarding a mutating operation.
///
struct Report {
    template <typename... Args>
    Report(
        double operation_time,
        double groundtruth_time,
        double search_time,
        double recall,
        Args&&... args
    )
        : operation_time_{operation_time}
        , groundtruth_time_{groundtruth_time}
        , search_time_{search_time}
        , recall_{recall}
        , message_{stringify(std::forward<Args>(args)...)} {}

    Report(
        double operation_time,
        double groundtruth_time,
        double search_time,
        double recall,
        std::string message
    )
        : operation_time_{operation_time}
        , groundtruth_time_{groundtruth_time}
        , search_time_{search_time}
        , recall_{recall}
        , message_{std::move(message)} {}

    ///// Members
    double operation_time_;
    double groundtruth_time_;
    double search_time_;
    double recall_;
    std::string message_;
};

std::ostream& operator<<(std::ostream& stream, const Report& report) {
    stream << "[" << report.message_ << "] -- {"
           << "operation: " << report.operation_time_
           << ", groundtruth: " << report.groundtruth_time_
           << ", search: " << report.search_time_ << ", recall: " << report.recall_ << "}";
    return stream;
}

///
/// Reference Dataset.
///

template <typename MutableIndex, typename Queries>
void do_check(
    MutableIndex& index,
    svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>& reference,
    const Queries& queries,
    double operation_time,
    std::string message,
    bool calibrate = false
) {
    // Compute groundtruth
    auto tic = svs::lib::now();
    auto gt = reference.groundtruth();
    double groundtruth_time = svs::lib::time_difference(tic);

    if (calibrate) {
        size_t window_size = find_windowsize(index, gt, queries);
        index.set_search_window_size(window_size);
    }

    // Run search
    tic = svs::lib::now();
    auto result = index.search(queries, NUM_NEIGHBORS);
    double search_time = svs::lib::time_difference(tic);

    // Extra ID checks
    reference.check_ids(result);
    reference.check_equal_ids(index);

    // compute recall
    double recall = svs::k_recall_at_n(gt, result, NUM_NEIGHBORS, NUM_NEIGHBORS);

    // Report the calibrated search window size if we calibrated this round.
    if (calibrate) {
        message += stringify(" - Calibrate window size: ", index.get_search_window_size());
    }

    std::cout
        << Report(operation_time, groundtruth_time, search_time, recall, std::move(message))
        << '\n';
}

///
/// Main Loop.
///

template <typename MutableIndex, typename Queries>
void test_loop(
    MutableIndex& index,
    svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>& reference,
    const Queries& queries,
    size_t num_points,
    size_t consolidate_every,
    size_t iterations
) {
    size_t consolidate_count = 0;
    for (size_t i = 0; i < iterations; ++i) {
        // Add Points
        {
            auto [points, time] = reference.add_points(index, num_points);
            index.debug_check_invariants(true);
            do_check(index, reference, queries, time, stringify("add ", points, " points"));
        }

        // Delete Points
        {
            auto [points, time] = reference.delete_points(index, num_points);
            index.debug_check_invariants(true);
            do_check(
                index, reference, queries, time, stringify("delete ", points, " points")
            );
        }

        // Maybe consolidate.
        ++consolidate_count;
        if (consolidate_count == consolidate_every) {
            auto tic = svs::lib::now();
            index.consolidate();
            double diff = svs::lib::time_difference(tic);
            index.debug_check_invariants(false);
            do_check(index, reference, queries, diff, "consolidate");
            consolidate_count = 0;

            // Compact
            tic = svs::lib::now();
            // Use a batchsize smaller than the whole dataset to ensure that the compaction
            // algorithm correctly handles this case.
            index.compact(reference.valid() / 10);
            diff = svs::lib::time_difference(tic);
            index.debug_check_invariants(false);
            do_check(index, reference, queries, diff, "compact");
        }
    }
}

int svs_main(std::vector<std::string> args) {
    const auto& data_path = args.at(1);
    const auto& query_path = args.at(2);
    auto modify_fraction = std::stof(args.at(3));
    auto initial_fraction = std::stof(args.at(4));
    auto num_threads = std::stoull(args.at(5));

    // Set hyper parameters here
    const size_t max_degree = 64;

    // Begin testing logic.
    if (modify_fraction < 0 || modify_fraction > 1) {
        throw ANNEXCEPTION(
            "Modify percent must be between 0 and 1. Instead, got ", modify_fraction, '!'
        );
    }

    if (initial_fraction < 0 || initial_fraction > 1) {
        throw ANNEXCEPTION(
            "Initial percent must be between 0 and 1. Instead, got ", initial_fraction, '!'
        );
    }

    // Load the base dataset and queries.
    auto queries = svs::VectorDataLoader<QueryEltype>(query_path).load();
    auto data = svs::VectorDataLoader<Eltype, N>(data_path).load();
    auto num_points = data.size();

    auto reference = svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>(
        std::move(data),
        Distance(),
        num_threads,
        div(num_points, 0.125 * modify_fraction),
        NUM_NEIGHBORS,
        queries
    );
    auto num_indices_to_add = div(reference.size(), initial_fraction);
    std::cout << "Initializing with " << num_indices_to_add << " entries!\n";

    // Construct a blocked dataset consisting of 50% of the base dataset.
    auto data_mutable = svs::data::BlockedData<Eltype, N>(num_indices_to_add, N);
    std::vector<Idx> initial_indices{};
    {
        auto [vectors, indices] = reference.generate(num_indices_to_add);
        // Copy assign ``initial_indices``
        initial_indices = indices;
        for (size_t i = 0; i < num_indices_to_add; ++i) {
            data_mutable.set_datum(i, vectors.get_datum(i));
        }
    }

    svs::index::vamana::VamanaBuildParameters parameters{
        ALPHA, max_degree, 2 * max_degree, 1000, max_degree, true};

    auto tic = svs::lib::now();
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data_mutable), initial_indices, Distance(), num_threads
    );
    double build_time = svs::lib::time_difference(tic);
    index.debug_check_invariants(false);
    do_check(
        index,
        reference,
        queries,
        build_time,
        stringify("initial build (", num_indices_to_add, ") points"),
        true
    );

    test_loop(index, reference, queries, div(reference.size(), modify_fraction), 4, 20);
    return 0;
}

SVS_DEFINE_MAIN();
