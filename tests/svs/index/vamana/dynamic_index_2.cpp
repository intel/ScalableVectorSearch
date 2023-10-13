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

// tests
#include "tests/utils/test_dataset.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

using Idx = uint32_t;
using Eltype = float;
using QueryEltype = float;
using Distance = svs::distance::DistanceL2;
const size_t N = 128;

const size_t NUM_NEIGHBORS = 10;
const double TARGET_RECALL = 0.95;

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
    CATCH_REQUIRE(gt.n_neighbors() == NUM_NEIGHBORS);
    CATCH_REQUIRE(gt.n_queries() == queries.size());

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
            CATCH_REQUIRE(points <= num_points);
            CATCH_REQUIRE(points > num_points - reference.bucket_size());
            index.debug_check_invariants(true);
            do_check(index, reference, queries, time, stringify("add ", points, " points"));
        }

        // Delete Points
        {
            auto [points, time] = reference.delete_points(index, num_points);
            CATCH_REQUIRE(points <= num_points);
            CATCH_REQUIRE(points > num_points - reference.bucket_size());
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

CATCH_TEST_CASE("Testing Graph Index", "[graph_index][dynamic_index]") {
    // Set hyper parameters here
    const size_t max_degree = 64;
#if defined(NDEBUG)
    const float initial_fraction = 0.25;
    const float modify_fraction = 0.05;
#else
    const float initial_fraction = 0.05;
    const float modify_fraction = 0.005;
#endif
    const size_t num_threads = 10;
    const float alpha = 1.2;

    // Load the base dataset and queries.
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto num_points = data.size();
    auto queries = test_dataset::queries();

    auto reference = svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>(
        std::move(data),
        Distance(),
        num_threads,
        div(num_points, 0.5 * modify_fraction),
        NUM_NEIGHBORS,
        queries,
        0x12345678
    );

    auto num_indices_to_add = div(reference.size(), initial_fraction);
    std::cout << "Initializing with " << num_indices_to_add << " entries!\n";

    // Construct a blocked dataset consisting of 50% of the base dataset.
    auto data_mutable = svs::data::BlockedData<Eltype, N>(num_indices_to_add, N);
    std::vector<Idx> initial_indices{};
    {
        auto [vectors, indices] = reference.generate(num_indices_to_add);
        // Copy assign ``initial_indices``
        auto num_points_added = indices.size();
        CATCH_REQUIRE(vectors.size() == num_points_added);
        CATCH_REQUIRE(num_points_added <= num_indices_to_add);
        CATCH_REQUIRE(num_points_added > num_indices_to_add - reference.bucket_size());

        initial_indices = indices;
        if (vectors.size() != num_indices_to_add || indices.size() != num_indices_to_add) {
            throw ANNEXCEPTION("Something when horribly wrong!");
        }

        for (size_t i = 0; i < num_indices_to_add; ++i) {
            data_mutable.set_datum(i, vectors.get_datum(i));
        }
    }

    svs::index::vamana::VamanaBuildParameters parameters{
        1.2, max_degree, 2 * max_degree, 1000, max_degree - 4, true};

    auto tic = svs::lib::now();
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data_mutable), initial_indices, Distance(), num_threads
    );
    double build_time = svs::lib::time_difference(tic);
    index.debug_check_invariants(false);

    // Verify that we can get and set alpha and the construction window size.
    CATCH_REQUIRE(index.get_alpha() == alpha);
    index.set_alpha(1.0);
    CATCH_REQUIRE(index.get_alpha() == 1.0);
    index.set_alpha(alpha);
    CATCH_REQUIRE(index.get_alpha() == alpha);

    const size_t expected_construction_window = 2 * max_degree;
    CATCH_REQUIRE(index.get_construction_window_size() == expected_construction_window);
    index.set_construction_window_size(10);
    CATCH_REQUIRE(index.get_construction_window_size() == 10);
    index.set_construction_window_size(expected_construction_window);
    CATCH_REQUIRE(index.get_construction_window_size() == expected_construction_window);

    reference.configure_extra_checks(true);
    CATCH_REQUIRE(reference.extra_checks_enabled());

    do_check(
        index,
        reference,
        queries,
        build_time,
        stringify("initial build (", num_indices_to_add, ") points"),
        true
    );

    test_loop(index, reference, queries, div(reference.size(), modify_fraction), 2, 6);

    // Try saving the index.
    svs_test::prepare_temp_directory();
    auto tmp = svs_test::temp_directory();
    index.save(tmp / "config", tmp / "graph", tmp / "data");

    auto reloaded = svs::index::vamana::auto_dynamic_assemble(
        tmp / "config",
        SVS_LAZY(svs::graphs::SimpleBlockedGraph<uint32_t>::load(tmp / "graph")),
        SVS_LAZY(svs::data::BlockedData<float>::load(tmp / "data")),
        svs::DistanceL2(),
        2
    );

    // Make sure parameters were saved across the saving.
    CATCH_REQUIRE(index.get_alpha() == reloaded.get_alpha());
    CATCH_REQUIRE(index.get_max_candidates() == reloaded.get_max_candidates());
    CATCH_REQUIRE(
        index.get_construction_window_size() == reloaded.get_construction_window_size()
    );
    CATCH_REQUIRE(index.size() == reloaded.size());
    // ID's preserved across runs.
    index.on_ids([&](size_t e) { CATCH_REQUIRE(reloaded.has_id(e)); });
}
