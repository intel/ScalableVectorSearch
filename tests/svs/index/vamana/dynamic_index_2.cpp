/*
 * Copyright 2023 Intel Corporation
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

// svs
#include "svs/core/medioid.h"
#include "svs/core/recall.h"
#include "svs/index/flat/flat.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/lib/float16.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/timing.h"

#include "svs/misc/dynamic_helper.h"

// tests
#include "spdlog/sinks/callback_sink.h"
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

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
auto find_windowsize(
    MutableIndex& index,
    const Groundtruth& groundtruth,
    const Queries& queries,
    double target_recall = TARGET_RECALL,
    size_t window_lower = NUM_NEIGHBORS,
    size_t window_upper = 1000
) -> svs::index::vamana::VamanaSearchParameters {
    auto range = svs::threads::UnitRange<size_t>(window_lower, window_upper);
    auto parameters = svs::index::vamana::VamanaSearchParameters();
    size_t window_size = *std::lower_bound(
        range.begin(),
        range.end(),
        target_recall,
        [&](size_t window_size, double recall) {
            parameters.buffer_config(window_size);
            auto result =
                svs::index::search_batch_with(index, queries, NUM_NEIGHBORS, parameters);
            auto this_recall = svs::k_recall_at_n(groundtruth, result);
            return this_recall < recall;
        }
    );

    parameters.buffer_config(window_size);
    return parameters;
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
        auto parameters = find_windowsize(index, gt, queries);
        index.set_search_parameters(parameters);
    }

    // Run search
    tic = svs::lib::now();
    auto result = svs::index::search_batch(index, queries, NUM_NEIGHBORS);
    double search_time = svs::lib::time_difference(tic);

    // Extra ID checks
    reference.check_ids(result);
    reference.check_equal_ids(index);

    // compute recall
    double recall = svs::k_recall_at_n(gt, result, NUM_NEIGHBORS, NUM_NEIGHBORS);

    // Report the calibrated search window size if we calibrated this round.
    if (calibrate) {
        auto search_window_size =
            index.get_search_parameters().buffer_config_.get_search_window_size();
        message += stringify(" - Calibrate window size: ", search_window_size);
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

    // Set up log
    std::vector<std::string> captured_logs;
    std::vector<svs::logging::Level> captured_levels;

    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&captured_logs, &captured_levels](const spdlog::details::log_msg& msg) {
            captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
            captured_levels.push_back(svs::logging::detail::from_spdlog(msg.level));
        }
    );
    callback_sink->set_level(spdlog::level::trace);
    auto test_logger = std::make_shared<spdlog::logger>("test_logger", callback_sink);
    test_logger->set_level(spdlog::level::trace);
    std::vector<std::string> global_captured_logs;
    auto global_callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&global_captured_logs](const spdlog::details::log_msg& msg) {
            global_captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    global_callback_sink->set_level(spdlog::level::trace);
    auto original_logger = svs::logging::get();
    original_logger->sinks().push_back(global_callback_sink);

    // Load the base dataset and queries.
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto data_copy = data;
    auto num_points = data.size();
    auto queries = test_dataset::queries();

    auto reference = svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>(
        std::move(data),
        Distance(),
        num_threads,
        div(num_points, 0.5 * modify_fraction),
        NUM_NEIGHBORS,
        queries,
        0x12345678,
        test_logger
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
        parameters,
        std::move(data_mutable),
        initial_indices,
        Distance(),
        num_threads,
        test_logger
    );
    double build_time = svs::lib::time_difference(tic);
    index.debug_check_invariants(false);

    CATCH_REQUIRE(captured_logs[0].find("Total / % Measured:") != std::string::npos);
    CATCH_REQUIRE(captured_levels[0] == svs::logging::Level::Debug);
    CATCH_REQUIRE(captured_logs[1].find("Vamana Build Parameters:") != std::string::npos);
    CATCH_REQUIRE(captured_levels[1] == svs::logging::Level::Debug);
    CATCH_REQUIRE(captured_logs[2].find("Number of syncs:") != std::string::npos);
    CATCH_REQUIRE(captured_levels[2] == svs::logging::Level::Trace);
    CATCH_REQUIRE(captured_logs[3].find("Batch Size:") != std::string::npos);
    CATCH_REQUIRE(captured_levels[3] == svs::logging::Level::Trace);

    // Test get_distance functionality
    svs::DistanceDispatcher dispatcher(svs::L2);
    dispatcher([&](auto dist) {
        svs_test::GetDistanceTester::test(index, dist, data_copy, initial_indices);
    });

    // Verify that we can get and set build parameters.
    CATCH_REQUIRE(index.get_alpha() == alpha);
    index.set_alpha(1.0);
    CATCH_REQUIRE(index.get_alpha() == 1.0);
    index.set_alpha(alpha);
    CATCH_REQUIRE(index.get_alpha() == alpha);

    CATCH_REQUIRE(index.get_graph_max_degree() == max_degree);

    const size_t expected_construction_window = 2 * max_degree;
    CATCH_REQUIRE(index.get_construction_window_size() == expected_construction_window);
    index.set_construction_window_size(10);
    CATCH_REQUIRE(index.get_construction_window_size() == 10);
    index.set_construction_window_size(expected_construction_window);
    CATCH_REQUIRE(index.get_construction_window_size() == expected_construction_window);

    CATCH_REQUIRE(index.get_max_candidates() == 1000);
    index.set_max_candidates(750);
    CATCH_REQUIRE(index.get_max_candidates() == 750);

    CATCH_REQUIRE(index.get_prune_to() == max_degree - 4);
    index.set_prune_to(max_degree - 2);
    CATCH_REQUIRE(index.get_prune_to() == max_degree - 2);

    CATCH_REQUIRE(index.get_full_search_history() == true);
    index.set_full_search_history(false);
    CATCH_REQUIRE(index.get_full_search_history() == false);

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

    do_check(
        reloaded,
        reference,
        queries,
        build_time,
        stringify("initial build (", num_indices_to_add, ") points"),
        true
    );

    reloaded = svs::index::vamana::auto_dynamic_assemble(
        tmp / "config",
        SVS_LAZY(svs::graphs::SimpleBlockedGraph<uint32_t>::load(tmp / "graph")),
        SVS_LAZY(svs::data::BlockedData<float>::load(tmp / "data")),
        svs::DistanceL2(),
        svs::threads::CppAsyncThreadPool(2)
    );

    do_check(
        reloaded,
        reference,
        queries,
        build_time,
        stringify("initial build (", num_indices_to_add, ") points"),
        true
    );

    reloaded = svs::index::vamana::auto_dynamic_assemble(
        tmp / "config",
        SVS_LAZY(svs::graphs::SimpleBlockedGraph<uint32_t>::load(tmp / "graph")),
        SVS_LAZY(svs::data::BlockedData<float>::load(tmp / "data")),
        svs::DistanceL2(),
        svs::threads::QueueThreadPoolWrapper(2)
    );

    do_check(
        reloaded,
        reference,
        queries,
        build_time,
        stringify("initial build (", num_indices_to_add, ") points"),
        true
    );

    // Make sure parameters were saved across the saving.
    CATCH_REQUIRE(index.get_alpha() == reloaded.get_alpha());
    CATCH_REQUIRE(index.get_graph_max_degree() == reloaded.get_graph_max_degree());
    CATCH_REQUIRE(index.get_max_candidates() == reloaded.get_max_candidates());
    CATCH_REQUIRE(
        index.get_construction_window_size() == reloaded.get_construction_window_size()
    );
    CATCH_REQUIRE(index.get_prune_to() == reloaded.get_prune_to());
    CATCH_REQUIRE(index.get_full_search_history() == reloaded.get_full_search_history());
    CATCH_REQUIRE(index.size() == reloaded.size());
    // ID's preserved across runs.
    index.on_ids([&](size_t e) { CATCH_REQUIRE(reloaded.has_id(e)); });

    CATCH_REQUIRE(global_captured_logs.empty());
}

CATCH_TEST_CASE("Dynamic MutableVamanaIndex Per-Index Logging Test", "[logging]") {
    // Vector to store captured log messages
    std::vector<std::string> captured_logs;
    std::vector<std::string> global_captured_logs;

    // Create a callback sink to capture log messages
    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&captured_logs](const spdlog::details::log_msg& msg) {
            captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    callback_sink->set_level(spdlog::level::trace); // Capture all log levels

    // Create a logger with the callback sink
    auto test_logger = std::make_shared<spdlog::logger>("test_logger", callback_sink);
    test_logger->set_level(spdlog::level::trace);

    auto global_callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&global_captured_logs](const spdlog::details::log_msg& msg) {
            global_captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    global_callback_sink->set_level(spdlog::level::trace);

    auto original_logger = svs::logging::get();
    original_logger->sinks().push_back(global_callback_sink);

    // Setup index
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    std::vector<size_t> initial_indices(data.size());
    std::iota(initial_indices.begin(), initial_indices.end(), 0);
    svs::index::vamana::VamanaBuildParameters buildParams(1.2, 64, 10, 20, 10, true);
    auto threadpool = svs::threads::DefaultThreadPool(1);
    auto index = svs::index::vamana::MutableVamanaIndex(
        buildParams,
        std::move(data),
        initial_indices,
        svs::DistanceL2(),
        std::move(threadpool),
        test_logger
    );

    // Verify the internal log messages
    CATCH_REQUIRE(global_captured_logs.empty());
    CATCH_REQUIRE(captured_logs[0].find("Vamana Build Parameters:") != std::string::npos);
    CATCH_REQUIRE(captured_logs[1].find("Number of syncs:") != std::string::npos);
    CATCH_REQUIRE(captured_logs[2].find("Batch Size:") != std::string::npos);
}

CATCH_TEST_CASE("Dynamic MutableVamanaIndex Default Logger Test", "[logging]") {
    // Setup index with default logger
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    std::vector<size_t> initial_indices(data.size());
    std::iota(initial_indices.begin(), initial_indices.end(), 0);
    svs::index::vamana::VamanaBuildParameters buildParams(1.2, 64, 10, 20, 10, true);
    auto threadpool = svs::threads::DefaultThreadPool(1);
    auto index = svs::index::vamana::MutableVamanaIndex(
        buildParams,
        std::move(data),
        initial_indices,
        svs::DistanceL2(),
        std::move(threadpool)
    );

    // Verify that the default logger is used
    auto default_logger = svs::logging::get();
    CATCH_REQUIRE(index.get_logger() == default_logger);
}

CATCH_TEST_CASE("Dynamic Vamana Index Default Parameters", "[parameter][vamana]") {
    using Catch::Approx;
    std::filesystem::path data_path = test_dataset::data_svs_file();

    CATCH_SECTION("L2 Distance Defaults") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        auto data_loader = svs::data::SimpleData<float>::load(data_path);

        // Get IDs for all points in the dataset
        std::vector<size_t> indices(data_loader.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Build dynamic index with L2 distance
        auto index = svs::index::vamana::MutableVamanaIndex(
            build_params, std::move(data_loader), indices, svs::distance::DistanceL2(), 2
        );

        CATCH_REQUIRE(index.get_alpha() == Approx(svs::VAMANA_ALPHA_MINIMIZE_DEFAULT));
    }

    CATCH_SECTION("MIP Distance Defaults") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::MIP, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        auto data_loader = svs::data::SimpleData<float>::load(data_path);

        // Get IDs for all points in the dataset
        std::vector<size_t> indices(data_loader.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Build dynamic index with MIP distance
        auto index = svs::index::vamana::MutableVamanaIndex(
            build_params, std::move(data_loader), indices, svs::distance::DistanceIP(), 2
        );

        CATCH_REQUIRE(index.get_alpha() == Approx(svs::VAMANA_ALPHA_MAXIMIZE_DEFAULT));
    }

    CATCH_SECTION("Invalid Alpha for L2") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        build_params.alpha = 0.8f;
        auto data_loader = svs::data::SimpleData<float>::load(data_path);

        // Get IDs for all points in the dataset
        std::vector<size_t> indices(data_loader.size());
        std::iota(indices.begin(), indices.end(), 0);

        CATCH_REQUIRE_THROWS_WITH(
            svs::index::vamana::MutableVamanaIndex(
                build_params,
                std::move(data_loader),
                indices,
                svs::distance::DistanceL2(),
                2
            ),
            "For L2 distance, alpha must be >= 1.0"
        );
    }

    CATCH_SECTION("Invalid Alpha for MIP") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::MIP, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        build_params.alpha = 1.2f;
        auto data_loader = svs::data::SimpleData<float>::load(data_path);

        // Get IDs for all points in the dataset
        std::vector<size_t> indices(data_loader.size());
        std::iota(indices.begin(), indices.end(), 0);

        CATCH_REQUIRE_THROWS_WITH(
            svs::index::vamana::MutableVamanaIndex(
                build_params,
                std::move(data_loader),
                indices,
                svs::distance::DistanceIP(),
                2
            ),
            "For MIP/Cosine distance, alpha must be <= 1.0"
        );
    }

    CATCH_SECTION("Invalid prune_to > graph_max_degree") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        build_params.prune_to = build_params.graph_max_degree + 10;
        auto data_loader = svs::data::SimpleData<float>::load(data_path);

        // Get IDs for all points in the dataset
        std::vector<size_t> indices(data_loader.size());
        std::iota(indices.begin(), indices.end(), 0);

        CATCH_REQUIRE_THROWS_WITH(
            svs::index::vamana::MutableVamanaIndex(
                build_params,
                std::move(data_loader),
                indices,
                svs::distance::DistanceL2(),
                2
            ),
            "prune_to must be <= graph_max_degree"
        );
    }

    CATCH_SECTION("L2 Distance Empty Params") {
        svs::index::vamana::VamanaBuildParameters params;
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(i + 1);
        }
        auto data_view = svs::data::SimpleDataView<float>(data.data(), 8, 4);
        std::vector<size_t> indices = {0, 1, 2, 3, 4, 5, 6, 7};
        auto index = svs::index::vamana::MutableVamanaIndex(
            params, std::move(data_view), indices, svs::distance::DistanceL2(), 1
        );
        CATCH_REQUIRE(index.get_alpha() == Approx(svs::VAMANA_ALPHA_MINIMIZE_DEFAULT));
        CATCH_REQUIRE(index.get_graph_max_degree() == svs::VAMANA_GRAPH_MAX_DEGREE_DEFAULT);
        CATCH_REQUIRE(index.get_prune_to() == svs::VAMANA_GRAPH_MAX_DEGREE_DEFAULT - 4);
        CATCH_REQUIRE(
            index.get_construction_window_size() == svs::VAMANA_WINDOW_SIZE_DEFAULT
        );
        CATCH_REQUIRE(
            index.get_max_candidates() == 2 * svs::VAMANA_GRAPH_MAX_DEGREE_DEFAULT
        );
        CATCH_REQUIRE(
            index.get_full_search_history() == svs::VAMANA_USE_FULL_SEARCH_HISTORY_DEFAULT
        );
    }
}