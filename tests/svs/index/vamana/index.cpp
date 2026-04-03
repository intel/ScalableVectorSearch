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

// Header under test
#include "svs/index/vamana/index.h"

// Logging
#include "spdlog/sinks/callback_sink.h"
#include "svs/core/logging.h"

// svs
#include "svs/index/vamana/build_params.h"
#include "svs/lib/preprocessor.h"
#include "svs/orchestrators/vamana.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"
// stl
#include <sstream>
#include <string_view>

namespace {

const std::string_view vamana_config_parameters_v0_0_0 = R"(
__schema__ = 'vamana_index_parameters'
__version__ = 'v0.0.0'
alpha = 1.2
construction_window_size = 200
default_search_window_size = 0
entry_point = 9426
max_candidates = 1000
max_out_degree = 128
name = 'vamana config parameters'
visited_set = false
)";

const std::string_view vamana_config_parameters_v0_0_1 = R"(
__schema__ = 'vamana_index_parameters'
__version__ = 'v0.0.1'
alpha = 1.2
construction_window_size = 200
default_search_window_size = 0
entry_point = 9426
max_candidates = 1000
max_out_degree = 128
name = 'vamana config parameters'
use_full_search_history = false
visited_set = false
)";

const std::string_view vamana_config_parameters_v0_0_2 = R"(
__schema__ = 'vamana_index_parameters'
__version__ = 'v0.0.2'
alpha = 1.2
construction_window_size = 200
default_search_window_size = 0
entry_point = 9426
max_candidates = 1000
max_out_degree = 128
name = 'vamana config parameters'
use_full_search_history = false
prune_to = 100
visited_set = false
)";

} // namespace

CATCH_TEST_CASE("Vamana Index Parameters", "[index][vamana]") {
    using VamanaIndexParameters = svs::index::vamana::VamanaIndexParameters;
    CATCH_SECTION("Loading v0.0.0") {
        auto table = toml::parse(vamana_config_parameters_v0_0_0);
        auto p = svs::lib::load<VamanaIndexParameters>(svs::lib::node_view(table));
        auto expected = VamanaIndexParameters(
            9426, {1.2f, 128, 200, 1000, 128, true}, {{0, 0}, false, 4, 1}
        );
        CATCH_REQUIRE(p == expected);
    }

    CATCH_SECTION("Loading v0.0.1") {
        auto table = toml::parse(vamana_config_parameters_v0_0_1);
        auto p = svs::lib::load<VamanaIndexParameters>(svs::lib::node_view(table));

        auto expected = VamanaIndexParameters(
            9426, {1.2f, 128, 200, 1000, 128, false}, {{0, 0}, false, 4, 1}
        );

        CATCH_REQUIRE(p == expected);
    }

    CATCH_SECTION("Loading v0.0.2") {
        auto table = toml::parse(vamana_config_parameters_v0_0_2);
        auto p = svs::lib::load<VamanaIndexParameters>(svs::lib::node_view(table));

        auto expected = VamanaIndexParameters(
            9426, {1.2f, 128, 200, 1000, 100, false}, {{0, 0}, false, 4, 1}
        );

        CATCH_REQUIRE(p == expected);
    }

    CATCH_SECTION("Current version") {
        auto p = VamanaIndexParameters{
            128, {12.4f, 478, 13, 4, 10, false}, {{10, 20}, true, 1, 1}};
        CATCH_REQUIRE(svs::lib::test_self_save_load_context_free(p));
    }
}

CATCH_TEST_CASE("Static VamanaIndex Per-Index Logging", "[logging]") {
    const size_t N = 128;
    using Eltype = float;

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

    // Create some minimal data
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto graph = svs::graphs::SimpleGraph<uint32_t>(data.size(), 64);
    svs::distance::DistanceL2 distance_function;
    uint32_t entry_point = 0;
    auto threadpool = svs::threads::DefaultThreadPool(1);

    // Build the VamanaIndex with the test logger
    svs::index::vamana::VamanaBuildParameters buildParams(1.2, 64, 10, 20, 10, true);
    svs::index::vamana::VamanaIndex index(
        buildParams,
        std::move(graph),
        std::move(data),
        entry_point,
        distance_function,
        std::move(threadpool),
        test_logger
    );

    // Verify the internal log messages
    CATCH_REQUIRE(global_captured_logs.empty());
    CATCH_REQUIRE(captured_logs[0].find("Vamana Build Parameters:") != std::string::npos);
    CATCH_REQUIRE(captured_logs[1].find("Number of syncs:") != std::string::npos);
    CATCH_REQUIRE(captured_logs[2].find("Batch Size:") != std::string::npos);
}

CATCH_TEST_CASE("Vamana Index Save and Load", "[vamana][index][saveload]") {
    const size_t N = 128;
    using Eltype = float;
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto graph = svs::graphs::SimpleGraph<uint32_t>(data.size(), 64);
    svs::distance::DistanceL2 distance_function;
    uint32_t entry_point = 0;
    auto threadpool = svs::threads::DefaultThreadPool(1);

    // Build the VamanaIndex with the test logger
    svs::index::vamana::VamanaBuildParameters buildParams(1.2, 64, 10, 20, 10, true);
    svs::index::vamana::VamanaIndex index(
        buildParams,
        std::move(graph),
        std::move(data),
        entry_point,
        distance_function,
        std::move(threadpool)
    );

    const size_t NUM_NEIGHBORS = 10;
    auto queries = test_dataset::queries();
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{NUM_NEIGHBORS};

    auto results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);
    index.search(results.view(), queries.cview(), search_params);

    CATCH_SECTION("Load Vamana Index being serialized with intermediate files") {
        std::stringstream stream;
        {
            svs::lib::UniqueTempDirectory tempdir{"svs_vamana_save"};
            const auto config_dir = tempdir.get() / "config";
            const auto graph_dir = tempdir.get() / "graph";
            const auto data_dir = tempdir.get() / "data";
            std::filesystem::create_directories(config_dir);
            std::filesystem::create_directories(graph_dir);
            std::filesystem::create_directories(data_dir);
            index.save(config_dir, graph_dir, data_dir);
            svs::lib::DirectoryArchiver::pack(tempdir, stream);
        }
        {
            using Data_t = svs::data::SimpleData<Eltype, N>;

            auto loaded_index = svs::Vamana::assemble<Eltype, Data_t>(
                stream, distance_function, svs::threads::DefaultThreadPool(1)
            );

            CATCH_REQUIRE(loaded_index.size() == index.size());
            CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);

            loaded_index.search(loaded_results.view(), queries.cview(), search_params);
            for (size_t q = 0; q < queries.size(); ++q) {
                for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
                    CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                    CATCH_REQUIRE(
                        loaded_results.distance(q, i) ==
                        Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                    );
                }
            }
        }
    }

    CATCH_SECTION("Load Vamana Index being serialized natively to stream") {
        std::stringstream stream;
        index.save(stream);

        {
            using Data_t = svs::data::SimpleData<Eltype, N>;

            auto loaded_index = svs::Vamana::assemble<Eltype, Data_t>(
                stream, distance_function, svs::threads::DefaultThreadPool(1)
            );

            CATCH_REQUIRE(loaded_index.size() == index.size());
            CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);

            loaded_index.search(loaded_results.view(), queries.cview(), search_params);
            for (size_t q = 0; q < queries.size(); ++q) {
                for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
                    CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                    CATCH_REQUIRE(
                        loaded_results.distance(q, i) ==
                        Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                    );
                }
            }
        }
    }

    CATCH_SECTION("Load with pointing to in-memory stream buffer") {
        // We will load the Vamana index's data as a SimpleDataView directly from the
        // stream, without copying.
        using ViewData_t = svs::data::SimpleDataView<Eltype, N>;
        using Graph_t = svs::graphs::SimpleGraph<uint32_t, svs::View<uint32_t>>;

        // Save the full index to a stringstream.
        auto ss = std::stringstream{};
        index.save(ss);

        // Load the Vamana index from the stream.
        ss.seekg(0);
        auto loaded_index = svs::Vamana::assemble<float, ViewData_t>(
            ss, distance_function, svs::threads::DefaultThreadPool(1)
        );

        CATCH_REQUIRE(loaded_index.size() == index.size());
        CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

        auto loaded_results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);

        loaded_index.search(loaded_results.view(), queries.cview(), search_params);
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
                CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                CATCH_REQUIRE(
                    loaded_results.distance(q, i) ==
                    Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                );
            }
        }

        // We cannot extract the pointer to the FlatIndex's internal data directly.
        // To validate if the loaded Flat index is zero-copy,
        // we will load a separate SimpleDataView, modify the view's data and check if it
        // reflects in the loaded index's data. Load a SimpleDataView (zero-copy): its data_
        // must point into ss's buffer. We should follow the stream layout written by
        // Vamana::assemble:
        ss.seekg(0);
        // First: load deserializer.
        auto deserializer = svs::lib::detail::Deserializer::build(ss);
        CATCH_REQUIRE(deserializer.is_native());
        // Following svs::index::vamana::auto_assemble():
        // Second: load config parameters (not strictly necessary to validate the view
        // loading, but good to check that we can load the parameters as expected).
        auto config_parameters =
            svs::lib::load_from_stream<svs::index::vamana::VamanaIndexParameters>(ss);
        CATCH_REQUIRE(config_parameters.build_parameters == buildParams);
        // Third: load vectors data
        auto view = svs::lib::load_from_stream<ViewData_t>(ss);
        CATCH_REQUIRE(view.size() == index.size());
        CATCH_REQUIRE(view.dimensions() == index.dimensions());
        // Fourth: load graph (also not strictly necessary, but good to check that we can
        // load the graph as expected).
        auto graph = svs::lib::load_from_stream<Graph_t>(ss);
        CATCH_REQUIRE(graph.n_nodes() == index.size());

        // Check if view's data pointer points into the stringstream's internal buffer
        // (i.e., zero-copy).
        CATCH_REQUIRE(view.data() > svs::io::begin_ptr<float>(ss));
        CATCH_REQUIRE(view.data() < svs::io::end_ptr<float>(ss));
        // Now update the view's data and check if it reflects in the loaded index (since it
        // should be zero-copy). For that we will copy a vector from queries into the view's
        // data and check if the get_distance() result changes accordingly.
        auto data_index =
            std::rand() % view.size(); // Randomly select a data point to modify.
        auto query_index =
            std::rand() % queries.size(); // Randomly select a query to test against.
        auto original_distance =
            loaded_index.get_distance(data_index, queries.get_datum(query_index));
        // Verify that original distance is correct before modification.
        CATCH_REQUIRE(
            original_distance == Catch::Approx(svs::distance::compute(
                                                   distance_function,
                                                   view.get_datum(data_index),
                                                   queries.get_datum(query_index)
                                               ))
                                     .epsilon(1e-5)
        );
        // Modify the view's data by copying a query vector into it.
        view.set_datum(data_index, queries.get_datum(query_index));
        // Now the distance from the modified data point to the query should be zero (or
        // very close to zero due to floating point precision), since we copied the query
        // vector into the data point.
        auto modified_distance =
            loaded_index.get_distance(data_index, queries.get_datum(query_index));
        CATCH_REQUIRE(modified_distance == Catch::Approx(0.0).epsilon(1e-5));
    }

    CATCH_SECTION("Load with SimpleDataView pointing to memory mapped file") {
        // We will load the Vamana index's data as a SimpleDataView directly from the
        // stream, without copying.
        using ViewData_t = svs::data::SimpleDataView<Eltype, N>;
        using Graph_t = svs::graphs::SimpleGraph<uint32_t, svs::View<uint32_t>>;

        // Save the full index to a file
        svs::lib::UniqueTempDirectory tempdir{"svs_flat_save"};
        auto index_path = tempdir.get() / "index.bin";
        auto os = std::ofstream{index_path, std::ios::binary};
        index.save(os);
        os.close();

        auto index_is = svs::io::mmstream(index_path);

        // Load the Vamana index from the stream.
        auto loaded_index = svs::Vamana::assemble<float, ViewData_t>(
            index_is, distance_function, svs::threads::DefaultThreadPool(1)
        );

        CATCH_REQUIRE(loaded_index.size() == index.size());
        CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

        auto loaded_results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);

        loaded_index.search(loaded_results.view(), queries.cview(), search_params);
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
                CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                CATCH_REQUIRE(
                    loaded_results.distance(q, i) ==
                    Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                );
            }
        }

        // We cannot extract the pointer to the FlatIndex's internal data directly.
        // To validate if the loaded Flat index is zero-copy,
        // we will load a separate SimpleDataView, modify the view's data and check if it
        // reflects in the loaded index's data. Load a SimpleDataView (zero-copy): its data_
        // must point into ss's buffer. We should follow the stream layout written by
        // Vamana::assemble:
        auto view_is = svs::io::mmstream(index_path);
        // First: load deserializer.
        auto deserializer = svs::lib::detail::Deserializer::build(view_is);
        CATCH_REQUIRE(deserializer.is_native());
        // Following svs::index::vamana::auto_assemble():
        // Second: load config parameters (not strictly necessary to validate the view
        // loading, but good to check that we can load the parameters as expected).
        auto config_parameters =
            svs::lib::load_from_stream<svs::index::vamana::VamanaIndexParameters>(view_is);
        CATCH_REQUIRE(config_parameters.build_parameters == buildParams);
        // Third: load vectors data
        auto view = svs::lib::load_from_stream<ViewData_t>(view_is);
        CATCH_REQUIRE(view.size() == index.size());
        CATCH_REQUIRE(view.dimensions() == index.dimensions());
        // Fourth: load graph (also not strictly necessary, but good to check that we can
        // load the graph as expected).
        auto graph = svs::lib::load_from_stream<Graph_t>(view_is);
        CATCH_REQUIRE(graph.n_nodes() == index.size());

        // Check if view's data pointer points into the stringstream's internal buffer
        // (i.e., zero-copy).
        CATCH_REQUIRE(view.data() > svs::io::begin_ptr<float>(view_is));
        CATCH_REQUIRE(view.data() < svs::io::end_ptr<float>(view_is));
        // Now update the view's data and check if it reflects in the loaded index (since it
        // should be zero-copy). For that we will copy a vector from queries into the view's
        // data and check if the get_distance() result changes accordingly.
        auto data_index =
            std::rand() % view.size(); // Randomly select a data point to modify.
        auto query_index =
            std::rand() % queries.size(); // Randomly select a query to test against.
        auto original_distance =
            loaded_index.get_distance(data_index, queries.get_datum(query_index));
        // Verify that original distance is correct before modification.
        CATCH_REQUIRE(
            original_distance == Catch::Approx(svs::distance::compute(
                                                   distance_function,
                                                   view.get_datum(data_index),
                                                   queries.get_datum(query_index)
                                               ))
                                     .epsilon(1e-5)
        );
        // Modify the view's data by copying a query vector into it.
        view.set_datum(data_index, queries.get_datum(query_index));
        // Now the distance from the modified data point to the query should be zero (or
        // very close to zero due to floating point precision), since we copied the query
        // vector into the data point.
        auto modified_distance =
            loaded_index.get_distance(data_index, queries.get_datum(query_index));
        CATCH_REQUIRE(modified_distance == Catch::Approx(0.0).epsilon(1e-5));
    }
}

CATCH_TEST_CASE("Vamana Index Default Parameters", "[long][parameter][vamana]") {
    using Catch::Approx;
    std::filesystem::path data_path = test_dataset::data_svs_file();

    CATCH_SECTION("L2 Distance Defaults") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        auto data_loader = svs::data::SimpleData<float>::load(data_path);
        svs::Vamana index = svs::Vamana::build<float>(build_params, data_loader, svs::L2);
        CATCH_REQUIRE(index.get_alpha() == Approx(svs::VAMANA_ALPHA_MINIMIZE_DEFAULT));
    }

    CATCH_SECTION("MIP Distance Defaults") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::MIP, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        auto data_loader = svs::data::SimpleData<float>::load(data_path);
        svs::Vamana index = svs::Vamana::build<float>(build_params, data_loader, svs::MIP);
        CATCH_REQUIRE(index.get_alpha() == Approx(svs::VAMANA_ALPHA_MAXIMIZE_DEFAULT));
    }

    CATCH_SECTION("Invalid Alpha for L2") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        build_params.alpha = 0.8f;
        auto data_loader = svs::data::SimpleData<float>::load(data_path);
        CATCH_REQUIRE_THROWS_WITH(
            svs::Vamana::build<float>(build_params, data_loader, svs::L2),
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
        CATCH_REQUIRE_THROWS_WITH(
            svs::Vamana::build<float>(build_params, data_loader, svs::MIP),
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
        CATCH_REQUIRE_THROWS_WITH(
            svs::Vamana::build<float>(build_params, data_loader, svs::L2),
            "prune_to must be <= graph_max_degree"
        );
    }

    CATCH_SECTION("L2 Distance Empty Params") {
        svs::index::vamana::VamanaBuildParameters empty_params;
        auto data_loader = svs::data::SimpleData<float>::load(data_path);
        svs::Vamana index = svs::Vamana::build<float>(empty_params, data_loader, svs::L2);
        CATCH_REQUIRE(index.get_alpha() == Approx(svs::VAMANA_ALPHA_MINIMIZE_DEFAULT));
        CATCH_REQUIRE(index.get_graph_max_degree() == svs::VAMANA_GRAPH_MAX_DEGREE_DEFAULT);
        CATCH_REQUIRE(index.get_prune_to() == svs::VAMANA_GRAPH_MAX_DEGREE_DEFAULT - 4);
        CATCH_REQUIRE(
            index.get_construction_window_size() == svs::VAMANA_WINDOW_SIZE_DEFAULT
        );
        CATCH_REQUIRE(index.get_max_candidates() == 3 * svs::VAMANA_WINDOW_SIZE_DEFAULT);
        CATCH_REQUIRE(
            index.get_full_search_history() == svs::VAMANA_USE_FULL_SEARCH_HISTORY_DEFAULT
        );
    }
}
