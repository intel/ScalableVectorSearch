/*
 * Copyright 2025 Intel Corporation
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

#include "svs/index/flat/flat.h"
#include "svs/core/logging.h"
#include "svs/lib/file.h"
#include "svs/lib/saveload/load.h"
#include "svs/orchestrators/exhaustive.h"

// tests
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// spd log
#include "spdlog/sinks/callback_sink.h"

CATCH_TEST_CASE("FlatIndex Logging Test", "[logging]") {
    // Vector to store captured log messages
    std::vector<std::string> captured_logs;
    std::vector<std::string> global_captured_logs;

    // Create a callback sink to capture log messages
    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&captured_logs](const spdlog::details::log_msg& msg) {
            captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    callback_sink->set_level(spdlog::level::trace);

    // Set up the FlatIndex with the test logger
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

    std::vector<float> data{1.0f, 2.0f};
    auto dataView = svs::data::SimpleDataView<float>(data.data(), 2, 1);
    svs::distance::DistanceL2 dist;
    auto threadpool = svs::threads::DefaultThreadPool(1);

    svs::index::flat::FlatIndex index(
        std::move(dataView), dist, std::move(threadpool), test_logger
    );

    // Log a message
    test_logger->info("Test FlatIndex Logging");

    // Verify the log output
    CATCH_REQUIRE(global_captured_logs.empty());
    CATCH_REQUIRE(captured_logs.size() == 1);
    CATCH_REQUIRE(captured_logs[0] == "Test FlatIndex Logging");
}

CATCH_TEST_CASE("Flat Index Save and Load", "[flat][index][saveload]") {
    using Data_t = svs::data::SimpleData<float>;
    using Distance_t = svs::distance::DistanceL2;
    using Index_t = svs::index::flat::FlatIndex<Data_t, Distance_t>;

    // Load test data
    auto data = Data_t::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();

    // Build index
    Distance_t dist;
    Index_t index = Index_t(std::move(data), dist, svs::threads::DefaultThreadPool(1));

    size_t num_neighbors = 10;
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(results.view(), queries.cview(), {});

    CATCH_SECTION("Load Flat being serialized natively to stream") {
        std::stringstream ss;
        index.save(ss);

        auto loaded_index = svs::Flat::assemble<float, Data_t>(
            ss, dist, svs::threads::DefaultThreadPool(1)
        );

        CATCH_REQUIRE(loaded_index.size() == index.size());
        CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

        auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        loaded_index.search(loaded_results.view(), queries.cview(), {});

        // Compare results - should be identical
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
                CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                CATCH_REQUIRE(
                    loaded_results.distance(q, i) ==
                    Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                );
            }
        }
    }

    CATCH_SECTION("Load Flat being serialized with intermediate files") {
        std::stringstream ss;

        svs::lib::UniqueTempDirectory tempdir{"svs_flat_save"};
        index.save(tempdir);
        svs::lib::DirectoryArchiver::pack(tempdir, ss);

        auto loaded_index = svs::Flat::assemble<float, Data_t>(
            ss, dist, svs::threads::DefaultThreadPool(1)
        );

        CATCH_REQUIRE(loaded_index.size() == index.size());
        CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

        auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        loaded_index.search(loaded_results.view(), queries.cview(), {});

        // Compare results - should be identical
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
                CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                CATCH_REQUIRE(
                    loaded_results.distance(q, i) ==
                    Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                );
            }
        }
    }

    CATCH_SECTION("Load with pointing to in-memory stream buffer") {
        // We will load the FlatIndex's data as a SimpleDataView directly from the stream,
        // without copying.
        using ViewData_t = svs::data::SimpleData<
            Data_t::element_type,
            svs::Dynamic,
            svs::io::MemoryStreamAllocator<Data_t::element_type>>;

        // Save the full index to a stringstream.
        auto ss = std::stringstream{};
        index.save(ss);
        ss.seekg(0);

        // Load the FlatIndex from the stream.
        auto loaded_index = svs::Flat::assemble<float, ViewData_t>(
            ss, dist, svs::threads::DefaultThreadPool(1)
        );

        CATCH_REQUIRE(loaded_index.size() == index.size());
        CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

        auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        loaded_index.search(loaded_results.view(), queries.cview(), {});

        // Compare results - should be identical
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
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
        // FlatIndex::assemble:
        ss.seekg(0);
        // First: load deserializer.
        auto deserializer = svs::lib::detail::Deserializer::build(ss);
        CATCH_REQUIRE(deserializer.is_native());
        // Second: load vectors data
        auto view = svs::lib::load_from_stream<ViewData_t>(ss);

        CATCH_REQUIRE(view.size() == index.size());
        CATCH_REQUIRE(view.dimensions() == index.dimensions());
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
                                                   dist,
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
        using ViewData_t = svs::data::SimpleData<
            Data_t::element_type,
            svs::Dynamic,
            svs::io::MemoryStreamAllocator<Data_t::element_type>>;

        svs::lib::UniqueTempDirectory tempdir{"svs_flat_save"};
        auto index_path = tempdir.get() / "index.bin";
        auto os = std::ofstream{index_path, std::ios::binary};
        index.save(os);
        os.close();

        auto index_is = svs::io::mmstream(index_path);
        // Load the FlatIndex from the stream.
        auto loaded_index = svs::Flat::assemble<float, ViewData_t>(
            index_is, dist, svs::threads::DefaultThreadPool(1)
        );

        CATCH_REQUIRE(loaded_index.size() == index.size());
        CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

        auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        loaded_index.search(loaded_results.view(), queries.cview(), {});

        // Compare results - should be identical
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
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
        // FlatIndex::assemble:
        auto view_is = svs::io::mmstream(index_path);
        // First: load deserializer.
        auto deserializer = svs::lib::detail::Deserializer::build(view_is);
        CATCH_REQUIRE(deserializer.is_native());
        // Second: load vectors data
        auto view = svs::lib::load_from_stream<ViewData_t>(view_is);

        CATCH_REQUIRE(view.size() == index.size());
        CATCH_REQUIRE(view.dimensions() == index.dimensions());
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
                                                   dist,
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
