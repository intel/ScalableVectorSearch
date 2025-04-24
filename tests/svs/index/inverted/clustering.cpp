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

// header under test
#include "svs/index/inverted/clustering.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <unordered_map>

// logging
#include "spdlog/sinks/callback_sink.h"

namespace {

template <svs::data::ImmutableMemoryDataset Data, typename Distance>
svs::index::inverted::Clustering<uint32_t> randomly_cluster(
    const Data& data,
    const svs::index::vamana::VamanaBuildParameters& primary_parameters,
    const svs::index::inverted::ClusteringParameters& clustering_parameters,
    const Distance& distance,
    size_t num_threads
) {
    auto threadpool = svs::threads::DefaultThreadPool(num_threads);

    // Select Centroids.
    auto centroids = svs::index::inverted::randomly_select_centroids(
        data.size(),
        svs::lib::narrow_cast<size_t>(
            std::floor(data.size() * clustering_parameters.percent_centroids_.value())
        ),
        clustering_parameters.seed_
    );

    // Build Primary Index.
    auto index = svs::index::inverted::build_primary_index(
        data,
        svs::lib::as_const_span(centroids),
        primary_parameters,
        distance,
        std::move(threadpool)
    );

    // Cluster the dataset with the help of the primary index.
    return svs::index::inverted::cluster_with(
        data, svs::lib::as_const_span(centroids), clustering_parameters, index
    );
}

} // namespace

CATCH_TEST_CASE("Random Clustering", "[inverted][random_clustering]") {
    namespace inverted = svs::index::inverted;
    CATCH_SECTION("Clustering Parameters") {
        auto p = inverted::ClusteringParameters();

        // Test the setter methods.
#define XX(name, v)                        \
    CATCH_REQUIRE(p.name##_ != v);         \
    CATCH_REQUIRE(p.name(v).name##_ == v); \
    CATCH_REQUIRE(p.name##_ == v);

        XX(percent_centroids, svs::lib::Percent(0.5));
        XX(epsilon, 0.99);
        XX(max_replicas, 20);
        XX(max_cluster_size, 10);
        XX(seed, 0x12345789);
        XX(batchsize, 10);
        XX(search_window_size, 100);
        XX(num_intermediate_results, 100);
        XX(refinement_alpha, 1.234F);
#undef XX

        // Saving and loading
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        CATCH_REQUIRE(svs::lib::test_self_save_load(p, dir));
    }

    CATCH_SECTION("Centroid Selection") {
        constexpr size_t data_size = 10000;
        auto ids = inverted::randomly_select_centroids(data_size, data_size / 10, 0xc0ffee);

        // Make sure we have the currect number of IDs and that they are all in-bounds.
        CATCH_REQUIRE(ids.size() == data_size * 0.1);
        for (auto id : ids) {
            CATCH_REQUIRE(id < data_size);
        }

        // Retry with a different percent of centroids.
        ids = inverted::randomly_select_centroids(data_size, data_size / 100, 0xc0ffee);
        CATCH_REQUIRE(ids.size() == data_size * 0.01);
        for (auto id : ids) {
            CATCH_REQUIRE(id < data_size);
        }
    }

    CATCH_SECTION("Cluster") {
        auto cluster = inverted::Cluster<uint32_t>(10);
        CATCH_REQUIRE(cluster.size() == 0);
        CATCH_REQUIRE(cluster.centroid() == 10);

        auto expected =
            std::vector<svs::Neighbor<uint32_t>>({{0, 5.0}, {20, 2.0}, {40, 1.0}});

        for (auto i : expected) {
            cluster.push_back(i);
        }
        CATCH_REQUIRE(cluster.size() == 3);
        CATCH_REQUIRE(cluster.centroid() == 10);

        auto eq = [](auto b, auto e, auto b2) {
            return std::equal(b, e, b2, svs::NeighborEqual());
        };

        {
            auto& elements = cluster.elements();
            CATCH_REQUIRE(eq(elements.begin(), elements.end(), expected.begin()));
        }

        {
            auto& elements = std::as_const(cluster).elements();
            CATCH_REQUIRE(eq(elements.begin(), elements.end(), expected.begin()));
        }

        // iterators
        const auto& cluster_ref = cluster;
        CATCH_REQUIRE(eq(cluster.begin(), cluster.end(), expected.begin()));
        CATCH_REQUIRE(eq(cluster_ref.begin(), cluster_ref.end(), expected.begin()));
        CATCH_REQUIRE(eq(cluster.cbegin(), cluster.cend(), expected.begin()));

        CATCH_REQUIRE(eq(cluster.rbegin(), cluster.rend(), expected.rbegin()));
        CATCH_REQUIRE(eq(cluster_ref.rbegin(), cluster_ref.rend(), expected.rbegin()));
        CATCH_REQUIRE(eq(cluster.crbegin(), cluster.crend(), expected.rbegin()));

        // Sorting.
        auto cluster_copy = cluster;
        cluster.sort(std::less<>());
        std::sort(expected.begin(), expected.end());
        CATCH_REQUIRE(cluster != cluster_copy);

        cluster_copy = cluster;
        CATCH_REQUIRE(cluster == cluster_copy);
        cluster_copy.centroid_ = 0;
        CATCH_REQUIRE(cluster != cluster_copy);
        cluster_copy = cluster;
        CATCH_REQUIRE(cluster == cluster_copy);
        cluster_copy.push_back({2, 2.0});
        CATCH_REQUIRE(cluster != cluster_copy);

        // Serialization.
        CATCH_REQUIRE(svs_test::prepare_temp_directory());
        auto file = svs_test::temp_directory();
        file /= "file.bin";

        auto get_serialized_size = []<typename I>(const inverted::Cluster<I>& c) {
            return sizeof(I) + sizeof(size_t) + c.size() * sizeof(svs::Neighbor<I>);
        };

        {
            auto io = svs::lib::open_write(file);
            auto bytes = cluster.serialize(io);
            CATCH_REQUIRE(bytes == get_serialized_size(cluster));
            bytes = cluster_copy.serialize(io);
            CATCH_REQUIRE(bytes == get_serialized_size(cluster_copy));
        }

        {
            auto io = svs::lib::open_read(file);
            auto cluster_des = inverted::Cluster<uint32_t>::deserialize(io);
            auto cluster_copy_des = inverted::Cluster<uint32_t>::deserialize(io);

            CATCH_REQUIRE(cluster_des == cluster);
            CATCH_REQUIRE(cluster_copy_des == cluster_copy);
        }
    }

    CATCH_SECTION("Clustering") {
        auto ids = std::vector<uint32_t>({3, 1, 2});
        auto clustering = inverted::Clustering<uint32_t>(ids.begin(), ids.end());

        CATCH_REQUIRE(clustering.size() == ids.size());
        // Element access
        for (auto id : ids) {
            CATCH_REQUIRE(clustering.contains(id));
            CATCH_REQUIRE(clustering.at(id).size() == 0);
            CATCH_REQUIRE(std::as_const(clustering).at(id).size() == 0);
        }
        CATCH_REQUIRE(!clustering.contains(0));
        CATCH_REQUIRE_THROWS(clustering.at(0));
        CATCH_REQUIRE(clustering.total_size() == 3);
        clustering.insert(1, svs::Neighbor<uint32_t>(5, 2.0));
        clustering.insert(1, svs::Neighbor<uint32_t>(8, 3.0));

        CATCH_REQUIRE(clustering.at(1).size() == 2);
        CATCH_REQUIRE(clustering.total_size() == 5);

        auto new_cluster = inverted::Cluster<uint32_t>(10);
        new_cluster.push_back(svs::Neighbor<uint32_t>(8, 4.0));
        clustering.insert(std::move(new_cluster));
        CATCH_REQUIRE(clustering.size() == 4);
        CATCH_REQUIRE(clustering.total_size() == 7);

        auto histogram = clustering.leaf_histogram();
        CATCH_REQUIRE(histogram.size() == 2);
        CATCH_REQUIRE(histogram.at(5) == 1);
        CATCH_REQUIRE(histogram.at(8) == 2);

        {
            auto stats = clustering.statistics();
            CATCH_REQUIRE(stats.min_size_ == 0);
            CATCH_REQUIRE(stats.max_size_ == 2);
            CATCH_REQUIRE(stats.num_clusters_ == clustering.size());
            CATCH_REQUIRE(stats.empty_clusters_ == 2);
            CATCH_REQUIRE(stats.num_leaves_ == 3);
            auto report = stats.report();
            auto has_field = [&](std::string_view field) {
                return report.find(field) != report.npos;
            };
            CATCH_REQUIRE(has_field("min_size"));
            CATCH_REQUIRE(has_field("max_size"));
            CATCH_REQUIRE(has_field("empty_clusters"));
            CATCH_REQUIRE(has_field("num_clusters"));
            CATCH_REQUIRE(has_field("num_leaves"));
            CATCH_REQUIRE(has_field("mean_size"));
            CATCH_REQUIRE(has_field("std_size"));
        }

        // Complement.
        auto v = clustering.complement(10);
        CATCH_REQUIRE(v == std::vector<uint32_t>({0, 4, 5, 6, 7, 8, 9}));
        v = clustering.complement_range(svs::threads::UnitRange(9, 11));
        CATCH_REQUIRE(v == std::vector<uint32_t>({9}));

        auto test_iterator = [](auto begin, auto end) {
            auto seen = std::vector<uint32_t>();
            for (auto it = begin; it != end; ++it) {
                seen.push_back(it->first);
            }
            std::sort(seen.begin(), seen.end());
            CATCH_REQUIRE(seen == std::vector<uint32_t>({1, 2, 3, 10}));
        };
        test_iterator(clustering.begin(), clustering.end());
        test_iterator(std::as_const(clustering).begin(), std::as_const(clustering).end());
        test_iterator(clustering.cbegin(), clustering.cend());

        // Saving and loading.
        CATCH_REQUIRE(svs_test::prepare_temp_directory());
        auto dir = svs_test::temp_directory();
        svs::lib::test_self_save_load(clustering, dir);
    }
}

// End-to-end tests
namespace {

template <svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_end_to_end_clustering(
    const Data& data, Distance distance, float construction_alpha
) {
    namespace inverted = svs::index::inverted;

    auto compare = svs::distance::comparator(distance);
    double epsilon = 10;
    auto percent_centroids = svs::lib::Percent(0.10);
    auto lower_bound_percents = std::unordered_map<size_t, double>({
        {2, 0.099},
        {8, 0.099},
    });

    auto vamana_parameters = svs::index::vamana::VamanaBuildParameters{
        construction_alpha, 64, 200, 1000, 60, true};

    // Build the index once and reuse it multiple times to help speed up tests.
    for (size_t max_replicas : {2, 8}) {
        for (size_t max_cluster_size : {50, 40}) {
            // auto data =
            // svs::VectorDataLoader<float>(test_dataset::data_svs_file()).load();
            auto params = inverted::ClusteringParameters()
                              .percent_centroids(percent_centroids)
                              .epsilon(epsilon)
                              .max_replicas(max_replicas)
                              .max_cluster_size(max_cluster_size);

            auto clustering =
                randomly_cluster(data, vamana_parameters, params, distance, 2);

            clustering.sort_clusters(compare);
            auto clustering_copy = clustering;
            CATCH_REQUIRE(clustering == clustering_copy);

            // Set the required maximum cluster size to an absurdly low number.
            // Ensure that the data structure does not change when performing a dry
            // run on the cluster resizing.
            CATCH_REQUIRE(clustering_copy.reduce_maxsize(10, compare, true) == false);
            CATCH_REQUIRE(clustering_copy == clustering);
            CATCH_REQUIRE_THROWS_AS(
                clustering_copy.reduce_maxsize(10, compare, false), svs::ANNException
            );
            CATCH_REQUIRE(clustering_copy != clustering);

            // Make sure that the number of centroids is less than the perscribed
            // amount. Since some clusters can be empty and are thus absorbed back into
            // the clustering, the total number of clusters can be less than the desired
            // percent.
            //
            // Use the epsilon value below to heuristically set a lower bound.
            double lower_bound = lower_bound_percents.at(max_replicas);
            CATCH_REQUIRE(clustering.size() <= percent_centroids.value() * data.size());
            CATCH_REQUIRE(clustering.size() >= lower_bound * data.size());

            auto seen = std::vector<size_t>(data.size(), 0);
            auto set_seen = [&](size_t i) { seen.at(i) += 1; };

            // Make sure the distances between centroid and leaf elements were computed
            // properly.
            bool max_cluster_size_seen = false;
            for (const auto& list : clustering) {
                auto& cluster = list.second;
                CATCH_REQUIRE(cluster.centroid() == list.first);
                auto centroid_id = cluster.centroid();
                set_seen(centroid_id);
                auto lhs = data.get_datum(centroid_id);
                auto cluster_size = cluster.size();

                CATCH_REQUIRE(cluster_size <= max_cluster_size);
                if (cluster_size == max_cluster_size) {
                    max_cluster_size_seen = true;
                }

                for (auto neighbor : cluster.elements()) {
                    // Leaf ID's should be separate from the
                    auto id = neighbor.id();
                    CATCH_REQUIRE(!clustering.contains(id));
                    set_seen(id);
                    auto expected =
                        svs::distance::compute(distance, lhs, data.get_datum(id));
                    CATCH_REQUIRE(neighbor.distance() == expected);
                }
            }

            CATCH_REQUIRE(max_cluster_size_seen);

            // Post-process
            // Ensure:
            // * All ids have been seen.
            // * All entries have at most the maximum number of replicas.
            // * At least one entry has the maximum number of replicas.
            size_t max_seen_replicas = 0;
            auto computed_histogram = clustering.leaf_histogram();
            for (size_t i = 0, imax = data.size(); i < imax; ++i) {
                auto seen_val = seen.at(i);
                CATCH_REQUIRE(seen_val > 0);
                CATCH_REQUIRE(seen_val <= 1 + max_replicas);
                max_seen_replicas = std::max(max_seen_replicas, seen_val - 1);
                if (clustering.contains(i)) {
                    CATCH_REQUIRE(seen_val == 1);
                } else {
                    CATCH_REQUIRE(seen_val == computed_histogram.at(i));
                }
            }
            CATCH_REQUIRE(max_seen_replicas == max_replicas);
        }
    }
}

} // namespace

CATCH_TEST_CASE("Random Clustering - End to End", "[inverted][random_clustering]") {
    CATCH_SECTION("Uncompressed Data") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
        test_end_to_end_clustering(data, svs::DistanceL2(), 1.2f);
        test_end_to_end_clustering(data, svs::DistanceIP(), 0.9f);
    }
}

CATCH_TEST_CASE("Clustering with Logger", "[logging]") {
    // Setup logger
    std::vector<std::string> captured_logs;
    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&captured_logs](const spdlog::details::log_msg& msg) {
            captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    callback_sink->set_level(spdlog::level::trace); // Capture all log levels
    auto test_logger = std::make_shared<spdlog::logger>("test_logger", callback_sink);
    test_logger->set_level(spdlog::level::trace);

    // Setup cluster
    auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto vamana_parameters =
        svs::index::vamana::VamanaBuildParameters{1.2, 64, 200, 1000, 60, true};
    auto clustering_parameters = svs::index::inverted::ClusteringParameters()
                                     .percent_centroids(svs::lib::Percent(0.1))
                                     .epsilon(0.05)
                                     .max_replicas(12)
                                     .max_cluster_size(300);
    auto centroids = svs::index::inverted::randomly_select_centroids(
        data.size(),
        svs::lib::narrow_cast<size_t>(
            std::floor(data.size() * clustering_parameters.percent_centroids_.value())
        ),
        clustering_parameters.seed_
    );
    auto threadpool = svs::threads::DefaultThreadPool(2);
    auto index = svs::index::inverted::build_primary_index(
        data,
        svs::lib::as_const_span(centroids),
        vamana_parameters,
        svs::DistanceL2(),
        std::move(threadpool)
    );
    auto clustering = svs::index::inverted::cluster_with(
        data, svs::lib::as_const_span(centroids), clustering_parameters, index, test_logger
    );

    // Verify the internal log messages
    CATCH_REQUIRE(captured_logs[0].find("Processing batch") != std::string::npos);
}