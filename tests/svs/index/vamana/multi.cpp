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

// header under test
#include "svs/index/vamana/multi.h"

// svstest
#include "tests/utils/test_dataset.h"
#include "tests/utils/vamana_reference.h"

// catch2
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

template <typename Distance> float pick_alpha(Distance SVS_UNUSED(dist)) {
    if constexpr (std::is_same_v<Distance, svs::DistanceL2>) {
        return 1.2;
    } else if constexpr (std::is_same_v<Distance, svs::DistanceIP>) {
        return 0.95;
    } else if constexpr (std::is_same_v<Distance, svs::DistanceCosineSimilarity>) {
        return 0.95;
    } else {
        throw ANNEXCEPTION("Unsupported distance type!");
    }
}

} // namespace

CATCH_TEMPLATE_TEST_CASE(
    "Multi-vector dynamic vamana index",
    "[long][index][vamana][multi]",
    svs::DistanceL2,
    svs::DistanceIP,
    svs::DistanceCosineSimilarity
) {
    using Eltype = float;
    using Distance = TestType;
    const size_t N = 128;
    const size_t max_degree = 64;
    const float alpha = pick_alpha(Distance());
    const size_t num_threads = 4;
    const size_t num_neighbors = 10;

    const auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    const auto num_points = data.size();
    const auto queries = test_dataset::queries();
    const auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);

    const svs::index::vamana::VamanaBuildParameters build_parameters{
        alpha, max_degree, 2 * max_degree, 1000, max_degree - 4, true};

    const auto search_parameters = svs::index::vamana::VamanaSearchParameters();

    const float epsilon = 0.05f;
    std::vector<size_t> ref_indices(num_points);
    std::iota(ref_indices.begin(), ref_indices.end(), 0);

    auto ref_index = svs::index::vamana::MutableVamanaIndex(
        build_parameters, data, ref_indices, Distance(), num_threads
    );
    auto ref_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    ref_index.search(ref_results.view(), queries.view(), search_parameters);
    auto ref_recall = svs::k_recall_at_n(groundtruth, ref_results);

    // Original data label:
    // 0 1 2 3
    //
    // For each duplicate iteration, insert each vector with label increase by one
    // Suppose we duplicate three times (i.e., num_duplicated = 3):
    //   1 2 3 4
    //     2 3 4 5
    //       3 4 5 6
    //
    // After deleting all the original labels, the remaining
    // number of vectors will be :
    // (num_duplicated * (num_duplicated + 1)) / 2
    //
    // For the above examples, after deleting 0, 1, 2, 3
    // the remaining vectors becomes:
    //         4
    //         4 5
    //         4 5 6
    // And the number of remaining vectors becomes
    // (3 + 4) / 2 = 6 vectors
    CATCH_SECTION("Insertion/Deletion in duplicated test datasets") {
        const size_t num_duplicated = 3;

        std::vector<size_t> test_indices(num_points);
        std::iota(test_indices.begin(), test_indices.end(), 0);

        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );

        for (size_t i = 0; i < num_duplicated; ++i) {
            std::iota(test_indices.begin(), test_indices.end(), i + 1);
            test_index.add_points(data, test_indices);
        }
        CATCH_REQUIRE(test_index.labelcount() == ref_index.size() + num_duplicated);
        CATCH_REQUIRE(test_index.size() == ref_index.size() * (num_duplicated + 1));

        std::iota(test_indices.begin(), test_indices.end(), 0);
        test_index.delete_entries(test_indices);
        CATCH_REQUIRE(test_index.labelcount() == num_duplicated);
        CATCH_REQUIRE(test_index.size() == (num_duplicated * (num_duplicated + 1)) / 2);
    }
    CATCH_SECTION("Duplicated vectors with same labels") {
        const size_t num_duplicated = 3;

        std::vector<size_t> test_indices(num_points);
        std::iota(test_indices.begin(), test_indices.end(), 0);

        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );

        for (size_t i = 0; i < num_duplicated; ++i) {
            test_index.add_points(data, test_indices);
        }
        CATCH_REQUIRE(test_index.labelcount() == test_indices.size());
        CATCH_REQUIRE(test_index.size() == test_indices.size() * (num_duplicated + 1));

        auto test_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index.search(test_results.view(), queries.view(), search_parameters);
        auto test_recall = svs::k_recall_at_n(groundtruth, test_results);

        CATCH_REQUIRE(test_recall > ref_recall - epsilon);

        test_index.delete_entries(test_indices);
        CATCH_REQUIRE(test_index.labelcount() == 0);
        CATCH_REQUIRE(test_index.size() == 0);

        test_index.add_points(data, test_indices);
        test_index.consolidate();
        test_index.compact();
        for (size_t i = 0; i < num_duplicated; ++i) {
            test_index.add_points(data, test_indices);
        }

        auto test_results2 = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index.search(test_results2.view(), queries.view(), search_parameters);
        auto test_recall2 = svs::k_recall_at_n(groundtruth, test_results2);

        CATCH_REQUIRE(test_recall2 > test_recall - epsilon);
        CATCH_REQUIRE(test_recall2 < test_recall + epsilon);
    }

    CATCH_SECTION("Step grouping") {
        size_t start = 0;
        size_t step = 4;
        CATCH_REQUIRE(num_points % step == 0);
        size_t num_groups = num_points / step;

        auto remapped_groundtruth = groundtruth;
        CATCH_REQUIRE(remapped_groundtruth.size() == queries.size());

        // It is okay to have duplicated neighbor ids in groundtruth
        // as the recall is checked by counting intersect
        for (size_t i = 0; i < queries.size(); ++i) {
            auto arr = remapped_groundtruth.get_datum(i);
            for (auto& each : arr) {
                each /= step;
            }
        }

        std::vector<size_t> test_indices(num_points);
        for (size_t i = 0; i < num_points; i += step) {
            for (size_t s = 0; s < step; ++s) {
                test_indices[i + s] = start;
            }
            ++start;
        }

        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );
        test_index.add_points(data, test_indices);

        auto test_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index.search(test_results.view(), queries.view(), search_parameters);
        auto test_recall = svs::k_recall_at_n(remapped_groundtruth, test_results);

        CATCH_REQUIRE(test_recall > ref_recall - epsilon);

        // test get_distance
        for (size_t i = 0; i < queries.size(); ++i) {
            size_t k = std::rand() % num_groups;
            double ref_distance = svs::INVALID_DISTANCE;
            for (size_t s = 0; s < step; ++s) {
                if constexpr (std::is_same_v<Distance, svs::distance::DistanceL2>) {
                    ref_distance = std::fmin(
                        ref_distance,
                        ref_index.get_distance(
                            ref_indices[k * step + s], queries.get_datum(i)
                        )
                    );
                } else {
                    ref_distance = std::fmax(
                        ref_distance,
                        ref_index.get_distance(
                            ref_indices[k * step + s], queries.get_datum(i)
                        )
                    );
                }
            }

            double test_distance =
                test_index.get_distance(test_indices[k * step], queries.get_datum(i));
            CATCH_REQUIRE(test_distance == ref_distance);
        }
    }

    CATCH_SECTION("Logging") {
        std::vector<size_t> test_indices(num_points);
        std::iota(test_indices.begin(), test_indices.end(), 0);

        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );

        CATCH_REQUIRE(ref_index.get_logger() == test_index.get_logger());
    }

    CATCH_SECTION("Save/Load") {
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        auto config_dir = dir / "config";
        auto graph_dir = dir / "graph";
        auto data_dir = dir / "data";
        std::vector<size_t> test_indices(num_points);
        // Fill the test indices with labels in the range of num_labels
        // to ensure that there are labels mapped to more than 1 vector.
        const size_t per_label = 2;
        const auto num_labels = num_points / per_label;
        for (auto& i : test_indices) {
            i = std::rand() % num_labels;
        }
        auto test_index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, test_indices, Distance(), num_threads
        );
        auto test_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index.search(test_results.view(), queries.view(), search_parameters);
        auto test_recall = svs::k_recall_at_n(groundtruth, test_results);

        test_index.save(config_dir, graph_dir, data_dir);

        auto test_index_2 = svs::index::vamana::auto_multi_dynamic_assemble(
            config_dir,
            svs::GraphLoader(graph_dir),
            svs::VectorDataLoader<float>(data_dir),
            Distance(),
            svs::threads::CppAsyncThreadPool(2)
        );
        auto test_results_2 = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        test_index_2.search(test_results_2.view(), queries.view(), search_parameters);
        auto test_recall_2 = svs::k_recall_at_n(groundtruth, test_results_2);

        // Check that the results are the same
        CATCH_REQUIRE(test_results.n_neighbors() == test_results_2.n_neighbors());
        for (size_t i = 0; i < test_results.n_queries(); ++i) {
            for (size_t j = 0; j < test_results.n_neighbors(); ++j) {
                CATCH_REQUIRE(
                    test_results.indices().at(i, j) == test_results_2.indices().at(i, j)
                );
            }
        }

        CATCH_REQUIRE(test_index.size() == test_index_2.size());
        CATCH_REQUIRE(test_index.dimensions() == test_index_2.dimensions());
        // Index Properties
        CATCH_REQUIRE(test_index.get_alpha() == test_index_2.get_alpha());
        CATCH_REQUIRE(
            test_index.get_construction_window_size() ==
            test_index_2.get_construction_window_size()
        );
        CATCH_REQUIRE(test_index.get_max_candidates() == test_index_2.get_max_candidates());
        CATCH_REQUIRE(test_index.max_degree() == test_index_2.max_degree());
        CATCH_REQUIRE(test_index.get_prune_to() == test_index_2.get_prune_to());
        CATCH_REQUIRE(
            test_index.get_full_search_history() == test_index_2.get_full_search_history()
        );
        CATCH_REQUIRE(test_index.view_data() == test_index_2.view_data());

        CATCH_REQUIRE(test_recall_2 > test_recall - epsilon);
    }
}

CATCH_TEST_CASE(
    "MultiMutableVamana Index Save and Load", "[index][vamana][multi][saveload]"
) {
    using Eltype = float;
    using Distance = svs::DistanceL2;
    const size_t N = 128;
    const size_t num_threads = 4;
    const size_t num_neighbors = 10;
    const size_t max_degree = 64;

    const auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    const auto num_points = data.size();
    const auto queries = test_dataset::queries();
    const auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);

    const svs::index::vamana::VamanaBuildParameters build_parameters{
        1.2, max_degree, 10, 20, 10, true};

    const auto search_parameters = svs::index::vamana::VamanaSearchParameters();

    const float epsilon = 0.05f;

    std::vector<size_t> test_indices(num_points);
    const size_t per_label = 2;
    const auto num_labels = num_points / per_label;
    for (auto& i : test_indices) {
        i = std::rand() % num_labels;
    }

    auto index = svs::index::vamana::MultiMutableVamanaIndex(
        build_parameters, data, test_indices, Distance(), num_threads
    );
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(results.view(), queries.view(), search_parameters);

    CATCH_SECTION("Load MultiMutableVamana Index being serialized natively to stream") {
        std::stringstream stream;
        index.save(stream);
        {
            auto deserializer = svs::lib::detail::Deserializer::build(stream);
            CATCH_REQUIRE(deserializer.is_native());

            using Data_t = svs::data::SimpleData<Eltype, N>;
            using GraphType = svs::graphs::SimpleBlockedGraph<uint32_t>;

            auto loaded = svs::index::vamana::auto_multi_dynamic_assemble(
                stream,
                [&]() -> GraphType { return GraphType::load(stream); },
                [&]() -> Data_t { return svs::lib::load_from_stream<Data_t>(stream); },
                Distance(),
                num_threads
            );

            CATCH_REQUIRE(loaded.size() == index.size());
            CATCH_REQUIRE(loaded.dimensions() == index.dimensions());
            CATCH_REQUIRE(loaded.get_alpha() == index.get_alpha());
            CATCH_REQUIRE(
                loaded.get_construction_window_size() ==
                index.get_construction_window_size()
            );
            CATCH_REQUIRE(loaded.get_max_candidates() == index.get_max_candidates());
            CATCH_REQUIRE(loaded.max_degree() == index.max_degree());
            CATCH_REQUIRE(loaded.get_prune_to() == index.get_prune_to());
            CATCH_REQUIRE(
                loaded.get_full_search_history() == index.get_full_search_history()
            );
            CATCH_REQUIRE(loaded.view_data() == index.view_data());

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
            loaded.search(loaded_results.view(), queries.view(), search_parameters);
            for (size_t i = 0; i < results.n_queries(); ++i) {
                for (size_t j = 0; j < results.n_neighbors(); ++j) {
                    CATCH_REQUIRE(
                        results.indices().at(i, j) == loaded_results.indices().at(i, j)
                    );
                }
            }

            auto loaded_recall = svs::k_recall_at_n(groundtruth, loaded_results);
            auto test_recall = svs::k_recall_at_n(groundtruth, results);
            CATCH_REQUIRE(loaded_recall > test_recall - epsilon);
        }
    }

    CATCH_SECTION("Load MultiMutableVamana Index being serialized with intermediate files"
    ) {
        std::stringstream stream;
        svs::lib::UniqueTempDirectory tempdir{"svs_multivamana_save"};
        const auto config_dir = tempdir.get() / "config";
        const auto graph_dir = tempdir.get() / "graph";
        const auto data_dir = tempdir.get() / "data";
        std::filesystem::create_directories(config_dir);
        std::filesystem::create_directories(graph_dir);
        std::filesystem::create_directories(data_dir);
        index.save(config_dir, graph_dir, data_dir);
        svs::lib::DirectoryArchiver::pack(tempdir, stream);
        {
            using Data_t = svs::data::SimpleData<Eltype, N>;
            using GraphType = svs::graphs::SimpleBlockedGraph<uint32_t>;

            auto deserializer = svs::lib::detail::Deserializer::build(stream);
            CATCH_REQUIRE(!deserializer.is_native());
            svs::lib::DirectoryArchiver::unpack(stream, tempdir, deserializer.magic());

            auto loaded = svs::index::vamana::auto_multi_dynamic_assemble(
                config_dir,
                GraphType::load(graph_dir),
                Data_t::load(data_dir),
                Distance(),
                num_threads
            );

            CATCH_REQUIRE(loaded.size() == index.size());
            CATCH_REQUIRE(loaded.dimensions() == index.dimensions());
            CATCH_REQUIRE(loaded.get_alpha() == index.get_alpha());
            CATCH_REQUIRE(
                loaded.get_construction_window_size() ==
                index.get_construction_window_size()
            );
            CATCH_REQUIRE(loaded.get_max_candidates() == index.get_max_candidates());
            CATCH_REQUIRE(loaded.max_degree() == index.max_degree());
            CATCH_REQUIRE(loaded.get_prune_to() == index.get_prune_to());
            CATCH_REQUIRE(
                loaded.get_full_search_history() == index.get_full_search_history()
            );
            CATCH_REQUIRE(loaded.view_data() == index.view_data());

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
            loaded.search(loaded_results.view(), queries.view(), search_parameters);
            for (size_t i = 0; i < results.n_queries(); ++i) {
                for (size_t j = 0; j < results.n_neighbors(); ++j) {
                    CATCH_REQUIRE(
                        results.indices().at(i, j) == loaded_results.indices().at(i, j)
                    );
                }
            }

            auto loaded_recall = svs::k_recall_at_n(groundtruth, loaded_results);
            auto test_recall = svs::k_recall_at_n(groundtruth, results);
            CATCH_REQUIRE(loaded_recall > test_recall - epsilon);
        }
    }
}
