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
#include "catch2/catch_test_macros.hpp"

// stl
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Eltype = float;
using QueryEltype = float;
using Distance = svs::distance::DistanceL2;

CATCH_TEST_CASE("Vamana Multi", "[index][vamana][multi]") {
    const size_t N = 128;
    const size_t max_degree = 64;
    const float alpha = 1.2;
    const size_t num_threads = 4;
    const size_t num_neighbors = 10;

    const auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    const auto num_points = data.size();
    const auto queries = test_dataset::queries();
    const auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);

    const svs::index::vamana::VamanaBuildParameters build_parameters{
        alpha, max_degree, 2 * max_degree, 1000, max_degree - 4, true};

    const auto search_parameters = svs::index::vamana::VamanaSearchParameters();
    const size_t num_duplicated = 3;

    float epsilon = 0.005f;

    CATCH_SECTION("Insertion/Deletion in duplicated test datasets") {
        std::vector<size_t> indices(num_points);
        std::iota(indices.begin(), indices.end(), 0);

        auto index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, indices, Distance(), num_threads
        );

        for (size_t i = 0; i < num_duplicated; ++i) {
            std::iota(indices.begin(), indices.end(), i + 1);
            index.add_points(data, indices);
        }
        CATCH_REQUIRE(index.size() == indices.size() + num_duplicated);
        CATCH_REQUIRE(
            index.get_parent_index().size() == indices.size() * (num_duplicated + 1)
        );

        std::iota(indices.begin(), indices.end(), 0);
        index.delete_entries(indices);
        CATCH_REQUIRE(index.size() == num_duplicated);
        CATCH_REQUIRE(
            index.get_parent_index().size() == (num_duplicated * (num_duplicated + 1)) / 2
        );
    }
    CATCH_SECTION("Same vector with same labels") {
        std::vector<size_t> indices(num_points);
        std::iota(indices.begin(), indices.end(), 0);

        auto index = svs::index::vamana::MultiMutableVamanaIndex(
            build_parameters, data, indices, Distance(), num_threads
        );
        auto ref_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        index.search(ref_results.view(), queries.view(), search_parameters);
        auto ref_recall = svs::k_recall_at_n(groundtruth, ref_results);

        for (size_t i = 0; i < num_duplicated; ++i) {
            index.add_points(data, indices);
        }
        CATCH_REQUIRE(index.size() == indices.size());
        CATCH_REQUIRE(
            index.get_parent_index().size() == indices.size() * (num_duplicated + 1)
        );

        auto test_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        index.search(test_results.view(), queries.view(), search_parameters);
        auto test_recall = svs::k_recall_at_n(groundtruth, test_results);

        CATCH_REQUIRE(test_recall > ref_recall - epsilon);

        index.delete_entries(indices);
        CATCH_REQUIRE(index.size() == 0);
        CATCH_REQUIRE(index.get_parent_index().size() == 0);
    }

    CATCH_SECTION("Get distance") {}
}
