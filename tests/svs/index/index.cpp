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
#include "svs/index/index.h"

// svs-test
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

// A minimal, light-weight index for testing the query-processing pipeline.
struct TestIndex {
  public:
    // Simple wrapper-type for search parameters.
    struct SearchParameters {
        int value_;
    };

    using search_parameters_type = SearchParameters;

    // Default search parameters.
    SearchParameters default_parameters_ = {10};
    size_t expected_num_neighbors_ = 0;
    size_t expected_num_queries_ = 0;

    search_parameters_type get_search_parameters() const { return default_parameters_; }

    void search(
        svs::QueryResultView<size_t> result,
        svs::data::ConstSimpleDataView<float> queries,
        SearchParameters p,
        const svs::lib::DefaultPredicate& cancel =
            svs::lib::Returns(svs::lib::Const<false>())
    ) const {
        CATCH_REQUIRE(result.n_neighbors() == expected_num_neighbors_);
        CATCH_REQUIRE(result.n_queries() == expected_num_queries_);
        CATCH_REQUIRE(queries.size() == expected_num_queries_);

        auto v = p.value_;
        auto vf = svs::lib::narrow_cast<float>(v);
        for (size_t i = 0, imax = result.n_queries(); i < imax; ++i) {
            for (size_t j = 0, jmax = result.n_neighbors(); j < jmax; ++j) {
                if (cancel())
                    return;
                result.set(svs::Neighbor{v, vf}, i, j);
            }
        }
    }
};

bool check_all_are(svs::QueryResultView<size_t> result, size_t value) {
    auto value_float = svs::lib::narrow<float>(value);
    for (size_t i = 0, imax = result.n_queries(); i < imax; ++i) {
        for (size_t j = 0, jmax = result.n_neighbors(); j < jmax; ++j) {
            auto id = result.index(i, j);
            auto dist = result.distance(i, j);
            if (id != value || dist != value_float) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

CATCH_TEST_CASE("Query Processing", "[index][general]") {
    auto index = TestIndex{};
    auto queries = test_dataset::queries();

    CATCH_REQUIRE(index.default_parameters_.value_ == 10);
    CATCH_REQUIRE(index.expected_num_neighbors_ == 0);
    CATCH_REQUIRE(index.expected_num_queries_ == 0);

    CATCH_SECTION("search_batch_into_with") {
        index.expected_num_queries_ = queries.size();
        index.expected_num_neighbors_ = 1;

        auto these_parameters = TestIndex::SearchParameters{100};

        // Set the default parameters to different values to ensure the externally supplied
        // parameters are given.
        index.default_parameters_ = {20};

        auto result =
            svs::QueryResult<size_t>(queries.size(), index.expected_num_neighbors_);
        svs::index::search_batch_into_with(
            index, result.view(), queries.cview(), these_parameters
        );

        // Ensure all values are set.
        CATCH_REQUIRE(check_all_are(result.view(), these_parameters.value_));
        CATCH_REQUIRE(!check_all_are(result.view(), 0));

        // Change parameters and run again.
        these_parameters = TestIndex::SearchParameters{0};
        svs::index::search_batch_into_with(
            index, result.view(), queries.cview(), these_parameters
        );

        // Ensure all values are set.
        CATCH_REQUIRE(check_all_are(result.view(), these_parameters.value_));
        CATCH_REQUIRE(!check_all_are(result.view(), 100));
    }

    CATCH_SECTION("search_batch_into") {
        index.expected_num_queries_ = queries.size();
        index.expected_num_neighbors_ = 10;

        index.default_parameters_ = TestIndex::SearchParameters{12};
        auto result =
            svs::QueryResult<size_t>(queries.size(), index.expected_num_neighbors_);

        // Ensure default parameters are provided.
        svs::index::search_batch_into(index, result.view(), queries.cview());
        CATCH_REQUIRE(check_all_are(result.view(), 12));

        // Change default parameters - check that it is propagated.
        index.default_parameters_ = {20};
        svs::index::search_batch_into(index, result.view(), queries.cview());
        CATCH_REQUIRE(check_all_are(result.view(), 20));
    }

    CATCH_SECTION("search_batch_with") {
        size_t num_neighbors = 5;

        index.expected_num_queries_ = queries.size();
        index.expected_num_neighbors_ = num_neighbors;

        index.default_parameters_ = {10};
        auto these_parameters = TestIndex::SearchParameters{5};

        auto results = svs::index::search_batch_with(
            index, queries.cview(), num_neighbors, these_parameters
        );

        CATCH_REQUIRE(check_all_are(results.view(), 5));
        CATCH_REQUIRE(!check_all_are(results.view(), 10));

        // Change parameters - ensure propagation.
        these_parameters = {2};
        results = svs::index::search_batch_with(
            index, queries.cview(), num_neighbors, these_parameters
        );
        CATCH_REQUIRE(check_all_are(results.view(), 2));
        CATCH_REQUIRE(!check_all_are(results.view(), 5));
    }

    CATCH_SECTION("search_batch") {
        size_t num_neighbors = 2;
        index.expected_num_queries_ = queries.size();
        index.expected_num_neighbors_ = num_neighbors;
        index.default_parameters_ = {123};

        // Ensure default values are used.
        auto results = svs::index::search_batch(index, queries.cview(), num_neighbors);
        CATCH_REQUIRE(check_all_are(results.view(), 123));
        CATCH_REQUIRE(!check_all_are(results.view(), 10));

        // Ensure propagation.
        index.default_parameters_ = {234};
        results = svs::index::search_batch(index, queries.cview(), num_neighbors);
        CATCH_REQUIRE(check_all_are(results.view(), 234));
        CATCH_REQUIRE(!check_all_are(results.view(), 123));
    }
}

