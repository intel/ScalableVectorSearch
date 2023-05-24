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

// Header under test.
#include "svs/core/recall.h"

// Helpers
#include "svs/core/data.h"
#include "svs/lib/array.h"

// Test helpers
#include "tests/utils/generators.h"

// Catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <initializer_list>
#include <span>

// Tests
namespace {
template <svs::data::MemoryDataset Data, typename T>
void set(Data& data, size_t i, const std::initializer_list<T>& x) {
    data.set_datum(i, std::span(x.begin(), x.end()));
}

} // namespace

CATCH_TEST_CASE("Recall", "[core][recall]") {
    CATCH_SECTION("Simple Recall") {
        // Allocate mock containers.
        // For now, only use a single entry.
        auto groundtruth = svs::data::SimpleData<int64_t>(1, 4);
        auto results = svs::data::SimpleData<int64_t>(1, 8);

        set(groundtruth, 0, {1, 2, 3, 4});
        set(results, 0, {1, 0, 5, 6, 7, 2, 3, 4});
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 1, 1) == 1.0);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 2, 2) == 0.5);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 3, 3) == 1.0 / 3.0);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 4, 4) == 0.25);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 4, 5) == 0.25);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 4, 6) == 0.5);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 4, 7) == 0.75);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 4, 8) == 1.0);

        // Make the number of returned results smaller than the groundtruth.
        results = svs::data::SimpleData<int64_t>(1, 2);
        set(results, 0, {0, 2});
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 1, 2) == 0);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results, 2, 2) == 0.5);
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, results) == 0.5);

        // set "k" > "n"
        CATCH_REQUIRE_THROWS_AS(
            svs::k_recall_at_n(groundtruth, results, 2, 1), svs::ANNException
        );

        // set "n" > data.dimensions()
        CATCH_REQUIRE_THROWS_AS(
            svs::k_recall_at_n(groundtruth, results, 2, 5), svs::ANNException
        );

        // Set "k" > groundtruth.dimensinos()
        results = svs::data::SimpleData<int64_t>(1, 10);
        CATCH_REQUIRE_THROWS_AS(
            svs::k_recall_at_n(groundtruth, results), svs::ANNException
        );

        CATCH_REQUIRE_THROWS_AS(
            svs::k_recall_at_n(groundtruth, results, 5, 5), svs::ANNException
        );
    }

    // Bulk recall.
    CATCH_SECTION("Bulk Recall") {
        const size_t num_queries = 16;

        auto groundtruth_row = svs::data::SimpleData<int64_t>(1, 4);
        auto groundtruth = svs::data::SimpleData<int64_t>(num_queries, 4);

        auto results_row = svs::data::SimpleData<int64_t>(1, 8);
        auto results = svs::data::SimpleData<int64_t>(num_queries, 8);

        auto buffer = std::vector<int64_t>();
        auto generator = svs_test::make_generator<int64_t>(0, 100);
        double sum = 0;
        for (size_t i = 0; i < num_queries; ++i) {
            svs_test::populate(buffer, generator, groundtruth.dimensions());
            groundtruth_row.set_datum(0, svs::lib::as_span(buffer));
            groundtruth.set_datum(i, svs::lib::as_span(buffer));

            svs_test::populate(buffer, generator, results.dimensions());
            results_row.set_datum(0, svs::lib::as_span(buffer));
            results.set_datum(i, svs::lib::as_span(buffer));

            sum += svs::k_recall_at_n(groundtruth_row, results_row, 3, 5);
        }
        auto mean = sum / num_queries;
        CATCH_REQUIRE(
            svs::k_recall_at_n(groundtruth, results, 3, 5) ==
            Catch::Approx(mean).epsilon(0.00001)
        );
    }
}
