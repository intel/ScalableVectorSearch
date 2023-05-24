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

// stdlib
#include <functional>
#include <type_traits>

// svs
#include "svs/index/flat/inserters.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// test
#include "tests/utils/generators.h"
#include "tests/utils/utils.h"

namespace {
template <typename T, typename Cmp>
void test_bulk_inserter(svs::index::flat::BulkInserter<T, Cmp>& inserter) {
    // Allocate per-lane insertion values.
    auto batchsize = inserter.batch_size();

    auto generator = svs_test::make_generator<T>(0, 100);
    std::vector<std::vector<T>> values(batchsize);
    CATCH_REQUIRE(values.size() == batchsize);

    // Populate each test vector.
    const size_t test_length = 1000;
    for (auto& v : values) {
        svs_test::populate(v, generator, test_length);
    }

    inserter.prepare();
    for (size_t i = 0; i < test_length; ++i) {
        for (size_t j = 0; j < batchsize; ++j) {
            auto v = values.at(j).at(i);
            inserter.insert(j, v);
        }
    }
    inserter.cleanup();

    // Generate groundtruth results.
    size_t result_index = 0;
    for (auto& v : values) {
        std::sort(v.begin(), v.end(), Cmp{});
        auto result = inserter.result(result_index);
        CATCH_REQUIRE(v.size() >= result.size());
        CATCH_REQUIRE(result.size() == inserter.num_neighbors());
        CATCH_REQUIRE(std::equal(result.begin(), result.end(), v.begin()));

        ++result_index;
    }
}
} // namespace

CATCH_TEST_CASE("Sorters", "[core][sorters]") {
    CATCH_SECTION("Linear Inserter") {
        std::vector<int> x(3);
        int sentinel = std::numeric_limits<int>::max();
        CATCH_REQUIRE(x.size() == 3);

        auto inserter = svs::index::flat::LinearInserter(x.begin(), x.end(), std::less<>{});
        inserter.prepare();

        CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [sentinel](int i) {
            return i == sentinel;
        }));

        // Insert in the front.
        inserter.insert(10);
        CATCH_REQUIRE(x[0] == 10);
        for (size_t i = 1; i < x.size(); ++i) {
            CATCH_REQUIRE(x[i] == sentinel);
        }

        // Insert after previously inserted element.
        inserter.insert(20);
        CATCH_REQUIRE(x[0] == 10);
        CATCH_REQUIRE(x[1] == 20);
        for (size_t i = 2; i < x.size(); ++i) {
            CATCH_REQUIRE(x[i] == sentinel);
        }

        // Insert at the front.
        // Now all elements are valid.
        inserter.insert(5);
        CATCH_REQUIRE(x[0] == 5);
        CATCH_REQUIRE(x[1] == 10);
        CATCH_REQUIRE(x[2] == 20);

        // Insert off the end.
        inserter.insert(100);
        CATCH_REQUIRE(x[0] == 5);
        CATCH_REQUIRE(x[1] == 10);
        CATCH_REQUIRE(x[2] == 20);

        // Insert in the middle
        inserter.insert(15);
        CATCH_REQUIRE(x[0] == 5);
        CATCH_REQUIRE(x[1] == 10);
        CATCH_REQUIRE(x[2] == 15);
    }

    CATCH_SECTION("Heap Inserter") {
        CATCH_SECTION("Less Than") {
            std::vector<int> x(3);
            int sentinel = std::numeric_limits<int>::max();
            CATCH_REQUIRE(x.size() == 3);

            auto inserter =
                svs::index::flat::HeapInserter(x.begin(), x.end(), std::less<>{});
            inserter.prepare();

            CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [sentinel](int i) {
                return i == sentinel;
            }));

            for (auto i : {10, 1, 5, 2, 100, 3}) {
                inserter.insert(i);
            }
            inserter.cleanup();
            CATCH_REQUIRE(x.at(0) == 1);
            CATCH_REQUIRE(x.at(1) == 2);
            CATCH_REQUIRE(x.at(2) == 3);
        }

        CATCH_SECTION("Greater Than") {
            std::vector<int> x(3);
            int sentinel = std::numeric_limits<int>::min();
            CATCH_REQUIRE(x.size() == 3);

            auto inserter =
                svs::index::flat::HeapInserter(x.begin(), x.end(), std::greater<>{});
            inserter.prepare();

            CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [sentinel](int i) {
                return i == sentinel;
            }));

            for (auto i : {10, 1, 5, 2, 100, 3}) {
                inserter.insert(i);
            }
            inserter.cleanup();
            CATCH_REQUIRE(x.at(0) == 100);
            CATCH_REQUIRE(x.at(1) == 10);
            CATCH_REQUIRE(x.at(2) == 5);
        }
    }

    CATCH_SECTION("Bulk Inserter") {
        auto inserter =
            svs::index::flat::BulkInserter<float, std::less<>>{200, 50, std::less{}};
        CATCH_REQUIRE(inserter.batch_size() == 200);
        CATCH_REQUIRE(inserter.num_neighbors() == 50);
        test_bulk_inserter(inserter);

        // Change batch size.
        inserter.resize_batch(123);
        CATCH_REQUIRE(inserter.batch_size() == 123);
        CATCH_REQUIRE(inserter.num_neighbors() == 50);
        test_bulk_inserter(inserter);

        // Change number of neighbors.
        inserter.resize_neighbors(10);
        CATCH_REQUIRE(inserter.batch_size() == 123);
        CATCH_REQUIRE(inserter.num_neighbors() == 10);
        test_bulk_inserter(inserter);

        // Increase the batch size greater than the original amount.
        inserter.resize_batch(250);
        CATCH_REQUIRE(inserter.batch_size() == 250);
        CATCH_REQUIRE(inserter.num_neighbors() == 10);
        test_bulk_inserter(inserter);

        // Increase the number of neighbors
        inserter.resize_neighbors(70);
        CATCH_REQUIRE(inserter.batch_size() == 250);
        CATCH_REQUIRE(inserter.num_neighbors() == 70);
        test_bulk_inserter(inserter);
    }
}
