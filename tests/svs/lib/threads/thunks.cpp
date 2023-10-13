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

// unit under test
#include "svs/lib/threads/thunks.h"

// svs
#include "svs/lib/exception.h"

// tests
#include "tests/utils/generators.h"

// catch macros
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <string>
#include <vector>

CATCH_TEST_CASE("Thread Thunks", "[core][threads]") {
    namespace threads = svs::threads;
    std::vector<size_t> v{};
    std::vector<size_t> u{};
    CATCH_SECTION("Default Thunk") {
        auto f = [&v](uint64_t i) { v.push_back(i); };
        auto wrapped = threads::thunks::wrap(threads::ThreadCount{1}, f);
        auto f_ref = threads::FunctionRef(wrapped);
        CATCH_REQUIRE(f_ref.arg == &wrapped);
        CATCH_REQUIRE(v.empty());
        f_ref(1);
        f_ref(2);
        f_ref(3);
        CATCH_REQUIRE(v.size() == 3);
        CATCH_REQUIRE(v.at(0) == 1);
        CATCH_REQUIRE(v.at(1) == 2);
        CATCH_REQUIRE(v.at(2) == 3);

        threads::ThreadFunctionRef g{f_ref, 10};
        CATCH_REQUIRE(g.fn.arg == f_ref.arg);
        CATCH_REQUIRE(g.fn.fn == f_ref.fn);
        g();
        CATCH_REQUIRE(v.size() == 4);
        CATCH_REQUIRE(v.at(3) == 10);
    }

    CATCH_SECTION("Static Index Partition") {
        auto f = [&v, &u](const auto& indices, uint64_t id) {
            for (auto i : indices) {
                v.push_back(i);
                u.push_back(id);
            }
        };
        // Partitions:
        //
        // 0   1   2   3   4   5   6   7   8   9
        // |       |   |       |   |   |   |   |
        // +-------+   +-------+   +---+   +---+
        //    T0          T1         T2      T3
        auto thunk =
            threads::thunks::wrap(threads::ThreadCount{4}, f, threads::StaticPartition{10});
        auto wrapped = threads::FunctionRef(thunk);

        CATCH_REQUIRE(v.empty());
        CATCH_REQUIRE(u.empty());

        ///// Parition 3
        wrapped(3);
        CATCH_REQUIRE(v.size() == 2);
        CATCH_REQUIRE(u.size() == 2);

        CATCH_REQUIRE(v.at(0) == 8);
        CATCH_REQUIRE(v.at(1) == 9);
        CATCH_REQUIRE(u.at(0) == 3);
        CATCH_REQUIRE(u.at(1) == 3);
        v.clear();
        u.clear();

        ///// Partition 0
        wrapped(0);
        CATCH_REQUIRE(v.size() == 3);
        CATCH_REQUIRE(u.size() == 3);

        CATCH_REQUIRE(v.at(0) == 0);
        CATCH_REQUIRE(v.at(1) == 1);
        CATCH_REQUIRE(v.at(2) == 2);

        CATCH_REQUIRE(u.at(0) == 0);
        CATCH_REQUIRE(u.at(1) == 0);
        CATCH_REQUIRE(u.at(2) == 0);
        v.clear();
        u.clear();

        ///// Partition 2
        wrapped(2);
        CATCH_REQUIRE(v.size() == 2);
        CATCH_REQUIRE(u.size() == 2);

        CATCH_REQUIRE(v.at(0) == 6);
        CATCH_REQUIRE(v.at(1) == 7);

        CATCH_REQUIRE(u.at(0) == 2);
        CATCH_REQUIRE(u.at(1) == 2);
        v.clear();
        u.clear();

        ///// Partition 1
        wrapped(1);
        CATCH_REQUIRE(v.size() == 3);
        CATCH_REQUIRE(u.size() == 3);

        CATCH_REQUIRE(v.at(0) == 3);
        CATCH_REQUIRE(v.at(1) == 4);
        CATCH_REQUIRE(v.at(2) == 5);

        CATCH_REQUIRE(u.at(0) == 1);
        CATCH_REQUIRE(u.at(1) == 1);
        CATCH_REQUIRE(u.at(2) == 1);
        v.clear();
        u.clear();
    }

    CATCH_SECTION("Static Partition over Vectors") {
        std::vector<size_t> input(10);
        svs_test::populate(input, svs_test::make_generator<size_t>(0, 100));

        auto partition = threads::StaticPartition{input};
        auto f = [&v, &u](const auto& indices, uint64_t id) {
            for (auto i : indices) {
                v.push_back(i);
                u.push_back(id);
            }
        };

        auto thunk =
            threads::thunks::wrap(threads::ThreadCount{4}, f, partition);
        auto wrapped = threads::FunctionRef(thunk);

        CATCH_REQUIRE(v.empty());
        CATCH_REQUIRE(u.empty());

        ///// Parition 3
        wrapped(3);
        CATCH_REQUIRE(v.size() == 2);
        CATCH_REQUIRE(u.size() == 2);

        CATCH_REQUIRE(v.at(0) == input.at(8));
        CATCH_REQUIRE(v.at(1) == input.at(9));
        CATCH_REQUIRE(u.at(0) == 3);
        CATCH_REQUIRE(u.at(1) == 3);
        v.clear();
        u.clear();

        ///// Partition 0
        wrapped(0);
        CATCH_REQUIRE(v.size() == 3);
        CATCH_REQUIRE(u.size() == 3);

        CATCH_REQUIRE(v.at(0) == input.at(0));
        CATCH_REQUIRE(v.at(1) == input.at(1));
        CATCH_REQUIRE(v.at(2) == input.at(2));

        CATCH_REQUIRE(u.at(0) == 0);
        CATCH_REQUIRE(u.at(1) == 0);
        CATCH_REQUIRE(u.at(2) == 0);
        v.clear();
        u.clear();

        ///// Partition 2
        wrapped(2);
        CATCH_REQUIRE(v.size() == 2);
        CATCH_REQUIRE(u.size() == 2);

        CATCH_REQUIRE(v.at(0) == input.at(6));
        CATCH_REQUIRE(v.at(1) == input.at(7));

        CATCH_REQUIRE(u.at(0) == 2);
        CATCH_REQUIRE(u.at(1) == 2);
        v.clear();
        u.clear();

        ///// Partition 1
        wrapped(1);
        CATCH_REQUIRE(v.size() == 3);
        CATCH_REQUIRE(u.size() == 3);

        CATCH_REQUIRE(v.at(0) == input.at(3));
        CATCH_REQUIRE(v.at(1) == input.at(4));
        CATCH_REQUIRE(v.at(2) == input.at(5));

        CATCH_REQUIRE(u.at(0) == 1);
        CATCH_REQUIRE(u.at(1) == 1);
        CATCH_REQUIRE(u.at(2) == 1);
        v.clear();
        u.clear();
    }

    CATCH_SECTION("Dunamic Index Partition") {
        auto f = [&v, &u](const auto& indices, uint64_t id) {
            for (auto i : indices) {
                v.push_back(i);
                u.push_back(id);
            }
        };
        // Partitions:
        //
        // 0   1   2   3   4   5   6   7   8   9
        // |       |   |       |   |       |   |
        // +-------+   +-------+   +-------+   |
        //    1st         2nd         3rd     4th
        auto thunk = threads::thunks::wrap(
            threads::ThreadCount{4}, f, threads::DynamicPartition{10, 3}
        );
        auto wrapped = threads::FunctionRef(thunk);

        CATCH_REQUIRE(v.empty());
        CATCH_REQUIRE(u.empty());

        // With dynamic ordering, calling the wrapped function will execute the entire
        // workload because it loops until the the range has been satisfied.
        wrapped(3);
        CATCH_REQUIRE(v.size() == 10);
        CATCH_REQUIRE(u.size() == 10);

        for (size_t i = 0; i < v.size(); ++i) {
            CATCH_REQUIRE(v.at(i) == i);
            CATCH_REQUIRE(u.at(i) == 3);
        }
        v.clear();
        u.clear();
    }
}
