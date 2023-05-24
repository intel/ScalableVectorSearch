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
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>

// local includes
#include "svs/lib/threads/types.h"

// catch macros
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Thread Helper Types", "[core][threads]") {
    CATCH_SECTION("Index Iterator") {
        using IndexIterator = svs::threads::IndexIterator<size_t>;
        // Legacy Iterator.
        CATCH_SECTION("Legacy Iterator") {
            auto iter = IndexIterator{5};
            // Dereferencable.
            CATCH_REQUIRE(*iter == 5);
            // Incrementable.
            ++iter;
            CATCH_REQUIRE(*iter == 6);
            CATCH_REQUIRE(*++iter == 7);
        }

        CATCH_SECTION("Legacy Forward Iterator") {
            // Equality
            auto iter = IndexIterator{5};
            CATCH_REQUIRE(iter == IndexIterator{5});
            CATCH_REQUIRE(iter != IndexIterator{4});
            CATCH_REQUIRE(iter != IndexIterator{6});

            CATCH_REQUIRE(!(iter != IndexIterator{5}));
            CATCH_REQUIRE(!(iter == IndexIterator{4}));
            CATCH_REQUIRE(!(iter == IndexIterator{6}));

            // Post-increment.
            CATCH_REQUIRE(*iter == 5);
            auto v = iter++;
            CATCH_REQUIRE(*v == 5);
            CATCH_REQUIRE(*iter == 6);

            CATCH_REQUIRE(*iter++ == 6);
            CATCH_REQUIRE(*iter == 7);
        }

        CATCH_SECTION("Legacy Bidirectional Iterator") {
            auto a = IndexIterator{5};
            CATCH_REQUIRE(*(a--) == 5);
            CATCH_REQUIRE(*a == 4);
            CATCH_REQUIRE(*(--a) == 3);

            CATCH_REQUIRE(std::addressof(--a) == std::addressof(a));

            // Post-decrement yields the previous value of the operand.
            {
                auto a = IndexIterator{10};
                auto b = a;
                CATCH_REQUIRE(a == b);
                CATCH_REQUIRE(a-- == b);
            }

            // Post-decrement and pre-decrement perform the same modification on its
            // operand.
            {
                auto a = IndexIterator{10};
                auto b = a;
                a--;
                --b;
                CATCH_REQUIRE(a == b);
            }

            // Increemnt and decrement are inverses of eachother.
            {
                auto a = IndexIterator{10};
                auto b = a;
                CATCH_REQUIRE(--(++a) == b);
                CATCH_REQUIRE(++(--a) == b);
            }
        }

        CATCH_SECTION("Random Access Iterator") {
            auto a = IndexIterator{20};
            auto b = IndexIterator{30};
            auto n = b - a;

            CATCH_SECTION("Self Addition") {
                CATCH_REQUIRE((a += n) == b);
                CATCH_REQUIRE(std::addressof(a += n) == std::addressof(a));
            }

            CATCH_SECTION("Equivalence of addition") {
                auto x = a + n;
                CATCH_REQUIRE(x == (a += n));
            }

            CATCH_SECTION("Symmetry of addition") {
                CATCH_REQUIRE((a + n) == (n + a));
                CATCH_REQUIRE(a + (1 + 2) == (a + 1) + 2);
                CATCH_REQUIRE(a + 0 == a);
                CATCH_REQUIRE(a + (n - 1) == --b);
            }

            CATCH_SECTION("Subtraction 1") { CATCH_REQUIRE((b += -n) == a); }

            CATCH_SECTION("Subtraction 2") { CATCH_REQUIRE((b + -n) == a); }

            CATCH_SECTION("Subtraction 3") {
                CATCH_REQUIRE(std::addressof(b -= n) == std::addressof(b));
            }

            CATCH_SECTION("Subtraction 4") {
                auto x = b - n;
                CATCH_REQUIRE(x == (b -= n));
            }

            CATCH_SECTION("Indexing") { CATCH_REQUIRE(a[n] == *b); }

            CATCH_SECTION("Inequality") { CATCH_REQUIRE(a <= b); }
        }
    }

    CATCH_SECTION("Unit Range") {
        CATCH_SECTION("General behavior") {
            auto range = svs::threads::UnitRange(0, 10);
            CATCH_REQUIRE(range.size() == 10);
            CATCH_REQUIRE(range.max_size() == std::numeric_limits<size_t>::max());
            CATCH_REQUIRE(range.empty() == false);

            std::vector<size_t> v;

            // Range based for-loops.
            for (auto i : range) {
                v.push_back(i);
            }
            for (size_t j = 0; j < v.size(); ++j) {
                CATCH_REQUIRE(v[j] == j);
            }
            v.clear();

            for (const auto& i : range) {
                v.push_back(i);
            }
            for (size_t j = 0; j < v.size(); ++j) {
                CATCH_REQUIRE(v[j] == j);
            }
            v.clear();

            auto range2 = svs::threads::UnitRange(0, 10);
            CATCH_REQUIRE(range == range2);
            CATCH_REQUIRE(!(range != range2));

            range2 = svs::threads::UnitRange(1, 11);
            CATCH_REQUIRE(range != range2);
            CATCH_REQUIRE(!(range == range2));
            CATCH_REQUIRE(range.size() == range2.size());

            // Type promotion.
            CATCH_REQUIRE(std::is_same_v<int, decltype(range)::value_type>);
            auto range3 = svs::threads::UnitRange<size_t>(5, 6);
            CATCH_REQUIRE(std::is_same_v<size_t, decltype(range3)::value_type>);

            auto range4 = svs::threads::UnitRange(uint8_t{2}, int16_t{0});
            CATCH_REQUIRE(std::is_same_v<
                          std::common_type_t<uint8_t, int16_t>,
                          decltype(range4)::value_type>);
            CATCH_REQUIRE(range4.size() == 0);
            CATCH_REQUIRE(range4.empty() == true);

            // Construction from iterator pair.
            auto pair = svs::threads::IteratorPair{
                svs::threads::IndexIterator(0),
                svs::threads::IndexIterator(100),
            };
            auto r = svs::threads::UnitRange{pair};
            CATCH_REQUIRE(std::is_same_v<decltype(r), svs::threads::UnitRange<int>>);
            CATCH_REQUIRE(r.front() == 0);
            CATCH_REQUIRE(*(r.begin()) == 0);
            CATCH_REQUIRE(r.back() == 99);
            CATCH_REQUIRE(*(r.end() - 1) == 99);
        }

        CATCH_SECTION("Printing") {
            auto range = svs::threads::UnitRange<size_t>(100, 200);
            auto repr = fmt::format("{}", range);
            CATCH_REQUIRE(repr == "UnitRange<uint64>(100, 200)");
        }

        CATCH_SECTION("Indexing") {
            auto range = svs::threads::UnitRange(100, 200);
            CATCH_REQUIRE(range.front() == 100);
            CATCH_REQUIRE(range.back() == 199);
            for (size_t i = 0; i < range.size(); ++i) {
                int ref = range.front() + i;
                CATCH_REQUIRE(range[i] == ref);
                CATCH_REQUIRE(range.at(i) == ref);
            }

            for (size_t i = 0; i < range.size(); ++i) {
                try {
                    range.at(range.size() + i);
                } catch (const std::out_of_range& err) { CATCH_REQUIRE(true); }
            }
        }

        CATCH_SECTION("Load balancing") {
            namespace threads = svs::threads;
            CATCH_SECTION("Balance 211") {
                size_t n = 4;
                uint32_t nthreads = 3;
                auto a = threads::balance(n, nthreads, 0);
                CATCH_REQUIRE(a == threads::UnitRange<size_t>(0, 2));
                CATCH_REQUIRE(a.size() == 2);

                auto b = threads::balance(n, nthreads, 1);
                CATCH_REQUIRE(b == threads::UnitRange<size_t>(2, 3));
                CATCH_REQUIRE(b.size() == 1);

                auto c = threads::balance(n, nthreads, 2);
                CATCH_REQUIRE(c == threads::UnitRange<size_t>(3, 4));
                CATCH_REQUIRE(c.size() == 1);
            }

            CATCH_SECTION("Over Subscribe") {
                size_t n = 4;
                size_t nthreads = 6;
                for (size_t i = 0; i < 4; ++i) {
                    auto a = threads::balance(n, nthreads, i);
                    CATCH_REQUIRE(a.size() == 1);
                    CATCH_REQUIRE(a == threads::UnitRange<size_t>(i, i + 1));
                }

                auto b = threads::balance(n, nthreads, 4);
                CATCH_REQUIRE(b.size() == 0);
                CATCH_REQUIRE(b.empty());

                auto c = threads::balance(n, nthreads, 5);
                CATCH_REQUIRE(c.size() == 0);
                CATCH_REQUIRE(c.empty());
            }

            CATCH_SECTION("Corner Cases") {
                auto a = threads::balance(0, 10, 5);
                CATCH_REQUIRE(a.size() == 0);
                CATCH_REQUIRE(a == threads::UnitRange<int>(0, 0));

                auto b = threads::balance(100, 1, 0);
                CATCH_REQUIRE(b.size() == 100);
                CATCH_REQUIRE(b == threads::UnitRange<int>(0, 100));
            }
        }
    }
}
