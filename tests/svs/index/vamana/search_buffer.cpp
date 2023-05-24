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

// svs
#include "svs/index/vamana/search_buffer.h"
#include "svs/index/vamana/dynamic_search_buffer.h"

// tests
#include "tests/utils/generators.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <cstdint>
#include <functional>
#include <type_traits>

namespace {
template <typename I, typename Cmp>
void check_view(const svs::index::vamana::SearchBuffer<I, Cmp>& buffer) {
    auto sp = buffer.view();
    CATCH_REQUIRE(sp.size() == buffer.size());
    for (size_t i = 0, imax = sp.size(); i < imax; ++i) {
        CATCH_REQUIRE(sp[i] == buffer[i]);
    }
}
} // namespace

CATCH_TEST_CASE("Testing Search Buffer", "[core][search_buffer]") {
    // Type Traits
    CATCH_REQUIRE(
        std::is_move_constructible_v<svs::index::vamana::SearchBuffer<uint32_t>> == true
    );
    CATCH_REQUIRE(
        std::is_move_assignable_v<svs::index::vamana::SearchBuffer<uint32_t>> == true
    );

    constexpr auto buffersize = 5;
    svs::NeighborEqual eq{};

    svs::index::vamana::SearchBuffer<uint32_t> buffer{buffersize};
    CATCH_SECTION("Can't push off the edge") {
        buffer.push_back({1, 1});
        buffer.push_back({2, 2});
        buffer.push_back({3, 3});
        buffer.push_back({4, 4});
        buffer.push_back({5, 5});
        CATCH_REQUIRE(buffer.size() == 5);
        buffer.push_back({6, 6});
        CATCH_REQUIRE(buffer.size() == 5);
        check_view(buffer);
    }

    CATCH_SECTION("Basic Behavior") {
        CATCH_REQUIRE(buffer.size() == 0);
        CATCH_REQUIRE(buffer.capacity() == buffersize);
        check_view(buffer);
        for (auto i = 0; i < buffersize; ++i) {
            // We don't know the contents of the node, but we know that it should
            // at least not be visited.
            const auto node = buffer[i];
            CATCH_REQUIRE(node.visited() == false);
        }

        buffer.push_back({1, 2});
        CATCH_REQUIRE(buffer.size() == 1);
        auto& push_back_node = buffer[0];
        CATCH_REQUIRE(push_back_node.id_ == 1);
        CATCH_REQUIRE(push_back_node.distance() == 2);
        CATCH_REQUIRE(push_back_node.visited() == false);
        push_back_node.set_visited();
        const auto& push_back_node_const = buffer[0];
        CATCH_REQUIRE(push_back_node_const.visited() == true);
        check_view(buffer);

        buffer.clear();
        CATCH_REQUIRE(buffer.size() == 0);
    }

    CATCH_SECTION("Insert") {
        // We need to explore the following cases:
        // 1a. Insert at the end of a non-full buffer.
        // 1b. Insert at the end of a non-full buffer with duplicate id.
        // 2a. Insert at the beginning of a non-full buffer.
        // 2b. Insert at the beginning of a non-full buffer with duplicate id.
        // 3a. Insert in the middle of a non-full buffer.
        // 3b. Insert in the middle of a non-full buffer with duplicate id.
        //
        // 4. Insert at the end of a full buffer.
        // 5a. Insert at the beginning of a full buffer.
        // 5b. Insert at the beginning of a full buffer with duplicate id.
        // 6a. Insert at in the middle of a full buffer.
        // 6b. Insert at in the middle of a full buffer with duplicate id.
        CATCH_REQUIRE(buffer.size() == 0);

        // Initialize
        buffer.push_back({1, 10});
        CATCH_REQUIRE(eq(buffer[0], {1, 10}));

        // Case 1a
        auto idx = buffer.insert({2, 20});
        CATCH_REQUIRE(buffer.size() == 2);
        CATCH_REQUIRE(idx == 1);
        CATCH_REQUIRE(eq(buffer[0], {1, 10}));
        CATCH_REQUIRE(eq(buffer[1], {2, 20}));
        buffer[1].set_visited();
        CATCH_REQUIRE(buffer[1].visited());

        // Case 1b
        idx = buffer.insert({2, 20});
        CATCH_REQUIRE(buffer.size() == 2);
        CATCH_REQUIRE(idx == (buffer.size() + 1));
        CATCH_REQUIRE(eq(buffer[0], {1, 10}));
        CATCH_REQUIRE(eq(buffer[1], {2, 20, true}));
        CATCH_REQUIRE(!eq(buffer[2], {2, 20}));

        // Case 2a
        idx = buffer.insert({3, 5});
        CATCH_REQUIRE(buffer.size() == 3);
        CATCH_REQUIRE(idx == 0);
        CATCH_REQUIRE(eq(buffer[0], {3, 5}));
        CATCH_REQUIRE(eq(buffer[1], {1, 10}));
        CATCH_REQUIRE(eq(buffer[2], {2, 20, true}));
        CATCH_REQUIRE(buffer[2].visited());

        // Case 2b
        idx = buffer.insert({3, 5});
        CATCH_REQUIRE(buffer.size() == 3);
        CATCH_REQUIRE(idx == buffer.size() + 1);
        CATCH_REQUIRE(eq(buffer[0], {3, 5}));
        CATCH_REQUIRE(eq(buffer[1], {1, 10}));
        CATCH_REQUIRE(eq(buffer[2], {2, 20, true}));

        // Case 3a
        idx = buffer.insert({4, 15});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(idx == 2);
        CATCH_REQUIRE(eq(buffer[0], {3, 5}));
        CATCH_REQUIRE(eq(buffer[1], {1, 10}));
        CATCH_REQUIRE(eq(buffer[2], {4, 15}));
        CATCH_REQUIRE(eq(buffer[3], {2, 20, true}));
        CATCH_REQUIRE(buffer[3].visited());

        // Case 3b
        idx = buffer.insert({4, 15});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(idx == buffer.size() + 1);
        CATCH_REQUIRE(eq(buffer[0], {3, 5}) == true);
        CATCH_REQUIRE(eq(buffer[1], {1, 10}) == true);
        CATCH_REQUIRE(eq(buffer[2], {4, 15}) == true);
        CATCH_REQUIRE(eq(buffer[3], {2, 20, true}) == true);

        // Prep for case 4
        idx = buffer.insert({5, 30});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(idx == 4);
        CATCH_REQUIRE(eq(buffer[0], {3, 5}));
        CATCH_REQUIRE(eq(buffer[1], {1, 10}));
        CATCH_REQUIRE(eq(buffer[2], {4, 15}));
        CATCH_REQUIRE(eq(buffer[3], {2, 20, true}));
        CATCH_REQUIRE(eq(buffer[4], {5, 30}));

        // Case 4
        idx = buffer.insert({6, 1000});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(idx == 5);
        CATCH_REQUIRE(eq(buffer[0], {3, 5}));
        CATCH_REQUIRE(eq(buffer[1], {1, 10}));
        CATCH_REQUIRE(eq(buffer[2], {4, 15}));
        CATCH_REQUIRE(eq(buffer[3], {2, 20, true}));
        CATCH_REQUIRE(eq(buffer[4], {5, 30}));

        // Case 5a
        idx = buffer.insert({7, 1});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(idx == 0);
        CATCH_REQUIRE(eq(buffer[0], {7, 1}));
        CATCH_REQUIRE(eq(buffer[1], {3, 5}));
        CATCH_REQUIRE(eq(buffer[2], {1, 10}));
        CATCH_REQUIRE(eq(buffer[3], {4, 15}));
        CATCH_REQUIRE(eq(buffer[4], {2, 20, true}));

        // Case 5b
        idx = buffer.insert({7, 1});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(idx == buffer.size() + 1);
        CATCH_REQUIRE(eq(buffer[0], {7, 1}));
        CATCH_REQUIRE(eq(buffer[1], {3, 5}));
        CATCH_REQUIRE(eq(buffer[2], {1, 10}));
        CATCH_REQUIRE(eq(buffer[3], {4, 15}));
        CATCH_REQUIRE(eq(buffer[4], {2, 20, true}));

        // Case 6a
        idx = buffer.insert({8, 8});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(idx == 2);
        CATCH_REQUIRE(eq(buffer[0], {7, 1}));
        CATCH_REQUIRE(eq(buffer[1], {3, 5}));
        CATCH_REQUIRE(eq(buffer[2], {8, 8}));
        CATCH_REQUIRE(eq(buffer[3], {1, 10}));
        CATCH_REQUIRE(eq(buffer[4], {4, 15}));

        // Case 6b
        idx = buffer.insert({8, 8});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(idx == buffer.size() + 1);
        CATCH_REQUIRE(eq(buffer[0], {7, 1}));
        CATCH_REQUIRE(eq(buffer[1], {3, 5}));
        CATCH_REQUIRE(eq(buffer[2], {8, 8}));
        CATCH_REQUIRE(eq(buffer[3], {1, 10}));
        CATCH_REQUIRE(eq(buffer[4], {4, 15}));

        // Wrap up
        check_view(buffer);
        buffer.clear();
        CATCH_REQUIRE(buffer.size() == 0);
    }

    CATCH_SECTION("Sorting") {
        buffer.push_back({1, 100});
        buffer.push_back({2, 10});
        buffer.push_back({3, 50});
        CATCH_REQUIRE(buffer.size() == 3);
        buffer.sort();
        CATCH_REQUIRE(eq(buffer[0], {2, 10}));
        CATCH_REQUIRE(eq(buffer[1], {3, 50}));
        CATCH_REQUIRE(eq(buffer[2], {1, 100}));

        // Also try with reverse ordering.
        auto buffer2 = svs::index::vamana::SearchBuffer<uint32_t, std::greater<>>(5);
        buffer2.push_back({1, 100});
        buffer2.push_back({2, 10});
        buffer2.push_back({3, 50});
        CATCH_REQUIRE(buffer2.size() == 3);
        buffer2.sort();
        CATCH_REQUIRE(eq(buffer2[0], {1, 100}));
        CATCH_REQUIRE(eq(buffer2[1], {3, 50}));
        CATCH_REQUIRE(eq(buffer2[2], {2, 10}));
    }

    CATCH_SECTION("Visited Set") {
        auto x = svs::index::vamana::SearchBuffer<size_t>(10);
        CATCH_REQUIRE(x.visited_set_enabled() == false);
        // Marking items as visited should not have an effect.
        for (int i = 0; i < 10; ++i) {
            x.set_visited(i);
        }
        for (int i = 0; i < 10; ++i) {
            CATCH_REQUIRE(x.visited(i) == false);
        }

        // Now, we enable the visited set.
        x.enable_visited_set();
        CATCH_REQUIRE(x.visited_set_enabled() == true);
        for (int i = 0; i < 10; ++i) {
            x.set_visited(i);
        }
        for (int i = 0; i < 10; ++i) {
            CATCH_REQUIRE(x.visited(i) == true);
        }
        x.clear();
        for (int i = 0; i < 10; ++i) {
            CATCH_REQUIRE(x.visited(i) == false);
        }

        // Make sure we can go the other way and disable the visited set once it has been
        // enabled.
        x.disable_visited_set();
        CATCH_REQUIRE(x.visited_set_enabled() == false);
        for (int i = 0; i < 10; ++i) {
            x.set_visited(i);
        }
        for (int i = 0; i < 10; ++i) {
            CATCH_REQUIRE(x.visited(i) == false);
        }
    }

    CATCH_SECTION("Shallow Copy") {
        auto x = svs::index::vamana::SearchBuffer<size_t>(10);
        CATCH_REQUIRE(svs::threads::shallow_copyable_v<decltype(x)>);

        // Shallow copy without the visited set enabled.
        auto y = svs::threads::shallow_copy(x);
        CATCH_REQUIRE(x.visited_set_enabled() == false);
        CATCH_REQUIRE(y.visited_set_enabled() == false);
        CATCH_REQUIRE(x.capacity() == 10);
        CATCH_REQUIRE(y.capacity() == 10);

        // Shallow copy with the visited set enabled.
        x.change_maxsize(20);
        x.enable_visited_set();
        auto z = svs::threads::shallow_copy(x);
        CATCH_REQUIRE(x.visited_set_enabled() == true);
        CATCH_REQUIRE(y.visited_set_enabled() == false);
        CATCH_REQUIRE(z.visited_set_enabled() == true);
        CATCH_REQUIRE(x.capacity() == 20);
        CATCH_REQUIRE(y.capacity() == 10);
        CATCH_REQUIRE(z.capacity() == 20);
    }
}

template <typename Cmp> void fuzz_test(Cmp cmp) {
    constexpr size_t buffersize = 100;
    constexpr size_t test_length = 1000;
    svs::index::vamana::SearchBuffer<uint32_t, Cmp> buffer{buffersize};
    std::vector<svs::SearchNeighbor<uint32_t>> reference{};

    auto generator = svs_test::make_generator<float>(-100, 100);

    buffer.push_back({0, 0});
    reference.push_back({0, 0});
    for (uint32_t i = 1; i <= test_length; ++i) {
        const float distance = svs_test::generate(generator);
        svs::SearchNeighbor<uint32_t> neighbor{i, distance};

        // Purposely insert twice to make sure the duplicate ID detection at least kind of
        // works.
        buffer.insert({i, distance});
        buffer.insert({i, distance});

        // Insert into the appropriate position in the reference vector and resize.
        reference.insert(
            std::find_if(
                reference.begin(),
                reference.end(),
                [&neighbor, cmp](const auto& other) {
                    return cmp(neighbor.distance(), other.distance());
                }
            ),
            neighbor
        );
        reference.resize(std::min(reference.size(), buffer.capacity()));
    }
    CATCH_REQUIRE(reference.size() == buffer.size());
    const bool equal = std::equal(
        reference.begin(), reference.end(), buffer.begin(), svs::NeighborEqual()
    );
    check_view(buffer);
    CATCH_REQUIRE(equal == true);
}

CATCH_TEST_CASE("Fuzzing", "[core][search_buffer]") {
    fuzz_test(std::less<>{});
    fuzz_test(std::greater<>{});
}

///
/// Mutable Buffer
///

CATCH_TEST_CASE("MutableBuffer", "[core][search_buffer]") {
    using buffer_type = svs::index::vamana::MutableBuffer<size_t>;
    svs::NeighborEqual eq{};

    auto buffer = buffer_type{4};
    // Test inserting all valid elements.
    CATCH_SECTION("All Valid") {
        CATCH_REQUIRE(buffer.size() == 0);
        buffer.insert({0, 100, false});
        CATCH_REQUIRE(buffer.size() == 1);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(eq(buffer[0], {0, 100, false}));

        buffer.insert({1, 50, false});
        CATCH_REQUIRE(buffer.size() == 2);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(eq(buffer[0], {1, 50, false}));
        CATCH_REQUIRE(eq(buffer[1], {0, 100, false}));

        buffer.insert({2, 150, false});
        CATCH_REQUIRE(buffer.size() == 3);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(eq(buffer[0], {1, 50, false}));
        CATCH_REQUIRE(eq(buffer[1], {0, 100, false}));
        CATCH_REQUIRE(eq(buffer[2], {2, 150, false}));

        buffer.insert({3, 40, false});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {3, 40, false}));
        CATCH_REQUIRE(eq(buffer[1], {1, 50, false}));
        CATCH_REQUIRE(eq(buffer[2], {0, 100, false}));
        CATCH_REQUIRE(eq(buffer[3], {2, 150, false}));

        // Now that the search buffer is full, adding a new larger element to the end
        // will have no effect.
        buffer.insert({4, 1000, false});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {3, 40, false}));
        CATCH_REQUIRE(eq(buffer[1], {1, 50, false}));
        CATCH_REQUIRE(eq(buffer[2], {0, 100, false}));
        CATCH_REQUIRE(eq(buffer[3], {2, 150, false}));

        // Adding a smaller element to the front will shift everything back.
        buffer.insert({5, 0, false});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {5, 0, false}));
        CATCH_REQUIRE(eq(buffer[1], {3, 40, false}));
        CATCH_REQUIRE(eq(buffer[2], {1, 50, false}));
        CATCH_REQUIRE(eq(buffer[3], {0, 100, false}));

        // Now, if we add a skipped element to the front, the buffer size should grow in
        // order to maintain the correct number of valid elements.
        buffer.insert({6, 1, true});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {5, 0, false}));
        CATCH_REQUIRE(eq(buffer[1], {6, 1, true}));
        CATCH_REQUIRE(eq(buffer[2], {3, 40, false}));
        CATCH_REQUIRE(eq(buffer[3], {1, 50, false}));
        CATCH_REQUIRE(eq(buffer[4], {0, 100, false}));

        // Appending a skipped element at the end should still get dropped.
        buffer.insert({7, 2000, true});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {5, 0, false}));
        CATCH_REQUIRE(eq(buffer[1], {6, 1, true}));
        CATCH_REQUIRE(eq(buffer[2], {3, 40, false}));
        CATCH_REQUIRE(eq(buffer[3], {1, 50, false}));
        CATCH_REQUIRE(eq(buffer[4], {0, 100, false}));
    }

    // One behavior of the MutableBuffer is that it will continue to acrue candidates until
    // the target number of valid candidates is achieved.
    //
    // If these valid candidates are all very near the query, the queued invalid
    // elements should then be dropped.
    CATCH_SECTION("Collapsing") {
        for (size_t i = 0; i < 100; ++i) {
            buffer.insert({i, svs::lib::narrow_cast<float>(1000 - i), true});
        }
        CATCH_REQUIRE(buffer.size() == 100);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(buffer.valid() == 0);

        buffer.insert({100, 10, false});
        CATCH_REQUIRE(buffer.size() == 101);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(buffer.valid() == 1);

        buffer.insert({101, 8, false});
        CATCH_REQUIRE(buffer.size() == 102);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(buffer.valid() == 2);

        buffer.insert({102, 6, false});
        CATCH_REQUIRE(buffer.size() == 103);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(buffer.valid() == 3);

        buffer.insert({103, 4, false});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(buffer.full() == true);
    }
}

namespace {
template <typename Cmp> void fuzz_mutable(size_t ntests) {
    Cmp cmp{};
    size_t target = 100;
    auto buffer = svs::index::vamana::MutableBuffer<size_t, Cmp>(target);
    std::vector<svs::SkippableSearchNeighbor<size_t>> reference{};

    auto generator = svs_test::make_generator<float>(-1000, 1000);
    auto valid_generator = svs_test::make_generator<size_t>(0, 1);

    buffer.push_back({0, 0, false});
    reference.push_back({0, 0, false});
    for (size_t i = 1; i <= ntests; ++i) {
        float distance = svs_test::generate(generator);
        bool valid = svs_test::generate(valid_generator) == 1;

        svs::SkippableSearchNeighbor<size_t> neighbor{i, distance, !valid};

        // Purposely insert twice to make sure the duplicate ID detection at least kind of
        // works.
        buffer.insert({i, distance, !valid});
        buffer.insert({i, distance, !valid});

        if (valid) {
            reference.insert(
                std::find_if(
                    reference.begin(),
                    reference.end(),
                    [&neighbor, cmp](const auto& other) {
                        return cmp(neighbor.distance(), other.distance());
                    }
                ),
                neighbor
            );
            reference.resize(std::min(reference.size(), target));
        }
    }

    CATCH_REQUIRE(buffer.full());
    CATCH_REQUIRE(buffer.size() > target);
    buffer.cleanup();
    CATCH_REQUIRE(buffer.full());
    CATCH_REQUIRE(buffer.size() == target);

    bool passed = std::equal(
        reference.begin(), reference.end(), buffer.begin(), svs::NeighborEqual()
    );
    CATCH_REQUIRE(passed);
}

} // namespace

CATCH_TEST_CASE("Fuzzing Mutable", "[core][search_buffer]") {
    fuzz_mutable<std::less<>>(1'000);
    fuzz_mutable<std::greater<>>(2'000);
}
