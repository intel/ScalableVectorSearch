/**
 *    Copyright (C) 2023, Intel Corporation
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
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_contains.hpp"
#include "catch2/matchers/catch_matchers_exception.hpp"

// stdlib
#include <cstdint>
#include <functional>
#include <type_traits>

namespace vamana = svs::index::vamana;

namespace {
template <typename I, typename Cmp>
void check_view(const svs::index::vamana::SearchBuffer<I, Cmp>& buffer) {
    auto sp = buffer.view();
    CATCH_REQUIRE(sp.size() == buffer.size());
    for (size_t i = 0, imax = sp.size(); i < imax; ++i) {
        CATCH_REQUIRE(sp[i] == buffer[i]);
    }
}

/////
///// Reference Implementation
/////

struct SearchBufferNeighbor {
    svs::Neighbor<uint32_t> neighbor_{};
    bool valid_ = true;
    bool visited_ = true;
};

namespace detail {
bool valid(svs::Neighbor<uint32_t, svs::Visited>) { return true; }
bool valid(const svs::PredicatedSearchNeighbor<uint32_t>& n) { return n.valid(); }

template <svs::NeighborLike As>
As convert_to(svs::lib::Type<As>, const SearchBufferNeighbor& x) {
    return As{x.neighbor_};
}

svs::PredicatedSearchNeighbor<uint32_t> convert_to(
    svs::lib::Type<svs::PredicatedSearchNeighbor<uint32_t>>, const SearchBufferNeighbor& x
) {
    return svs::PredicatedSearchNeighbor<uint32_t>{x.neighbor_, x.valid_};
}

} // namespace detail

template <typename Cmp = std::less<>> struct SearchBufferReference {
  public:
    SearchBufferReference(size_t roi_size, size_t valid_capacity)
        : neighbors_{}
        , roi_size_{roi_size}
        , valid_capacity_{valid_capacity} {}

    void insert(svs::Neighbor<uint32_t> neighbor, bool valid) {
        if (visited_.contains(neighbor.id())) {
            return;
        }
        visited_.insert(neighbor.id());

        // Find the first entry where the inserted neighbor is "further" than the already
        // stored neighbor.
        auto itr = neighbors_.begin();
        auto end = neighbors_.end();
        auto compare = Cmp{};
        while (itr != end) {
            if (compare(neighbor, itr->neighbor_)) {
                break;
            }
            ++itr;
        }
        neighbors_.insert(itr, SearchBufferNeighbor{neighbor, valid, false});
        shrink_to_fit();
    }

    size_t size() const { return neighbors_.size(); }
    size_t valid() const {
        size_t s = 0;
        for (const auto& n : neighbors_) {
            if (n.valid_) {
                ++s;
            }
        }
        return s;
    }

    void shrink_to_fit() {
        size_t num_valid = valid();
        while (size() > 0) {
            // Check to see if popping off the last element will drop us below the target
            // number of valid.
            bool is_valid = neighbors_.back().valid_;
            if (!is_valid) {
                neighbors_.pop_back();
            } else if (num_valid >= valid_capacity_ + 1) {
                neighbors_.pop_back();
                --num_valid;
            } else {
                break;
            }
        }
    }

    size_t best_unvisited() const {
        size_t s = 0;
        for (const auto& n : neighbors_) {
            if (!n.visited_) {
                return s;
            }
            ++s;
        }
        return s;
    }

    svs::Neighbor<uint32_t> next() {
        size_t i = best_unvisited();
        neighbors_.at(i).visited_ = true;
        return neighbors_.at(i).neighbor_;
    }

    bool done() {
        auto best = best_unvisited();
        // Count up the number of valid entries to the best unvisited.
        // If the number of valid entries is equal to the roi_size - we're done.
        size_t valid_count = 0;
        for (size_t i = 0; i < best; ++i) {
            if (neighbors_.at(i).valid_) {
                ++valid_count;
            }
        }
        return valid_count >= roi_size_;
    }

    template <typename Buffer> void check(const Buffer& buffer, bool last = false) const {
        // Gather the valid neighbors
        auto valid_in_buffer = std::vector<svs::Neighbor<uint32_t>>();
        for (size_t i = 0; i < buffer.size(); ++i) {
            if (detail::valid(buffer[i])) {
                valid_in_buffer.push_back(buffer[i]);
            }
        }

        auto valid_neighbors = std::vector<svs::Neighbor<uint32_t>>();
        for (const auto& n : neighbors_) {
            if (n.valid_) {
                valid_neighbors.push_back(n.neighbor_);
            }
        }
        auto eq = svs::NeighborEqual();
        CATCH_REQUIRE(valid_in_buffer.size() == valid());
        if (last) {
            CATCH_REQUIRE(valid_in_buffer.size() == valid_capacity_);
        }
        CATCH_REQUIRE(valid_neighbors.size() == valid_in_buffer.size());

        for (size_t i = 0; i < valid_in_buffer.size(); ++i) {
            CATCH_REQUIRE(eq(valid_in_buffer.at(i), valid_neighbors.at(i)));
        }
    }

    ///// Members
    std::unordered_set<uint32_t> visited_;
    std::vector<SearchBufferNeighbor> neighbors_;
    size_t roi_size_;
    size_t valid_capacity_;
};

template <typename Buffer, typename Cmp>
void fuzz_test_impl(
    Buffer& buffer,
    SearchBufferReference<Cmp>& reference,
    std::vector<SearchBufferNeighbor>& dataset,
    size_t batchsize,
    uint64_t seed
) {
    using T = typename Buffer::value_type;
    auto as = svs::lib::Type<T>();

    // Make sure we compare equal when there are no elements in either buffer.
    reference.check(buffer);

    auto rng = std::mt19937_64(seed);
    auto dist = std::uniform_int_distribution<size_t>(0, dataset.size() - 1);
    auto sample = [&]() { return dataset.at(dist(rng)); };

    // Keep trying until we get a valid entry.
    auto initial = sample();
    while (!initial.valid_) {
        initial = sample();
    }

    buffer.push_back(detail::convert_to(as, initial));
    reference.insert(initial.neighbor_, initial.valid_);
    reference.check(buffer);

    auto eq = svs::NeighborEqual();
    while (!reference.done()) {
        CATCH_REQUIRE(!buffer.done());
        CATCH_REQUIRE(eq(svs::Neighbor<uint32_t>{buffer.next()}, reference.next()));
        for (size_t i = 0; i < batchsize; ++i) {
            auto n = sample();
            buffer.insert(detail::convert_to(as, n));
            reference.insert(n.neighbor_, n.valid_);
        }
        reference.check(buffer);
    }
    reference.check(buffer, true);
    CATCH_REQUIRE(buffer.done());
}

struct FuzzSetup {
    size_t num_trials;
    size_t dataset_size;
    size_t roi_size;
    size_t valid_capacity;
    size_t seed;
    bool allow_invalid;
};

template <typename Buffer> void fuzz_test(Buffer& buffer, const FuzzSetup& setup) {
    auto rng = std::mt19937_64(setup.seed);
    auto dist = std::uniform_real_distribution<float>(-1000, 1000);
    using Cmp = typename Buffer::compare_type;

    auto generate_distance = [&]() { return dist(rng); };

    auto sz = setup.dataset_size;
    for (size_t i = 0; i < setup.num_trials; ++i) {
        // Create the initial dataset.
        auto dataset = std::vector<SearchBufferNeighbor>(sz);
        for (size_t j = 0; j < sz; ++j) {
            auto& entry = dataset.at(j);
            entry.neighbor_ = svs::Neighbor<uint32_t>(j, generate_distance());
            if (setup.allow_invalid && generate_distance() < 0) {
                entry.valid_ = false;
            } else {
                entry.valid_ = true;
            }
        }

        buffer.clear();
        auto reference = SearchBufferReference<Cmp>(setup.roi_size, setup.valid_capacity);
        auto batchsize = sz / 100;
        fuzz_test_impl(buffer, reference, dataset, batchsize, rng());
    }
}

template <typename Buffer> void test_visited_set_interface() {
    auto x = Buffer{10};
    CATCH_REQUIRE(x.visited_set_enabled() == false);
    // Marking items as visited should not have an effect.
    for (int i = 0; i < 10; ++i) {
        CATCH_REQUIRE(x.emplace_visited(i) == false);
    }
    for (int i = 0; i < 10; ++i) {
        CATCH_REQUIRE(x.is_visited(i) == false);
    }

    // Now, we enable the visited set.
    x.enable_visited_set();
    CATCH_REQUIRE(x.visited_set_enabled() == true);
    for (int i = 0; i < 10; ++i) {
        CATCH_REQUIRE(x.emplace_visited(i) == false);
    }
    for (int i = 0; i < 10; ++i) {
        CATCH_REQUIRE(x.is_visited(i) == true);
    }
    x.clear();
    for (int i = 0; i < 10; ++i) {
        CATCH_REQUIRE(x.is_visited(i) == false);
    }

    // Make sure we can go the other way and disable the visited set once it has been
    // enabled.
    x.disable_visited_set();
    CATCH_REQUIRE(x.visited_set_enabled() == false);
    for (int i = 0; i < 10; ++i) {
        CATCH_REQUIRE(!x.emplace_visited(i));
    }
    for (int i = 0; i < 10; ++i) {
        CATCH_REQUIRE(!x.is_visited(i));
    }
}

} // namespace

CATCH_TEST_CASE("Testing Search Buffer", "[core][search_buffer]") {
    CATCH_SECTION("SearchBufferConfig") {
        auto config = vamana::SearchBufferConfig();
        CATCH_REQUIRE(config.get_search_window_size() == 0);
        CATCH_REQUIRE(config.get_total_capacity() == 0);

        config = vamana::SearchBufferConfig{10};
        CATCH_REQUIRE(config.get_search_window_size() == 10);
        CATCH_REQUIRE(config.get_total_capacity() == 10);

        config = vamana::SearchBufferConfig{10, 20};
        CATCH_REQUIRE(config.get_search_window_size() == 10);
        CATCH_REQUIRE(config.get_total_capacity() == 20);

        // Ensure we get an error if mis-configuring.
        auto f = []() { return vamana::SearchBufferConfig{10, 9}; };
        CATCH_REQUIRE_THROWS_AS(f(), svs::ANNException);
    }

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
        test_visited_set_interface<svs::index::vamana::SearchBuffer<uint32_t>>();
        test_visited_set_interface<svs::index::vamana::MutableBuffer<uint32_t>>();
    }

    CATCH_SECTION("Changing Size") {
        auto x = svs::index::vamana::SearchBuffer<uint32_t>(3);
        CATCH_REQUIRE(x.size() == 0);
        x.insert({10, 20.0});
        CATCH_REQUIRE(x.size() == 1);
        x.insert({20, 5.0});
        CATCH_REQUIRE(x.size() == 2);
        x.insert({5, 10.0});
        CATCH_REQUIRE(x.size() == 3);
        x.change_maxsize(5);

        CATCH_REQUIRE(x.size() == 3);
        x.insert({3, 1.0});
        CATCH_REQUIRE(x.size() == 4);
        x.change_maxsize(2);
        CATCH_REQUIRE(x.size() == 2);
    }

    CATCH_SECTION("Shallow Copy") {
        auto x = svs::index::vamana::SearchBuffer<uint32_t>(10);
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

CATCH_TEST_CASE("Fuzzing", "[core][search_buffer]") {
    size_t num_trials = 5;
    size_t dataset_size = 1000;
    size_t seed = 0xc0ffee;
    bool allow_invalid = false;

    auto run_test = [&]<typename Cmp>(Cmp SVS_UNUSED(cmp)) {
        auto setup = FuzzSetup{num_trials, dataset_size, 32, 32, seed, allow_invalid};
        auto buffer = svs::index::vamana::SearchBuffer<uint32_t, Cmp>{
            svs::index::vamana::SearchBufferConfig{32, 32}};
        fuzz_test(buffer, setup);

        // Change size;
        setup.roi_size = 32;
        setup.valid_capacity = 64;
        buffer.change_maxsize(svs::index::vamana::SearchBufferConfig{32, 64});
        fuzz_test(buffer, setup);
    };

    CATCH_SECTION("Less") { run_test(std::less<>()); }

    CATCH_SECTION("Greater") { run_test(std::greater<>()); }
}

///
/// Mutable Buffer
///

CATCH_TEST_CASE("MutableBuffer", "[core][search_buffer]") {
    using buffer_type = svs::index::vamana::MutableBuffer<uint32_t>;
    svs::NeighborEqual eq{};

    auto buffer = buffer_type{4};
    // Test inserting all valid elements.
    CATCH_SECTION("All Valid") {
        CATCH_REQUIRE(buffer.size() == 0);
        buffer.insert({0, 100, true});
        CATCH_REQUIRE(buffer.size() == 1);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(eq(buffer[0], {0, 100, true}));

        buffer.insert({1, 50, true});
        CATCH_REQUIRE(buffer.size() == 2);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(eq(buffer[0], {1, 50, true}));
        CATCH_REQUIRE(eq(buffer[1], {0, 100, true}));

        buffer.insert({2, 150, true});
        CATCH_REQUIRE(buffer.size() == 3);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(eq(buffer[0], {1, 50, true}));
        CATCH_REQUIRE(eq(buffer[1], {0, 100, true}));
        CATCH_REQUIRE(eq(buffer[2], {2, 150, true}));

        buffer.insert({3, 40, true});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {3, 40, true}));
        CATCH_REQUIRE(eq(buffer[1], {1, 50, true}));
        CATCH_REQUIRE(eq(buffer[2], {0, 100, true}));
        CATCH_REQUIRE(eq(buffer[3], {2, 150, true}));

        // Now that the search buffer is full, adding a new larger element to the end
        // will have no effect.
        buffer.insert({4, 1000, true});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {3, 40, true}));
        CATCH_REQUIRE(eq(buffer[1], {1, 50, true}));
        CATCH_REQUIRE(eq(buffer[2], {0, 100, true}));
        CATCH_REQUIRE(eq(buffer[3], {2, 150, true}));

        // Adding a smaller element to the front will shift everything back.
        buffer.insert({5, 0, true});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {5, 0, true}));
        CATCH_REQUIRE(eq(buffer[1], {3, 40, true}));
        CATCH_REQUIRE(eq(buffer[2], {1, 50, true}));
        CATCH_REQUIRE(eq(buffer[3], {0, 100, true}));

        // Now, if we add an invalid element to the front, the buffer size should grow in
        // order to maintain the correct number of valid elements.
        buffer.insert({6, 1, false});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {5, 0, true}));
        CATCH_REQUIRE(eq(buffer[1], {6, 1, false}));
        CATCH_REQUIRE(eq(buffer[2], {3, 40, true}));
        CATCH_REQUIRE(eq(buffer[3], {1, 50, true}));
        CATCH_REQUIRE(eq(buffer[4], {0, 100, true}));

        // Appending an invalid element at the end should still get dropped.
        buffer.insert({7, 2000, false});
        CATCH_REQUIRE(buffer.size() == 5);
        CATCH_REQUIRE(buffer.full() == true);
        CATCH_REQUIRE(eq(buffer[0], {5, 0, true}));
        CATCH_REQUIRE(eq(buffer[1], {6, 1, false}));
        CATCH_REQUIRE(eq(buffer[2], {3, 40, true}));
        CATCH_REQUIRE(eq(buffer[3], {1, 50, true}));
        CATCH_REQUIRE(eq(buffer[4], {0, 100, true}));
    }

    // One behavior of the MutableBuffer is that it will continue to acrue candidates until
    // the target number of valid candidates is achieved.
    //
    // If these valid candidates are all very near the query, the queued invalid
    // elements should then be dropped.
    CATCH_SECTION("Collapsing") {
        for (uint32_t i = 0; i < 100; ++i) {
            buffer.insert({i, svs::lib::narrow_cast<float>(1000 - i), false});
        }
        CATCH_REQUIRE(buffer.size() == 100);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(buffer.valid() == 0);

        buffer.insert({100, 10, true});
        CATCH_REQUIRE(buffer.size() == 101);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(buffer.valid() == 1);

        buffer.insert({101, 8, true});
        CATCH_REQUIRE(buffer.size() == 102);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(buffer.valid() == 2);

        buffer.insert({102, 6, true});
        CATCH_REQUIRE(buffer.size() == 103);
        CATCH_REQUIRE(buffer.full() == false);
        CATCH_REQUIRE(buffer.valid() == 3);

        buffer.insert({103, 4, true});
        CATCH_REQUIRE(buffer.size() == 4);
        CATCH_REQUIRE(buffer.full() == true);
    }

    // Push-back initialization.
    CATCH_SECTION("Push-back initialization") {
        auto b = buffer_type{{2, 4}};
        using T = svs::PredicatedSearchNeighbor<uint32_t>;
        auto make_visited = [](uint32_t id, float distance, bool valid) {
            auto n = T{id, distance, valid};
            n.set_visited();
            return n;
        };

        CATCH_SECTION("Full Buffer") {
            // We should be able to add elements to the buffer.
            // Valid elements should only be appended until 4 have been added.
            CATCH_REQUIRE(b.target() == 4);
            CATCH_REQUIRE(b.size() == 0);
            CATCH_REQUIRE(b.valid() == 0);
            CATCH_REQUIRE(!b.full());

            b.push_back({1, 10.0, true});
            CATCH_REQUIRE(b.size() == 1);
            CATCH_REQUIRE(b.valid() == 1);
            CATCH_REQUIRE(!b.full());

            b.push_back({2, 9.0, false});
            CATCH_REQUIRE(b.size() == 2);
            CATCH_REQUIRE(b.valid() == 1);
            CATCH_REQUIRE(!b.full());

            b.push_back({3, 8.0, true});
            CATCH_REQUIRE(b.size() == 3);
            CATCH_REQUIRE(b.valid() == 2);
            CATCH_REQUIRE(!b.full());

            b.push_back({4, 7.0, true});
            CATCH_REQUIRE(b.size() == 4);
            CATCH_REQUIRE(b.valid() == 3);
            CATCH_REQUIRE(!b.full());

            b.push_back({5, 6.0, false});
            CATCH_REQUIRE(b.size() == 5);
            CATCH_REQUIRE(b.valid() == 3);
            CATCH_REQUIRE(!b.full());

            b.push_back({6, 5.0, false});
            CATCH_REQUIRE(b.size() == 6);
            CATCH_REQUIRE(b.valid() == 3);
            CATCH_REQUIRE(!b.full());

            b.push_back({7, 4.0, true});
            CATCH_REQUIRE(b.size() == 7);
            CATCH_REQUIRE(b.valid() == 4);
            CATCH_REQUIRE(b.full());

            // Appending another valid item should have no effect.
            b.push_back({8, 3.0, true});
            CATCH_REQUIRE(b.size() == 7);
            CATCH_REQUIRE(b.valid() == 4);
            CATCH_REQUIRE(b.full());

            // Appending an invalid item should still grow the buffer.
            b.push_back({8, 2.0, false});
            CATCH_REQUIRE(b.size() == 8);
            CATCH_REQUIRE(b.valid() == 4);
            CATCH_REQUIRE(b.full());

            // Append a few more items that will fall off the end after sorting.
            b.push_back({9, 100.0, false});
            CATCH_REQUIRE(b.size() == 9);
            CATCH_REQUIRE(b.valid() == 4);

            b.push_back({10, 110.0, false});
            CATCH_REQUIRE(b.size() == 10);
            CATCH_REQUIRE(b.valid() == 4);

            // Now - invoke `sort()` to restore data structure invariants.
            //
            // The higher elements we appended should be implicitly dropped since the buffer
            // is not in a full state.
            b.sort();
            CATCH_REQUIRE(b.size() == 8);
            CATCH_REQUIRE(b.valid() == 4);
            CATCH_REQUIRE(b.back().valid());

            // Ensure the contents of the buffer are as expected.
            CATCH_REQUIRE(eq(b[0], {8, 2.0, false}));
            CATCH_REQUIRE(eq(b[1], {7, 4.0, true}));
            CATCH_REQUIRE(eq(b[2], {6, 5.0, false}));
            CATCH_REQUIRE(eq(b[3], {5, 6.0, false}));
            CATCH_REQUIRE(eq(b[4], {4, 7.0, true}));
            CATCH_REQUIRE(eq(b[5], {3, 8.0, true}));
            CATCH_REQUIRE(eq(b[6], {2, 9.0, false}));
            CATCH_REQUIRE(eq(b[7], {1, 10.0, true}));

            // Ensure that the ROI is configured properly.
            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(eq(b.next(), make_visited(8, 2.0, false)));

            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(eq(b.next(), make_visited(7, 4.0, true)));

            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(eq(b.next(), make_visited(6, 5.0, false)));

            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(eq(b.next(), make_visited(5, 6.0, false)));

            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(eq(b.next(), make_visited(4, 7.0, true)));

            CATCH_REQUIRE(b.done());
        }

        CATCH_SECTION("Partially Full Buffer") {
            // Here, we target a buffer that crosses the target_valid threshold, but not yet
            // the valid_capacity threshold.
            b.push_back({1, 10.0, true});
            CATCH_REQUIRE(b.size() == 1);
            CATCH_REQUIRE(b.valid() == 1);
            CATCH_REQUIRE(!b.full());

            b.push_back({2, 9.0, false});
            CATCH_REQUIRE(b.size() == 2);
            CATCH_REQUIRE(b.valid() == 1);
            CATCH_REQUIRE(!b.full());

            b.push_back({3, 8.0, true});
            CATCH_REQUIRE(b.size() == 3);
            CATCH_REQUIRE(b.valid() == 2);
            CATCH_REQUIRE(!b.full());

            b.push_back({4, 7.0, true});
            CATCH_REQUIRE(b.size() == 4);
            CATCH_REQUIRE(b.valid() == 3);
            CATCH_REQUIRE(!b.full());

            b.push_back({5, 20.0, false});
            CATCH_REQUIRE(b.size() == 5);
            CATCH_REQUIRE(b.valid() == 3);
            CATCH_REQUIRE(!b.full());

            // Invariant 6 hasn't kicked in, so we aren't guarenteed a valid last element.
            b.sort();
            CATCH_REQUIRE(b.size() == 5);
            CATCH_REQUIRE(b.valid() == 3);
            CATCH_REQUIRE(!b.back().valid());

            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(b.next() == make_visited(4, 7.0, true));
            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(b.next() == make_visited(3, 8.0, true));
            CATCH_REQUIRE(b.done());
        }

        CATCH_SECTION("Non full buffer") {
            b.push_back({1, 10.0, true});
            CATCH_REQUIRE(b.size() == 1);
            CATCH_REQUIRE(b.valid() == 1);
            CATCH_REQUIRE(!b.full());

            b.push_back({2, 9.0, false});
            CATCH_REQUIRE(b.size() == 2);
            CATCH_REQUIRE(b.valid() == 1);
            CATCH_REQUIRE(!b.full());

            b.push_back({3, 8.0, false});
            CATCH_REQUIRE(b.size() == 3);
            CATCH_REQUIRE(b.valid() == 1);
            CATCH_REQUIRE(!b.full());

            b.sort();
            CATCH_REQUIRE(b.size() == 3);
            CATCH_REQUIRE(b.valid() == 1);
            CATCH_REQUIRE(b.back().valid());

            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(b.next() == make_visited(3, 8.0, false));
            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(b.next() == make_visited(2, 9.0, false));
            CATCH_REQUIRE(!b.done());
            CATCH_REQUIRE(b.next() == make_visited(1, 10.0, true));
            CATCH_REQUIRE(b.done());
        }
    }
}

CATCH_TEST_CASE("Fuzzing Mutable", "[core][search_buffer]") {
    size_t num_trials = 5;
    size_t dataset_size = 1000;
    size_t seed = 0xc0ffee;
    bool allow_invalid = true;

    auto run_test = [&]<typename Cmp>(Cmp SVS_UNUSED(cmp)) {
        auto setup = FuzzSetup{num_trials, dataset_size, 32, 32, seed, allow_invalid};
        auto buffer = svs::index::vamana::MutableBuffer<uint32_t, Cmp>{
            svs::index::vamana::SearchBufferConfig{32, 32}};
        fuzz_test(buffer, setup);

        // Change size;
        setup.roi_size = 32;
        setup.valid_capacity = 64;
        buffer.change_maxsize(svs::index::vamana::SearchBufferConfig{32, 64});
        fuzz_test(buffer, setup);
    };

    CATCH_SECTION("Less") { run_test(std::less<>()); }
    CATCH_SECTION("Greater") { run_test(std::greater<>()); }
}
