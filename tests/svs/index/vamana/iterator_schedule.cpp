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
#include "svs/index/vamana/iterator_schedule.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

// tests
#include "tests/utils/utils.h"

namespace {

template <typename T>
concept IteratorSchedule = svs::index::vamana::IteratorSchedule<T>;

// Functor computing `m * x + b` for unsigned `m, x, b`.
struct Linear {
    size_t operator()(size_t x) const { return m * x + b; }

    size_t m;
    size_t b;
};

template <IteratorSchedule Schedule>
void test_default_schedule(
    const Schedule& schedule,
    size_t num_iterations,
    const Linear& search_window_size,
    const Linear& search_buffer_capacity,
    size_t batch_size,
    size_t prefetch_lookahead,
    size_t prefetch_step
) {
    for (size_t i = 0; i < num_iterations; ++i) {
        auto sp = schedule.for_iteration(i);
        CATCH_REQUIRE(sp.buffer_config_.get_search_window_size() == search_window_size(i));
        CATCH_REQUIRE(sp.buffer_config_.get_total_capacity() == search_buffer_capacity(i));
        CATCH_REQUIRE(sp.search_buffer_visited_set_ == false);
        CATCH_REQUIRE(sp.prefetch_lookahead_ == prefetch_lookahead);
        CATCH_REQUIRE(sp.prefetch_step_ == prefetch_step);

        CATCH_REQUIRE(schedule.max_candidates(i) == batch_size);
    }
}

template <IteratorSchedule Schedule>
void test_linear_schedule(
    const Schedule& schedule,
    size_t num_iterations,
    const Linear& search_window_size,
    const Linear& search_buffer_capacity,
    const Linear& batch_size,
    int64_t visited_after,
    size_t prefetch_lookahead,
    size_t prefetch_step
) {
    for (size_t i = 0; i < num_iterations; ++i) {
        // Search Parameters
        auto sp = schedule.for_iteration(i);
        CATCH_REQUIRE(sp.buffer_config_.get_search_window_size() == search_window_size(i));
        CATCH_REQUIRE(sp.buffer_config_.get_total_capacity() == search_buffer_capacity(i));
        if (visited_after >= 0 && i >= svs::lib::narrow<size_t>(visited_after)) {
            CATCH_REQUIRE(sp.search_buffer_visited_set_ == true);
        } else {
            CATCH_REQUIRE(sp.search_buffer_visited_set_ == false);
        }

        CATCH_REQUIRE(sp.prefetch_lookahead_ == prefetch_lookahead);
        CATCH_REQUIRE(sp.prefetch_step_ == prefetch_step);

        // Max Candidates
        CATCH_REQUIRE(schedule.max_candidates(i) == batch_size(i));
    }
}

} // namespace

CATCH_TEST_CASE("Iterator Schedules", "[vamana][index][iterator][iterator_schedule]") {
    using VSP = svs::index::vamana::VamanaSearchParameters;
    using LS = svs::index::vamana::LinearSchedule;
    CATCH_SECTION("Default Schedule") {
        auto base = VSP{{10, 20}, false, 1, 4};
        auto sched = svs::index::vamana::DefaultSchedule{base, 5};

        // `for_iteration` interface.
        CATCH_REQUIRE(sched.for_iteration(0) == base);
        CATCH_REQUIRE(sched.for_iteration(1) == VSP{{15, 25}, false, 1, 4});
        CATCH_REQUIRE(sched.for_iteration(2) == VSP{{20, 30}, false, 1, 4});
        CATCH_REQUIRE(sched.for_iteration(3) == VSP{{25, 35}, false, 1, 4});

        // `max_candidates` Interface.
        CATCH_REQUIRE(sched.max_candidates(0) == 5);
        CATCH_REQUIRE(sched.max_candidates(1) == 5);
        CATCH_REQUIRE(sched.max_candidates(2) == 5);
        CATCH_REQUIRE(sched.max_candidates(3) == 5);

        test_default_schedule(sched, 4, {5, 10}, {5, 20}, 5, 1, 4);
    }

    CATCH_SECTION("Linear Schedule") {
        auto base = VSP({10, 23}, false, 4, 0);
        CATCH_SECTION("Invariants") {
            // Buffer capacity scale must be greater than or equal to the capacity.
            CATCH_REQUIRE_THROWS_MATCHES(
                (LS{base, 20, 10, -1, 10, 0}),
                svs::ANNException,
                svs_test::ExceptionMatcher(
                    Catch::Matchers::ContainsSubstring("Capacity scaling must be at least")
                )
            );

            // Initial batch size must be non-zero.
            CATCH_REQUIRE_THROWS_MATCHES(
                (LS{base, 10, 10, -1, 0, 10}),
                svs::ANNException,
                svs_test::ExceptionMatcher(Catch::Matchers::ContainsSubstring(
                    "Batch size start must be at least 1"
                ))
            );
        }

        // Minimal constructor - should behave like the default schedule.
        test_default_schedule(LS{base, 4}, 4, {4, 10}, {4, 23}, 4, 4, 0);

        auto ls = LS{base, 4, 5, 3, 2, 20};
        test_linear_schedule(ls, 4, {4, 10}, {5, 23}, {20, 2}, 3, 4, 0);
        /// buffer scaling
        ls.buffer_scaling({5, 6});
        test_linear_schedule(ls, 4, {5, 10}, {6, 23}, {20, 2}, 3, 4, 0);

        /// visited set
        ls.enable_filter_after(0);
        test_linear_schedule(ls, 4, {5, 10}, {6, 23}, {20, 2}, 0, 4, 0);
        ls.disable_filter();
        test_linear_schedule(ls, 4, {5, 10}, {6, 23}, {20, 2}, -1, 4, 0);

        /// starting batch size
        ls.starting_batch_size(4);
        test_linear_schedule(ls, 4, {5, 10}, {6, 23}, {20, 4}, -1, 4, 0);

        // Should get an exception if misconfigured.
        CATCH_REQUIRE_THROWS_MATCHES(
            ls.starting_batch_size(0),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring("Starting batch size must be nonzero.")
            )
        );

        /// batch size scaling
        ls.batch_size_scaling(3);
        test_linear_schedule(ls, 4, {5, 10}, {6, 23}, {3, 4}, -1, 4, 0);

        ls.disable_batch_size_scaling();
        test_linear_schedule(ls, 4, {5, 10}, {6, 23}, {0, 4}, -1, 4, 0);
    }

    CATCH_SECTION("Abstract Iterator Schedule") {
        using LS = svs::index::vamana::LinearSchedule;
        auto base = VSP{{10, 20}, false, 1, 4};
        auto sched = svs::index::vamana::DefaultSchedule{base, 5};
        auto abstract = svs::index::vamana::AbstractIteratorSchedule{sched};

        auto test_default = [](const auto& sched, size_t batchsize) {
            test_default_schedule(
                sched, 4, {batchsize, 10}, {batchsize, 20}, batchsize, 1, 4
            );
        };

        test_default(sched, 5);
        test_default(abstract, 5);

        // Construct using `std::in_place_type`.
        // Also test the move-assignment operator while we're at it.
        abstract = svs::index::vamana::AbstractIteratorSchedule{
            std::in_place_type<svs::index::vamana::DefaultSchedule>, base, size_t{10}};

        test_default(abstract, 10);

        // Copy constructor.
        auto copy = abstract;
        test_default(abstract, 10);

        // Move assignment.
        {
            auto linear =
                svs::index::vamana::AbstractIteratorSchedule{LS{base, 2}
                                                                 .batch_size_scaling(20)
                                                                 .buffer_scaling({4, 5})
                                                                 .enable_filter_after(3)};
            test_linear_schedule(linear, 4, {4, 10}, {5, 20}, {20, 2}, 3, 1, 4);
            abstract = std::move(linear);
            test_linear_schedule(abstract, 4, {4, 10}, {5, 20}, {20, 2}, 3, 1, 4);
        }

        // copy-assignment.
        copy = abstract;
        test_linear_schedule(copy, 4, {4, 10}, {5, 20}, {20, 2}, 3, 1, 4);

        // move-construction
        {
            auto another_copy = std::move(copy);
            test_linear_schedule(another_copy, 4, {4, 10}, {5, 20}, {20, 2}, 3, 1, 4);
        }

        // Reset
        copy.reset(sched);
        test_default(copy, 5);
    }
}
