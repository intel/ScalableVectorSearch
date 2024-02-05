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
#include "svs/lib/prefetch.h"

// svs-test
#include "tests/utils/generators.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_random.hpp"

// stdlib
#include <algorithm>
#include <vector>

namespace {

///// Shared Logic

std::vector<size_t>
generate_expected_sequence(size_t total_items, size_t lookahead, size_t step) {
    // If `step == 0`, then no prefetching should be performed.
    if (step == 0) {
        total_items = 0;
    }

    auto expected = std::vector<size_t>();
    size_t iteration = 1;
    do {
        auto back = std::min<size_t>({
            total_items,              // Don't prefetch off the end.
            0 + step * iteration,     // Ramp-up phase.
            lookahead + 1 * iteration // steady-state phase.
        });
        expected.push_back(back);
        ++iteration;
    } while (expected.back() != total_items);
    return expected;
}

///// Unpredicated prefetching.

// Check that all items have been marked up until `count` items. All further items should
// not be marked.
void check_until(const std::vector<bool>& marked, size_t count) {
    for (size_t i = 0; i < marked.size(); ++i) {
        if (i < count) {
            CATCH_REQUIRE(marked.at(i) == true);
        } else {
            CATCH_REQUIRE(marked.at(i) == false);
        }
    }
}

void check_unpredicated(
    svs::lib::PrefetchParameters p, size_t length, size_t effective_step
) {
    auto marked = std::vector<bool>(length);
    auto expected = generate_expected_sequence(length, p.lookahead, effective_step);
    auto f =
        svs::lib::make_prefetcher(p, marked.size(), [&](size_t i) { marked.at(i) = true; });
    for (size_t i = 0; i < expected.size(); ++i) {
        f();
        check_until(marked, expected.at(i));
    }
    // Run once more - should have no effect.
    f();
    check_until(marked, expected.back());
}

///// Predicated

// Check that all `true` marked items have been marked as `true` up until `count`, and that
// all items after `count` are marked as false.
void check_until(
    const std::vector<bool>& marked, const std::vector<uint8_t>& valid, size_t count
) {
    CATCH_REQUIRE(marked.size() == valid.size());
    size_t valid_seen = 0;
    for (size_t i = 0; i < marked.size(); ++i) {
        if (valid.at(i) == 0) {
            CATCH_REQUIRE(marked.at(i) == false);
            continue;
        }

        if (valid_seen < count) {
            CATCH_REQUIRE(marked.at(i) == true);
        } else {
            CATCH_REQUIRE(marked.at(i) == false);
        }
        ++valid_seen;
    }
}

void check_predicated(
    svs::lib::PrefetchParameters p, size_t length, size_t effective_step
) {
    auto marked = std::vector<bool>(length);
    auto predicate = std::vector<uint8_t>(length);
    auto generator = svs_test::make_generator<uint8_t>(0, 1);
    svs_test::populate(predicate, generator);

    size_t valid = std::count(predicate.begin(), predicate.end(), 1);

    auto expected = generate_expected_sequence(valid, p.lookahead, effective_step);
    auto f = svs::lib::make_prefetcher(
        p,
        marked.size(),
        [&](size_t i) { marked.at(i) = true; },
        [&](size_t i) { return predicate.at(i); }
    );
    for (size_t i = 0; i < expected.size(); ++i) {
        f();
        check_until(marked, predicate, expected.at(i));
    }
    // Run once more - should have no effect.
    f();
    check_until(marked, predicate, expected.back());
}

} // namespace

CATCH_TEST_CASE("Prefetcher", "[lib][prefetch]") {
    CATCH_SECTION("Expected Sequence") {
        CATCH_REQUIRE(
            generate_expected_sequence(10, 3, 4) ==
            std::vector<size_t>{4, 5, 6, 7, 8, 9, 10}
        );
    }

    CATCH_SECTION("Unpredicated") {
        // The following step sizes should all behave similarly with a hard ramp.
        check_unpredicated(svs::lib::PrefetchParameters{3, 1}, 10, 4);
        check_unpredicated(svs::lib::PrefetchParameters{3, 4}, 10, 4);
        check_unpredicated(svs::lib::PrefetchParameters{3, 10}, 10, 4);

        // More gradual ramps.
        check_unpredicated(svs::lib::PrefetchParameters{4, 2}, 10, 2);
        check_unpredicated(svs::lib::PrefetchParameters{4, 4}, 10, 4);

        // Pathological cases - lookahead is greater the total size.
        check_unpredicated(svs::lib::PrefetchParameters{20, 2}, 10, 2);
        check_unpredicated(svs::lib::PrefetchParameters{20, 1}, 10, 21);
        check_unpredicated(svs::lib::PrefetchParameters{20, 5}, 10, 5);
        check_unpredicated(svs::lib::PrefetchParameters{20, 10}, 10, 10);

        // Zero-sized array.
        check_unpredicated(svs::lib::PrefetchParameters{20, 2}, 0, 2);

        // Pathological edge-cases.
        // Zero-step
        check_unpredicated(svs::lib::PrefetchParameters{3, 0}, 10, 0);
        check_unpredicated(svs::lib::PrefetchParameters{0, 3}, 10, 0);
    }

    CATCH_SECTION("Predicated") {
        // The following step sizes should all behave similarly with a hard ramp.
        check_predicated(svs::lib::PrefetchParameters{3, 1}, 10, 4);
        check_predicated(svs::lib::PrefetchParameters{3, 4}, 10, 4);
        check_predicated(svs::lib::PrefetchParameters{3, 10}, 10, 4);

        // More gradual ramps.
        check_predicated(svs::lib::PrefetchParameters{4, 2}, 10, 2);
        check_predicated(svs::lib::PrefetchParameters{4, 4}, 10, 4);

        // Pathological cases - lookahead is greater the total size.
        check_predicated(svs::lib::PrefetchParameters{20, 2}, 10, 2);
        check_predicated(svs::lib::PrefetchParameters{20, 1}, 10, 21);
        check_predicated(svs::lib::PrefetchParameters{20, 5}, 10, 5);
        check_predicated(svs::lib::PrefetchParameters{20, 10}, 10, 10);

        // Zero-sized array.
        check_predicated(svs::lib::PrefetchParameters{20, 2}, 0, 2);

        // Zero-step
        check_predicated(svs::lib::PrefetchParameters{3, 0}, 10, 0);
        check_predicated(svs::lib::PrefetchParameters{0, 3}, 10, 0);
    }
}
