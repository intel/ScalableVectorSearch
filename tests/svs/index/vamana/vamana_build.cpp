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

// header under tests
// mainly tests the utility classes define in the header.
#include "svs/index/vamana/vamana_build.h"

// svs
#include "svs/lib/threads.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <chrono>
#include <random>
#include <thread>

namespace vamana = svs::index::vamana;

namespace {

void test_enabled_tracker(vamana::OptionalTracker<uint32_t>& tracker) {
    CATCH_REQUIRE(tracker.enabled());
    tracker.clear();
    CATCH_REQUIRE(tracker.size() == 0);

    auto neighbors = std::vector<svs::Neighbor<uint32_t>>{{
        {0, 10.0f},
        {10, 20.0f},
        {20, 5.0f},
        {10, 20.0f}, // intentional repeat
        {30, 1.0f},
    }};
    for (const auto& neighbor : neighbors) {
        tracker.visited(neighbor, 0);
    }

    auto unique = std::vector<svs::Neighbor<uint32_t>>{neighbors.begin(), neighbors.end()};
    std::sort(unique.begin(), unique.end());
    auto it = std::unique(unique.begin(), unique.end(), svs::NeighborEqual());
    unique.resize(it - unique.begin());
    CATCH_REQUIRE(unique.size() == neighbors.size() - 1);

    CATCH_REQUIRE(tracker.size() == unique.size());
    auto seen = std::vector<svs::Neighbor<uint32_t>>{tracker.begin(), tracker.end()};
    std::sort(seen.begin(), seen.end());
    CATCH_REQUIRE(seen.size() == unique.size());
    CATCH_REQUIRE(std::equal(seen.begin(), seen.end(), unique.begin(), svs::NeighborEqual())
    );
}
} // namespace

CATCH_TEST_CASE("Index Build Utilties", "[vamana][vamana_build]") {
    CATCH_SECTION("OptionalTracker") {
        // Ensure that the tracker satisfies the constraint requirement.
        static_assert(vamana::
                          GreedySearchTracker<vamana::OptionalTracker<uint32_t>, uint32_t>);

        auto tracker = vamana::OptionalTracker<uint32_t>(false);
        CATCH_REQUIRE(!tracker.enabled());
        CATCH_REQUIRE(tracker.size() == 0);
        // clearing should work
        tracker.clear();
        // size should still be zero.
        CATCH_REQUIRE(!tracker.enabled());
        CATCH_REQUIRE(tracker.size() == 0);

        tracker = vamana::OptionalTracker<uint32_t>(true);
        CATCH_REQUIRE(tracker.enabled());
        test_enabled_tracker(tracker);
    }

    CATCH_SECTION("BackedgeBuffer") {
        const size_t num_elements = 50;
        const size_t bucket_size = 25;
        auto buffer = vamana::BackedgeBuffer<uint32_t>{num_elements, bucket_size};
        CATCH_REQUIRE(buffer.num_buckets() == 2);

        // Bucket `i` will contain the entries `[10 * i, 10 * (i + 1))` to ensure unique
        // entries within each bucket.
        //
        // Entries `[10*i, 10*i + 7)` will be added by thread 1.
        // Entries `[10*i + 4, 20 * i)` will be added by thread 2.
        // The regions added by each thread intentionally overlap to ensure that the
        // buffer correctly handles repeated elements.
        auto threadpool = svs::threads::NativeThreadPool(2);
        CATCH_REQUIRE(threadpool.size() == 2);
        svs::threads::run(threadpool, [&](auto tid) {
            // Random number generator per thread.
            std::mt19937_64 engine{std::random_device{}()};
            auto dist = std::uniform_int_distribution<>{1, 10};

            for (size_t i = 0; i < num_elements; ++i) {
                size_t j = (tid == 0) ? (10 * i) : (10 * i + 4);
                size_t stop = (tid == 0) ? (10 * i + 7) : (10 * i + 10);
                for (; j < stop; ++j) {
                    buffer.add_edge(i, j);
                    std::this_thread::sleep_for(std::chrono::microseconds(dist(engine)));
                }
            }
        });

        // Make sure the results all make sense.
        const auto& buckets = buffer.buckets();
        CATCH_REQUIRE(buckets.size() == 2);

        // Check bucket `index`.
        auto check_bucket = [&](size_t index) {
            const auto& bucket = buckets.at(index);
            for (size_t i = bucket_size * index, imax = i + bucket_size; i < imax; ++i) {
                const auto& values = bucket.at(i);
                // Each bucket should contain 10 items.
                CATCH_REQUIRE(values.size() == 10);
                // Make sure each expected item is in the value-set.
                for (size_t j = 10 * i, jmax = 10 * (i + 1); j < jmax; ++j) {
                    CATCH_REQUIRE(values.contains(j));
                }
            }
        };

        check_bucket(0);
        check_bucket(1);

        // Check buffer reset.
        buffer.reset();
        for (const auto& b : buffer.buckets()) {
            CATCH_REQUIRE(b.empty());
        }
    }
}
