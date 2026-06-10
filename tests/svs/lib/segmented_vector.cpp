/*
 * Copyright 2026 Intel Corporation
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
#include "svs/lib/segmented_vector.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <atomic>
#include <thread>
#include <vector>

namespace {

// Small segment size so even modest element counts exercise many segments and
// several directory buckets.
using SmallVec = svs::lib::SegmentedVector<int, 4>;

} // namespace

CATCH_TEST_CASE("SegmentedVector basic semantics", "[core][segmented_vector]") {
    CATCH_SECTION("Default construction is empty") {
        SmallVec v;
        CATCH_REQUIRE(v.size() == 0);
    }

    CATCH_SECTION("Sized construction and read/write") {
        SmallVec v(10);
        CATCH_REQUIRE(v.size() == 10);
        for (size_t i = 0; i < 10; ++i) {
            v[i] = static_cast<int>(i * 2);
        }
        for (size_t i = 0; i < 10; ++i) {
            CATCH_REQUIRE(v[i] == static_cast<int>(i * 2));
        }
    }

    CATCH_SECTION("Fill construction") {
        SmallVec v(7, 42);
        CATCH_REQUIRE(v.size() == 7);
        for (size_t i = 0; i < 7; ++i) {
            CATCH_REQUIRE(v[i] == 42);
        }
    }

    CATCH_SECTION("resize grows and preserves existing elements") {
        SmallVec v(5);
        for (size_t i = 0; i < 5; ++i) {
            v[i] = static_cast<int>(i);
        }
        v.resize(100);
        CATCH_REQUIRE(v.size() == 100);
        for (size_t i = 0; i < 5; ++i) {
            CATCH_REQUIRE(v[i] == static_cast<int>(i));
        }
    }

    CATCH_SECTION("resize with fill") {
        SmallVec v(3, 1);
        v.resize(20, 9);
        CATCH_REQUIRE(v.size() == 20);
        for (size_t i = 0; i < 3; ++i) {
            CATCH_REQUIRE(v[i] == 1);
        }
        for (size_t i = 3; i < 20; ++i) {
            CATCH_REQUIRE(v[i] == 9);
        }
    }

    CATCH_SECTION("logical shrink via resize then regrow") {
        SmallVec v(50);
        for (size_t i = 0; i < 50; ++i) {
            v[i] = static_cast<int>(i);
        }
        v.resize(10);
        CATCH_REQUIRE(v.size() == 10);
        for (size_t i = 0; i < 10; ++i) {
            CATCH_REQUIRE(v[i] == static_cast<int>(i));
        }
        // Regrow: previously-allocated segments are reused, values intact below 50.
        v.resize(50);
        CATCH_REQUIRE(v.size() == 50);
        for (size_t i = 0; i < 50; ++i) {
            CATCH_REQUIRE(v[i] == static_cast<int>(i));
        }
    }

    CATCH_SECTION("shrink_to frees segments and lowers size") {
        SmallVec v(100);
        for (size_t i = 0; i < 100; ++i) {
            v[i] = static_cast<int>(i);
        }
        v.shrink_to(7);
        CATCH_REQUIRE(v.size() == 7);
        for (size_t i = 0; i < 7; ++i) {
            CATCH_REQUIRE(v[i] == static_cast<int>(i));
        }
        // Regrow after a real shrink: new elements default-constructed (0).
        v.resize(20);
        CATCH_REQUIRE(v.size() == 20);
        for (size_t i = 0; i < 7; ++i) {
            CATCH_REQUIRE(v[i] == static_cast<int>(i));
        }
    }
}

CATCH_TEST_CASE("SegmentedVector copy and move", "[core][segmented_vector]") {
    SmallVec v(30);
    for (size_t i = 0; i < 30; ++i) {
        v[i] = static_cast<int>(i + 100);
    }

    CATCH_SECTION("copy construction is a deep copy") {
        SmallVec c(v);
        CATCH_REQUIRE(c.size() == 30);
        for (size_t i = 0; i < 30; ++i) {
            CATCH_REQUIRE(c[i] == static_cast<int>(i + 100));
        }
        c[0] = -1;
        CATCH_REQUIRE(v[0] == 100); // original unaffected
    }

    CATCH_SECTION("move construction transfers contents") {
        SmallVec m(std::move(v));
        CATCH_REQUIRE(m.size() == 30);
        for (size_t i = 0; i < 30; ++i) {
            CATCH_REQUIRE(m[i] == static_cast<int>(i + 100));
        }
    }
}

// Address stability: a reference to v[i] obtained before a grow must remain valid
// (point to the same storage) after the grow. This is the core invariant Option C
// relies on.
CATCH_TEST_CASE(
    "SegmentedVector grow preserves element addresses", "[core][segmented_vector]"
) {
    SmallVec v(8);
    for (size_t i = 0; i < 8; ++i) {
        v[i] = static_cast<int>(i);
    }
    std::vector<int*> addrs;
    for (size_t i = 0; i < 8; ++i) {
        addrs.push_back(&v[i]);
    }
    // Force many segment + directory-bucket allocations.
    v.resize(10000);
    for (size_t i = 0; i < 8; ++i) {
        CATCH_REQUIRE(&v[i] == addrs[i]);                // address unchanged
        CATCH_REQUIRE(*addrs[i] == static_cast<int>(i)); // value intact
    }
}

// Concurrent grow vs. read: one writer repeatedly grows the vector and fills the new
// elements, publishing a separate `published` counter ONLY after the fill — exactly the
// "publish after construct" contract the dynamic index uses (num_valid_ bumped after a
// slot is fully Valid). Readers read every index < published and assert v[i] == i with
// no tolerance. This both (a) catches use-after-free / torn outer-pointer reads on grow
// and (b) is a hard value invariant. Under TSan it must report no data races.
CATCH_TEST_CASE("SegmentedVector concurrent grow and read", "[core][segmented_vector]") {
    using Vec = svs::lib::SegmentedVector<int, 64>;
    constexpr size_t kInitial = 100;
    constexpr size_t kFinal = 200000;
    constexpr size_t kReaders = 8;

    Vec v(kInitial);
    for (size_t i = 0; i < kInitial; ++i) {
        v[i] = static_cast<int>(i);
    }

    std::atomic<bool> start{false};
    std::atomic<bool> writer_done{false};
    std::atomic<size_t> published{kInitial};
    std::atomic<bool> failure{false};

    auto reader = [&]() {
        while (!start.load(std::memory_order_acquire)) {}
        while (!writer_done.load(std::memory_order_acquire)) {
            size_t n = published.load(std::memory_order_acquire);
            size_t step = n / 256 + 1;
            for (size_t i = 0; i < n; i += step) {
                if (v[i] != static_cast<int>(i)) {
                    failure.store(true, std::memory_order_relaxed);
                }
            }
        }
    };

    std::vector<std::thread> readers;
    for (size_t r = 0; r < kReaders; ++r) {
        readers.emplace_back(reader);
    }

    std::thread writer([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        size_t cur = kInitial;
        while (cur < kFinal) {
            size_t next = std::min(cur + 137, kFinal);
            v.resize(next);                       // allocate + publish segments (grow)
            for (size_t i = cur; i < next; ++i) { // fill new elements
                v[i] = static_cast<int>(i);
            }
            published.store(next, std::memory_order_release); // publish after fill
            cur = next;
        }
        writer_done.store(true, std::memory_order_release);
    });

    start.store(true, std::memory_order_release);
    writer.join();
    for (auto& t : readers) {
        t.join();
    }

    CATCH_REQUIRE(v.size() == kFinal);
    CATCH_REQUIRE_FALSE(failure.load(std::memory_order_relaxed));
}
