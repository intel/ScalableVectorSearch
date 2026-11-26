/*
 * Copyright 2023 Intel Corporation
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

// Multi-threaded tests of ``svs::ConcurrentIDTranslator``: reads issued while one writer
// mutates the translator. Its single-threaded behaviour is covered, together with the
// plain variant, in ``tests/svs/core/translation.cpp``.
//
// A reader can only check what is true at every instant, so each test fixes the set of
// external IDs an internal ID may ever map to and requires every read to land in that set.
// The data-race half of the contract is only visible to ThreadSanitizer; this file is part
// of the ``concurrent_tsan`` target for that reason.
//
// Assertions inside the reader loops accumulate into counters and are checked once at the
// end: Catch2's assertion bookkeeping is neither free nor safe to call from several
// threads at once.

// header under test.
#include "svs/core/translation.h"

// Catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using Translator = svs::ConcurrentIDTranslator;
using external_id_type = Translator::external_id_type;
using internal_id_type = Translator::internal_id_type;

constexpr size_t kReaders = 4;

// The two external IDs the tests ever give the translation that starts in slot `owner`.
constexpr external_id_type primary(size_t owner) { return 2 * owner; }
constexpr external_id_type renamed(size_t owner) { return 2 * owner + 1; }

// Defaults handed to the `_or` lookups. Neither is an ID the tests store, and
// `kNoExternal` differs from the translator's own reserved ID, so a lookup that leaked the
// reserved ID instead of the default would be caught.
constexpr external_id_type kNoExternal = Translator::invalid_external_id - 1;
constexpr internal_id_type kNoInternal = std::numeric_limits<internal_id_type>::max();

// Run `writer` once on its own thread while `kReaders` threads call `reader` in a loop.
// Every reader makes at least one pass, including one after the writer has finished.
template <typename Writer, typename Reader>
void run_against_readers(Writer&& writer, Reader&& reader) {
    std::atomic<bool> start{false};
    std::atomic<bool> writer_done{false};

    std::vector<std::thread> readers;
    for (size_t r = 0; r < kReaders; ++r) {
        readers.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            bool last_pass = false;
            while (!last_pass) {
                last_pass = writer_done.load(std::memory_order_acquire);
                reader();
            }
        });
    }
    std::thread writer_thread([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        writer();
        writer_done.store(true, std::memory_order_release);
    });

    start.store(true, std::memory_order_release);
    writer_thread.join();
    for (auto& t : readers) {
        t.join();
    }
}

} // namespace

// The writer appends translations in batches and publishes a counter only after a batch is
// inserted. Readers hold no lock: a published internal ID must read back exactly, and an
// unpublished one is either still unmapped or already holds its final value. This is the
// case that relocating storage would break, since appending grows the table under the
// readers.
CATCH_TEST_CASE(
    "ConcurrentIDTranslator lock-free reads during growth", "[concurrent][translation]"
) {
    constexpr size_t kFinal = 100'000;
    constexpr size_t kBatch = 137;

    auto translator = Translator();
    std::atomic<size_t> published{0};
    std::atomic<size_t> failures{0};

    auto writer = [&]() {
        auto external_ids = std::vector<external_id_type>();
        auto internal_ids = std::vector<internal_id_type>();
        for (size_t cur = 0; cur < kFinal;) {
            const size_t next = std::min(cur + kBatch, kFinal);
            external_ids.clear();
            internal_ids.clear();
            for (size_t i = cur; i < next; ++i) {
                external_ids.push_back(primary(i));
                internal_ids.push_back(static_cast<internal_id_type>(i));
            }
            translator.insert(external_ids, internal_ids);
            published.store(next, std::memory_order_release);
            cur = next;
        }
    };

    auto reader = [&]() {
        const size_t n = published.load(std::memory_order_acquire);
        size_t bad = 0;
        const size_t step = n / 256 + 1;
        for (size_t i = 0; i < n; i += step) {
            const auto internal = static_cast<internal_id_type>(i);
            bad += !translator.has_internal(internal);
            bad += translator.get_external_or(internal, kNoExternal) != primary(i);
        }
        // The batch being inserted right now, and the slots beyond the table.
        for (size_t i = n; i < n + 2 * kBatch; ++i) {
            const auto e =
                translator.get_external_or(static_cast<internal_id_type>(i), kNoExternal);
            bad += e != kNoExternal && e != primary(i);
        }
        failures.fetch_add(bad, std::memory_order_relaxed);
    };

    run_against_readers(writer, reader);

    CATCH_REQUIRE(failures.load() == 0);
    CATCH_REQUIRE(translator.size() == kFinal);
    for (size_t i = 0; i < kFinal; ++i) {
        const auto internal = static_cast<internal_id_type>(i);
        if (translator.get_external(internal) != primary(i) ||
            translator.get_internal(primary(i)) != internal) {
            CATCH_FAIL("wrong translation for internal ID " << i);
        }
    }
    CATCH_REQUIRE_FALSE(translator.has_internal(static_cast<internal_id_type>(kFinal)));
}

// The writer renames, deletes, re-inserts and moves translations over a fixed set of
// slots. Readers hold no lock. Slot `s` only ever holds the translation that started in
// slot `s % kSlots`, under one of its two external IDs, so every read must return one of
// those two IDs or report the slot unmapped.
CATCH_TEST_CASE(
    "ConcurrentIDTranslator lock-free reads during mutation", "[concurrent][translation]"
) {
    constexpr size_t kSlots = 512;
    constexpr size_t kRounds = 200;

    auto translator = Translator();
    auto all_external = std::vector<external_id_type>();
    auto all_internal = std::vector<internal_id_type>();
    auto even_external = std::vector<external_id_type>();
    auto even_internal = std::vector<internal_id_type>();
    for (size_t i = 0; i < kSlots; ++i) {
        all_external.push_back(primary(i));
        all_internal.push_back(static_cast<internal_id_type>(i));
        if (i % 2 == 0) {
            even_external.push_back(primary(i));
            even_internal.push_back(static_cast<internal_id_type>(i));
        }
    }
    translator.insert(all_external, all_internal);

    std::atomic<size_t> failures{0};

    auto writer = [&]() {
        for (size_t round = 0; round < kRounds; ++round) {
            for (size_t i = 0; i < kSlots; ++i) {
                translator.remap_external_id(primary(i), renamed(i));
            }
            // Even slots: delete under the new name, come back under the old one.
            translator.delete_internal(even_internal);
            translator.insert(even_external, even_internal);
            // Odd slots: move beyond the end of the original range and back, then take
            // the old name again.
            for (size_t i = 1; i < kSlots; i += 2) {
                const auto from = static_cast<internal_id_type>(i);
                const auto to = static_cast<internal_id_type>(i + kSlots);
                translator.remap_internal_id(from, to);
                translator.remap_internal_id(to, from);
                translator.remap_external_id(renamed(i), primary(i));
            }
        }
    };

    auto reader = [&]() {
        size_t bad = 0;
        for (size_t slot = 0; slot < 2 * kSlots; ++slot) {
            const size_t owner = slot % kSlots;
            const auto internal = static_cast<internal_id_type>(slot);
            auto legal = [&](external_id_type e) {
                return e == primary(owner) || e == renamed(owner);
            };

            const auto e = translator.get_external_or(internal, kNoExternal);
            bad += e != kNoExternal && !legal(e);

            // The throwing lookup, on a sample: an exception per slot would dominate the
            // loop.
            if (slot % 16 == 0) {
                try {
                    bad += !legal(translator.get_external(internal));
                } catch (const std::out_of_range&) {
                    // Unmapped at the instant of the read.
                }
            }
            // No assertion possible on the result, but the call must be race-free.
            static_cast<void>(translator.has_internal(internal));
        }
        failures.fetch_add(bad, std::memory_order_relaxed);
    };

    run_against_readers(writer, reader);

    CATCH_REQUIRE(failures.load() == 0);
    // Every round ends where it started.
    CATCH_REQUIRE(translator.size() == kSlots);
    for (size_t i = 0; i < kSlots; ++i) {
        const auto internal = static_cast<internal_id_type>(i);
        if (translator.get_external(internal) != primary(i) ||
            translator.get_internal(primary(i)) != internal ||
            translator.has_internal(static_cast<internal_id_type>(i + kSlots))) {
            CATCH_FAIL("wrong translation for internal ID " << i);
        }
    }
}

// The external-to-internal direction is a hash map and needs a lock. This mirrors how the
// concurrent index uses the translator: writers exclusive, readers shared. Under the
// shared lock a reader sees a state between two writer steps, in which the two directions
// agree with each other.
CATCH_TEST_CASE(
    "ConcurrentIDTranslator both directions under a shared lock",
    "[concurrent][translation]"
) {
    constexpr size_t kSlots = 256;
    constexpr size_t kRounds = 300;

    auto translator = Translator();
    auto all_external = std::vector<external_id_type>();
    auto all_internal = std::vector<internal_id_type>();
    auto even_external = std::vector<external_id_type>();
    auto even_internal = std::vector<internal_id_type>();
    for (size_t i = 0; i < kSlots; ++i) {
        all_external.push_back(primary(i));
        all_internal.push_back(static_cast<internal_id_type>(i));
        if (i % 2 == 0) {
            even_external.push_back(primary(i));
            even_internal.push_back(static_cast<internal_id_type>(i));
        }
    }
    translator.insert(all_external, all_internal);

    std::shared_mutex mutex;
    std::atomic<size_t> failures{0};
    // Bumped after every writer step; see the end of `reader`.
    std::atomic<size_t> writer_steps{0};
    std::atomic<bool> writer_finished{false};

    // Run one writer step under the exclusive lock.
    auto exclusive_step = [&](auto&& step) {
        {
            std::unique_lock lock{mutex};
            step();
        }
        writer_steps.fetch_add(1, std::memory_order_release);
    };

    auto writer = [&]() {
        for (size_t round = 0; round < kRounds; ++round) {
            exclusive_step([&]() { translator.delete_external(even_external); });
            // Odd slots move into the upper half of the slot range.
            exclusive_step([&]() {
                for (size_t i = 1; i < kSlots; i += 2) {
                    translator.remap_internal_id(
                        static_cast<internal_id_type>(i),
                        static_cast<internal_id_type>(i + kSlots)
                    );
                }
            });
            exclusive_step([&]() { translator.insert(even_external, even_internal); });
            exclusive_step([&]() {
                for (size_t i = 1; i < kSlots; i += 2) {
                    translator.remap_internal_id(
                        static_cast<internal_id_type>(i + kSlots),
                        static_cast<internal_id_type>(i)
                    );
                }
            });
        }
        writer_finished.store(true, std::memory_order_release);
    };

    auto reader = [&]() {
        const size_t step_seen = writer_steps.load(std::memory_order_acquire);
        size_t bad = 0;
        std::shared_lock lock{mutex};
        size_t mapped = 0;
        for (size_t owner = 0; owner < kSlots; ++owner) {
            const auto e = primary(owner);
            const auto internal = translator.get_internal_or(e, kNoInternal);
            bad += translator.has_external(e) != (internal != kNoInternal);
            if (internal == kNoInternal) {
                continue;
            }
            ++mapped;
            bad += internal % kSlots != owner;
            bad += translator.get_external_or(internal, kNoExternal) != e;
        }
        bad += translator.size() != mapped;
        // Slots without an owner mapped to them read as unmapped.
        size_t occupied = 0;
        for (size_t slot = 0; slot < 2 * kSlots; ++slot) {
            occupied += translator.has_internal(static_cast<internal_id_type>(slot));
        }
        bad += occupied != mapped;
        failures.fetch_add(bad, std::memory_order_relaxed);

        // One pass per writer step, waiting for the next step without the lock. Readers
        // that re-lock back to back never let the lock go free, and a reader-preferring
        // `std::shared_mutex` (glibc) then starves the writer forever.
        lock.unlock();
        while (writer_steps.load(std::memory_order_acquire) == step_seen &&
               !writer_finished.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    };

    run_against_readers(writer, reader);

    CATCH_REQUIRE(failures.load() == 0);
    CATCH_REQUIRE(writer_steps.load() == 4 * kRounds);
    CATCH_REQUIRE(translator.size() == kSlots);
    for (size_t i = 0; i < kSlots; ++i) {
        const auto internal = static_cast<internal_id_type>(i);
        if (translator.get_external(internal) != primary(i) ||
            translator.get_internal(primary(i)) != internal) {
            CATCH_FAIL("wrong translation for internal ID " << i);
        }
    }
}
