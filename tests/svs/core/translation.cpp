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

// header under test.
#include "svs/core/translation.h"

// misc utilities
#include "svs/lib/narrow.h"

// Test utils
#include "tests/utils/generators.h"

// Catch2
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
template <typename Begin, typename End, typename External, typename Internal>
void check_contents(
    const Begin& begin, const End& end, const External& external, const Internal& internal
) {
    int64_t dist = std::distance(begin, end);
    CATCH_REQUIRE(dist == svs::lib::narrow<int64_t>(external.size()));
    CATCH_REQUIRE(dist == svs::lib::narrow<int64_t>(internal.size()));

    size_t count = 0;
    for (auto i = begin; i != end; ++i, ++count) {
        CATCH_REQUIRE(i->first == external[count]);
        CATCH_REQUIRE(i->second == internal[count]);
    }
}

template <typename Translator, typename External, typename Internal>
void check_contents(
    const Translator& translator, const External& external, const Internal& internal
) {
    check_contents(translator.begin(), translator.end(), external, internal);
}

template <typename Translator, typename External, typename Internal>
void check_translation(
    const Translator& translator, const External& external, const Internal& internal
) {
    CATCH_REQUIRE(translator.size() == external.size());
    CATCH_REQUIRE(translator.size() == internal.size());

    for (size_t i = 0, imax = translator.size(); i < imax; ++i) {
        CATCH_REQUIRE(translator.has_external(external[i]));
        CATCH_REQUIRE(translator.has_internal(internal[i]));
        CATCH_REQUIRE(translator.get_internal(external[i]) == internal[i]);
        CATCH_REQUIRE(translator.get_external(internal[i]) == external[i]);
    }
}

template <typename Translator>
void check(
    const Translator& translator,
    std::vector<uint64_t>& external_ids,
    std::vector<uint32_t>& internal_ids
) {
    check_translation(translator, external_ids, internal_ids);
    check_contents(translator, external_ids, internal_ids);
}

// Both directions of `lhs` are present in `rhs`, and nothing else is.
template <typename Lhs, typename Rhs> void check_same(const Lhs& lhs, const Rhs& rhs) {
    CATCH_REQUIRE(lhs.size() == rhs.size());
    for (auto pair : lhs) {
        CATCH_REQUIRE(rhs.get_internal(pair.first) == pair.second);
        CATCH_REQUIRE(rhs.get_external(pair.second) == pair.first);
    }
}

template <typename Translator>
using other_variant = std::conditional_t<
    std::is_same_v<Translator, svs::IDTranslator>,
    svs::ConcurrentIDTranslator,
    svs::IDTranslator>;

} // namespace

// Single-threaded behaviour is the same for both variants; the lock-free reads of the
// concurrent one are covered in `tests/svs/concurrent/translation.cpp`.
CATCH_TEMPLATE_TEST_CASE(
    "Translation Table",
    "[core][translation]",
    svs::IDTranslator,
    svs::ConcurrentIDTranslator
) {
    using Translator = TestType;
    using external_id_type = typename Translator::external_id_type;
    using internal_id_type = typename Translator::internal_id_type;

    CATCH_SECTION("Identity Constructor") {
        auto translator = Translator(typename Translator::Identity(10));
        CATCH_REQUIRE(translator.size() == 10);
        for (size_t i = 0; i < 10; ++i) {
            CATCH_REQUIRE(translator.has_external(i));
            CATCH_REQUIRE(translator.has_internal(i));
            CATCH_REQUIRE(translator.get_external(i) == i);
            CATCH_REQUIRE(translator.get_internal(i) == i);
        }
    }

    CATCH_SECTION("Basic Tests") {
        auto translator = Translator();
        CATCH_REQUIRE(translator.size() == 0);

        auto external_ids = std::vector<uint64_t>{0, 2, 4, 6, 8};
        auto internal_ids = std::vector<uint32_t>{0, 10, 20, 30, 40};

        CATCH_REQUIRE(external_ids.size() == internal_ids.size());
        translator.insert(external_ids, internal_ids);

        check(translator, external_ids, internal_ids);

        ///
        /// Adding more IDs
        ///
        CATCH_SECTION("Adding more entries") {
            auto extra_external_ids = std::vector<uint64_t>{10, 12, 14};
            auto extra_internal_ids = std::vector<uint32_t>{50, 60, 70};
            translator.insert(extra_external_ids, extra_internal_ids);
            external_ids.insert(
                external_ids.end(), extra_external_ids.begin(), extra_external_ids.end()
            );
            internal_ids.insert(
                internal_ids.end(), extra_internal_ids.begin(), extra_internal_ids.end()
            );

            check(translator, external_ids, internal_ids);
        }

        ///
        /// Error Handling
        ///
        CATCH_SECTION("Mismatched number of entries") {
            // Mismatched number of internal and external IDs.
            auto external_mismatch_ids = std::vector<uint64_t>{10, 12, 14};
            auto internal_mismatch_ids = std::vector<uint32_t>{50, 60};
            CATCH_REQUIRE_THROWS_AS(
                translator.insert(external_mismatch_ids, internal_mismatch_ids),
                svs::ANNException
            );

            // State of the translator should be unchanged.
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Repeat External IDs") {
            auto external_mismatch_ids = std::vector<uint64_t>{10, 12, 8};
            auto internal_mismatch_ids = std::vector<uint32_t>{50, 60, 70};

            CATCH_REQUIRE_THROWS_AS(
                translator.insert(external_mismatch_ids, internal_mismatch_ids),
                svs::ANNException
            );

            // State of the translator should be unchanged.
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Non-unique External IDs") {
            auto external_mismatch_ids = std::vector<uint64_t>{10, 12, 10};
            auto internal_mismatch_ids = std::vector<uint32_t>{50, 60, 70};

            CATCH_REQUIRE_THROWS_AS(
                translator.insert(external_mismatch_ids, internal_mismatch_ids),
                svs::ANNException
            );

            // State of the translator should be unchanged.
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Repeat Internal IDs") {
            auto external_mismatch_ids = std::vector<uint64_t>{10, 12, 14};
            auto internal_mismatch_ids = std::vector<uint32_t>{50, 10, 70};

            CATCH_REQUIRE_THROWS_AS(
                translator.insert(external_mismatch_ids, internal_mismatch_ids),
                svs::ANNException
            );

            // State of the translator should be unchanged.
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Repeat Internal IDs") {
            auto external_mismatch_ids = std::vector<uint64_t>{10, 12, 14};
            auto internal_mismatch_ids = std::vector<uint32_t>{50, 60, 60};

            CATCH_REQUIRE_THROWS_AS(
                translator.insert(external_mismatch_ids, internal_mismatch_ids),
                svs::ANNException
            );

            // State of the translator should be unchanged.
            check(translator, external_ids, internal_ids);
        }

        ///
        /// Deletion
        ///

        CATCH_SECTION("Delete External") {
            auto external_id_delete = std::vector<uint64_t>{2, 6};
            translator.delete_external(external_id_delete);

            external_ids = {0, 4, 8};
            internal_ids = {0, 20, 40};
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Delete External Error") {
            // The external id `10` doesn't exist.
            // We should get an error without modifying the underlying container.
            auto external_id_delete = std::vector<uint64_t>{2, 10};
            CATCH_REQUIRE_THROWS_AS(
                translator.delete_external(external_id_delete), svs::ANNException
            );
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Delete Internal") {
            auto internal_id_delete = std::vector<uint64_t>{0, 20, 40};
            translator.delete_internal(internal_id_delete);

            external_ids = {2, 6};
            internal_ids = {10, 30};
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Delete Internal Error") {
            auto internal_id_delete = std::vector<uint64_t>{0, 20, 2};
            CATCH_REQUIRE_THROWS_AS(
                translator.delete_internal(internal_id_delete), svs::ANNException
            );
            check(translator, external_ids, internal_ids);
        }

        ///
        /// Renaming (relabeling)
        ///

        CATCH_SECTION("Rename External") {
            translator.remap_external_id(2, 100);

            CATCH_REQUIRE(!translator.has_external(2));
            CATCH_REQUIRE(translator.has_external(100));
            CATCH_REQUIRE(translator.get_internal(100) == 10);
            CATCH_REQUIRE(translator.get_external(10) == 100);
            CATCH_REQUIRE(translator.size() == external_ids.size());

            // Everything else is unaffected.
            CATCH_REQUIRE(translator.get_internal(0) == 0);
            CATCH_REQUIRE(translator.get_internal(4) == 20);
            CATCH_REQUIRE(translator.get_internal(6) == 30);
            CATCH_REQUIRE(translator.get_internal(8) == 40);
        }

        CATCH_SECTION("Rename External Error - Source Does Not Exist") {
            // External id `10` doesn't exist yet.
            CATCH_REQUIRE_THROWS_AS(
                translator.remap_external_id(10, 100), svs::ANNException
            );
            // State of the translator should be unchanged.
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Rename External Error - Destination Already Exists") {
            // External id `4` already exists.
            CATCH_REQUIRE_THROWS_AS(translator.remap_external_id(2, 4), svs::ANNException);
            // State of the translator should be unchanged.
            check(translator, external_ids, internal_ids);
        }

        ///
        /// Replacing stale mappings
        ///

        CATCH_SECTION("replace_stale_and_insert is transactional on a live duplicate") {
            // Batch: a fresh external id followed by a live (non-stale) duplicate.
            // The whole batch must be validated before any mutation, so the throw
            // leaves the translator unchanged -- in particular the fresh id `100` must
            // NOT have been committed.
            auto add_external = std::vector<uint64_t>{100, 8}; // 8 already exists
            auto add_internal = std::vector<uint32_t>{50, 60};
            auto never_stale = [](uint32_t) { return false; };

            CATCH_REQUIRE_THROWS_AS(
                translator.replace_stale_and_insert(
                    add_external, add_internal, never_stale
                ),
                svs::ANNException
            );

            CATCH_REQUIRE_FALSE(translator.has_external(100)); // not committed
            CATCH_REQUIRE_FALSE(translator.has_internal(50));
            check(translator, external_ids, internal_ids); // unchanged
        }

        CATCH_SECTION("replace_stale_and_insert replaces a stale mapping") {
            // Mark internal 20 (external 4) stale; re-add external 4 at a new
            // internal id plus a genuinely fresh id. Both should commit.
            auto add_external = std::vector<uint64_t>{4, 100};
            auto add_internal = std::vector<uint32_t>{50, 60};
            auto stale_if_20 = [](uint32_t i) { return i == 20; };

            CATCH_REQUIRE_NOTHROW(
                translator.replace_stale_and_insert(add_external, add_internal, stale_if_20)
            );

            CATCH_REQUIRE(translator.size() == external_ids.size() + 1);
            CATCH_REQUIRE(translator.get_internal(4) == 50);
            CATCH_REQUIRE(translator.get_internal(100) == 60);
            CATCH_REQUIRE(translator.get_external(50) == 4);
            CATCH_REQUIRE(translator.get_external(60) == 100);
            // Old internal 20 no longer maps back to any external.
            CATCH_REQUIRE_FALSE(translator.has_internal(20));
        }

        ///
        /// Unmapped IDs
        ///

        CATCH_SECTION("Unmapped Internal IDs") {
            // `5` is a hole inside the internal-to-external table, `1000` lies beyond its
            // end, and `10` becomes a hole once its translation is deleted.
            translator.delete_internal(std::vector<uint32_t>{10});
            for (uint32_t unmapped : {5u, 1000u, 10u}) {
                CATCH_REQUIRE_FALSE(translator.has_internal(unmapped));
                CATCH_REQUIRE_THROWS_AS(
                    translator.get_external(unmapped), std::out_of_range
                );
                // Regression: a hole inside the table used to yield
                // `invalid_external_id` instead of the caller's default.
                CATCH_REQUIRE(translator.get_external_or(unmapped, 77) == 77);
            }
            CATCH_REQUIRE(translator.get_external_or(20, 77) == 4);
        }

        CATCH_SECTION("Unmapped External IDs") {
            translator.delete_external(std::vector<uint64_t>{2});
            for (uint64_t unmapped : {3u, 2u}) {
                CATCH_REQUIRE_FALSE(translator.has_external(unmapped));
                CATCH_REQUIRE_THROWS_AS(
                    translator.get_internal(unmapped), std::out_of_range
                );
                CATCH_REQUIRE(translator.get_internal_or(unmapped, 77) == 77);
            }
            CATCH_REQUIRE(translator.get_internal_or(4, 77) == 20);
        }

        CATCH_SECTION("Reserved External ID") {
            // The reserved ID marks an unmapped slot, so storing it would leave the two
            // directions inconsistent. Every way in rejects it without side effects.
            const external_id_type reserved = Translator::invalid_external_id;
            auto add_external = std::vector<uint64_t>{10, reserved};
            auto add_internal = std::vector<uint32_t>{50, 60};
            auto never_stale = [](uint32_t) { return false; };

            CATCH_REQUIRE_THROWS_AS(
                translator.insert(add_external, add_internal), svs::ANNException
            );
            CATCH_REQUIRE_THROWS_AS(
                translator.replace_stale_and_insert(
                    add_external, add_internal, never_stale
                ),
                svs::ANNException
            );
            CATCH_REQUIRE_THROWS_AS(
                translator.insert_translation(reserved, 60), svs::ANNException
            );
            CATCH_REQUIRE_THROWS_AS(
                translator.remap_external_id(2, reserved), svs::ANNException
            );

            CATCH_REQUIRE_FALSE(translator.has_external(reserved));
            CATCH_REQUIRE_FALSE(translator.has_external(10));
            CATCH_REQUIRE_FALSE(translator.has_internal(50));
            CATCH_REQUIRE_FALSE(translator.has_internal(60));
            check(translator, external_ids, internal_ids);
        }

        ///
        /// Moving and reusing internal IDs
        ///

        CATCH_SECTION("Remap Internal") {
            // Into a hole inside the table, then beyond its end.
            translator.remap_internal_id(10, 5);
            translator.remap_internal_id(40, 1000);

            CATCH_REQUIRE_FALSE(translator.has_internal(10));
            CATCH_REQUIRE_FALSE(translator.has_internal(40));
            internal_ids = {0, 5, 20, 30, 1000};
            check(translator, external_ids, internal_ids);
        }

        CATCH_SECTION("Reusing a deleted internal ID") {
            translator.delete_external(std::vector<uint64_t>{4});
            CATCH_REQUIRE_FALSE(translator.has_internal(20));
            translator.insert(std::vector<uint64_t>{100}, std::vector<uint32_t>{20});

            CATCH_REQUIRE(translator.size() == external_ids.size());
            CATCH_REQUIRE_FALSE(translator.has_external(4));
            CATCH_REQUIRE(translator.get_internal(100) == 20);
            CATCH_REQUIRE(translator.get_external(20) == 100);
        }

        ///
        /// Memory accounting
        ///

        CATCH_SECTION("Memory Usage") {
            const size_t pair_bytes = sizeof(external_id_type) + sizeof(internal_id_type);
            CATCH_REQUIRE(
                Translator::estimate_memory_usage(10) ==
                10 * pair_bytes + 10 * sizeof(external_id_type)
            );

            // The internal-to-external table spans slots [0, 40] however few of them are
            // mapped, and deleting translations does not release it.
            const size_t num_slots = 41;
            auto min_usage = [&](size_t num_translations) {
                return num_translations * pair_bytes + num_slots * sizeof(external_id_type);
            };
            CATCH_REQUIRE(translator.get_memory_usage() >= min_usage(5));
            if constexpr (std::is_same_v<Translator, svs::ConcurrentIDTranslator>) {
                // The segmented table rounds its capacity up to its bucket layout.
                const size_t table_slots =
                    svs::lib::SegmentedVector<external_id_type>(num_slots).capacity();
                CATCH_REQUIRE(
                    translator.get_memory_usage() ==
                    5 * pair_bytes + table_slots * sizeof(external_id_type)
                );
            }

            translator.delete_internal(std::vector<uint32_t>{0, 20, 40});
            CATCH_REQUIRE(translator.get_memory_usage() >= min_usage(2));

            // The estimate is a lower bound for a dense translator.
            auto dense = Translator(typename Translator::Identity(100));
            CATCH_REQUIRE(
                dense.get_memory_usage() >= Translator::estimate_memory_usage(100)
            );
        }

        ///
        /// Converting between the variants
        ///

        CATCH_SECTION("Converting between variants") {
            using Other = other_variant<Translator>;
            // Leave a hole so the conversion has to reproduce an unmapped slot.
            translator.delete_external(std::vector<uint64_t>{4});
            external_ids = {0, 2, 6, 8};
            internal_ids = {0, 10, 30, 40};

            auto converted = Other(translator);
            check(converted, external_ids, internal_ids);
            CATCH_REQUIRE_FALSE(converted.has_internal(20));
            CATCH_REQUIRE(converted.get_external_or(20, 77) == 77);

            // The converted translator owns its tables.
            converted.insert(std::vector<uint64_t>{4}, std::vector<uint32_t>{20});
            CATCH_REQUIRE_FALSE(translator.has_external(4));
            CATCH_REQUIRE_FALSE(translator.has_internal(20));
            check(translator, external_ids, internal_ids);

            auto round_trip = Translator(std::move(converted));
            external_ids.push_back(4);
            internal_ids.push_back(20);
            check_translation(round_trip, external_ids, internal_ids);
        }

        ///
        /// Saving and loading
        ///

        CATCH_SECTION("Saving and Reloading") {
            svs_test::prepare_temp_directory();
            auto tempdir = svs_test::temp_directory();

            svs::lib::save_to_disk(translator, tempdir);
            auto reloaded = svs::lib::load_from_disk<Translator>(tempdir);

            check(translator, external_ids, internal_ids);
            check(reloaded, external_ids, internal_ids);
        }

        CATCH_SECTION("Saving and Reloading after mutation") {
            // Holes at internal 20 and 40, a renamed external ID and a moved internal ID.
            translator.delete_external(std::vector<uint64_t>{4});
            translator.remap_external_id(2, 100);
            translator.remap_internal_id(40, 7);

            svs_test::prepare_temp_directory();
            auto tempdir = svs_test::temp_directory();
            svs::lib::save_to_disk(translator, tempdir);

            // Both variants share one on-disk format: the concurrent index saves the
            // concurrent variant and the shared loader reads the plain one.
            auto check_reloaded = [&](const auto& reloaded) {
                CATCH_REQUIRE(reloaded.size() == 4);
                check_same(translator, reloaded);
                CATCH_REQUIRE_FALSE(reloaded.has_external(4));
                CATCH_REQUIRE_FALSE(reloaded.has_external(2));
                CATCH_REQUIRE_FALSE(reloaded.has_internal(20));
                CATCH_REQUIRE_FALSE(reloaded.has_internal(40));
                CATCH_REQUIRE(reloaded.get_internal(100) == 10);
                CATCH_REQUIRE(reloaded.get_external(7) == 8);
            };
            check_reloaded(svs::lib::load_from_disk<Translator>(tempdir));
            check_reloaded(svs::lib::load_from_disk<other_variant<Translator>>(tempdir));
        }
    }
}
