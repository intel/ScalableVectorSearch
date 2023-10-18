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

// header under test.
#include "svs/core/translation.h"

// misc utilties
#include "svs/lib/narrow.h"

// Test utils
#include "tests/utils/generators.h"

// Catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <iterator>

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

template <typename External, typename Internal>
void check_contents(
    const svs::IDTranslator& translator, const External& external, const Internal& internal
) {
    check_contents(translator.begin(), translator.end(), external, internal);
}

template <typename External, typename Internal>
void check_translation(
    const svs::IDTranslator& translator, const External& external, const Internal& internal
) {
    CATCH_REQUIRE(translator.size() == external.size());
    CATCH_REQUIRE(translator.size() == internal.size());

    for (size_t i = 0, imax = translator.size(); i < imax; ++i) {
        CATCH_REQUIRE(translator.get_internal(external[i]) == internal[i]);
        CATCH_REQUIRE(translator.get_external(internal[i]) == external[i]);
    }
}

void check(
    const svs::IDTranslator& translator,
    std::vector<uint64_t>& external_ids,
    std::vector<uint32_t>& internal_ids
) {
    check_translation(translator, external_ids, internal_ids);
    check_contents(translator, external_ids, internal_ids);
}

} // namespace

CATCH_TEST_CASE("Translation Table", "[core][translation]") {
    CATCH_SECTION("Identity Constructor") {
        auto translator = svs::IDTranslator(svs::IDTranslator::Identity(10));
        CATCH_REQUIRE(translator.size() == 10);
        for (size_t i = 0; i < 10; ++i) {
            CATCH_REQUIRE(translator.has_external(i));
            CATCH_REQUIRE(translator.has_internal(i));
            CATCH_REQUIRE(translator.get_external(i) == i);
            CATCH_REQUIRE(translator.get_internal(i) == i);
        }
    }

    CATCH_SECTION("Basic Tests") {
        auto translator = svs::IDTranslator();
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
        /// Saving and loading
        ///

        CATCH_SECTION("Saving and Reloading") {
            svs_test::prepare_temp_directory();
            auto tempdir = svs_test::temp_directory();

            svs::lib::save_to_disk(translator, tempdir);
            auto reloaded = svs::lib::load_from_disk<svs::IDTranslator>(tempdir);

            check(translator, external_ids, internal_ids);
            check(reloaded, external_ids, internal_ids);
        }
    }
}
