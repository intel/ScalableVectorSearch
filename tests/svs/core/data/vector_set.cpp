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
#include "svs/core/data/vector_set.h"

// svs
#include "svs/concepts/vector_set.h"
#include "svs/core/allocator.h"
#include "svs/core/data/simple.h"
#include "svs/core/data/vector_set_view.h"
#include "svs/lib/float16.h"
#include "svs/lib/saveload.h"

// tests
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace {

// 3 documents with cardinalities 2, 0, 3 over dimension 2, matching the fixture in
// vector_set_view.cpp so both headers are checked against the same layout.
svs::data::SimpleData<float> make_tokens() {
    auto tokens = svs::data::SimpleData<float>(5, 2);
    float values[5][2] = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}};
    for (size_t i = 0; i < 5; ++i) {
        tokens.set_datum(i, std::span<const float>(values[i], 2));
    }
    return tokens;
}

std::vector<uint64_t> make_offsets() { return {0, 2, 2, 5}; }

} // namespace

CATCH_TEST_CASE("Vector Set Data", "[core][data][vector_set]") {
    CATCH_SECTION("Construction by move is a no-copy takeover") {
        auto tokens = make_tokens();
        const float* original_ptr = tokens.data();

        auto container = svs::data::VectorSetData<float>(std::move(tokens), make_offsets());
        CATCH_REQUIRE(container.tokens().data() == original_ptr);

        // The moved-from source is left in a state that is safe to inspect (even if its
        // contents are unspecified); this should not throw or crash.
        CATCH_REQUIRE_NOTHROW(tokens.get_allocator());
    }

    CATCH_SECTION("Accessors") {
        auto container = svs::data::VectorSetData<float>(make_tokens(), make_offsets());

        CATCH_REQUIRE(container.size() == 3);
        CATCH_REQUIRE(container.dimensions() == 2);
        CATCH_REQUIRE(container.total_tokens() == 5);
        CATCH_REQUIRE(container.cardinality(0) == 2);
        CATCH_REQUIRE(container.cardinality(1) == 0);
        CATCH_REQUIRE(container.cardinality(2) == 3);

        auto set0 = container.get_datum(0);
        CATCH_REQUIRE(set0.cardinality() == 2);
        CATCH_REQUIRE(set0.token(0)[0] == 1.0f);
        CATCH_REQUIRE(set0.token(1)[1] == 4.0f);

        auto set1 = container.get_datum(1);
        CATCH_REQUIRE(set1.empty());

        auto set2 = container.get_datum(2);
        CATCH_REQUIRE(set2.cardinality() == 3);
        CATCH_REQUIRE(set2.token(2)[0] == 9.0f);
        CATCH_REQUIRE(set2.token(2)[1] == 10.0f);

        container.prefetch(0);
        container.prefetch(1);
        container.prefetch(2);

        auto offsets = container.offsets();
        CATCH_REQUIRE(offsets.size() == 4);
        CATCH_REQUIRE(offsets[3] == 5);
    }

    CATCH_SECTION("view() matches the container element-wise") {
        auto container = svs::data::VectorSetData<float>(make_tokens(), make_offsets());
        auto view = container.view();

        CATCH_REQUIRE(view.size() == container.size());
        CATCH_REQUIRE(view.dimensions() == container.dimensions());
        for (size_t i = 0; i < container.size(); ++i) {
            auto a = container.get_datum(i);
            auto b = view.get_datum(i);
            CATCH_REQUIRE(a.cardinality() == b.cardinality());
            for (size_t j = 0; j < a.cardinality(); ++j) {
                auto ta = a.token(j);
                auto tb = b.token(j);
                CATCH_REQUIRE(std::equal(ta.begin(), ta.end(), tb.begin()));
            }
        }
    }

    CATCH_SECTION("copy_from takes ownership of a copy of a borrowed view") {
        auto tokens = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        auto offsets = std::vector<uint64_t>{0, 1, 3};
        auto view = svs::data::ConstVectorSetView<float>(tokens, offsets, 2);

        auto container = svs::data::VectorSetData<float>::copy_from(view);
        CATCH_REQUIRE(container.tokens().data() != tokens.data());
        CATCH_REQUIRE(container.size() == 2);
        CATCH_REQUIRE(container.cardinality(0) == 1);
        CATCH_REQUIRE(container.cardinality(1) == 2);
        CATCH_REQUIRE(container.get_datum(1).token(1)[0] == 5.0f);
    }

    CATCH_SECTION("Validation failures") {
        CATCH_REQUIRE_THROWS_AS(
            svs::data::VectorSetData<float>(
                svs::data::SimpleData<float>(2, 2), std::vector<uint64_t>{}
            ),
            svs::ANNException
        );

        CATCH_REQUIRE_THROWS_AS(
            svs::data::VectorSetData<float>(
                svs::data::SimpleData<float>(2, 2), std::vector<uint64_t>{1, 2}
            ),
            svs::ANNException
        );

        CATCH_REQUIRE_THROWS_AS(
            svs::data::VectorSetData<float>(
                svs::data::SimpleData<float>(2, 2), std::vector<uint64_t>{0, 2, 1}
            ),
            svs::ANNException
        );

        CATCH_REQUIRE_THROWS_AS(
            svs::data::VectorSetData<float>(
                svs::data::SimpleData<float>(2, 2), std::vector<uint64_t>{0, 1}
            ),
            svs::ANNException
        );
    }

    CATCH_SECTION("Static extent, integer element type") {
        auto tokens = svs::data::SimpleData<uint32_t, 1>(3, 1);
        tokens.set_datum(0, std::vector<uint32_t>{7});
        tokens.set_datum(1, std::vector<uint32_t>{8});
        tokens.set_datum(2, std::vector<uint32_t>{9});
        auto offsets = std::vector<uint64_t>{0, 2, 3};

        auto container =
            svs::data::VectorSetData<uint32_t, 1>(std::move(tokens), std::move(offsets));
        CATCH_REQUIRE(container.dimensions() == 1);
        CATCH_REQUIRE(container.get_datum(0).token(1)[0] == 8);

        static_assert(svs::data::ImmutableVectorSetDataset<decltype(container)>);
    }

    CATCH_SECTION("Float16 element type") {
        auto tokens = svs::data::SimpleData<svs::Float16>(2, 3);
        tokens.set_datum(0, std::vector<float>{1.0f, 2.0f, 3.0f});
        tokens.set_datum(1, std::vector<float>{4.0f, 5.0f, 6.0f});
        auto offsets = std::vector<uint64_t>{0, 1, 2};

        auto container =
            svs::data::VectorSetData<svs::Float16>(std::move(tokens), std::move(offsets));
        CATCH_REQUIRE(container.size() == 2);
        CATCH_REQUIRE(float(container.get_datum(1).token(0)[0]) == 4.0f);
    }

    CATCH_SECTION("Blocked allocator is rejected") {
        // `VectorSetData` static_asserts against a Blocked allocator (its per-document
        // tokens would no longer be contiguous), so this checks the gating condition
        // without instantiating `VectorSetData<float, Dynamic, Blocked<...>>`, which
        // would be a hard compile error rather than a test failure.
        using BlockedAlloc = svs::data::Blocked<svs::lib::Allocator<float>>;
        static_assert(svs::data::is_blocked_v<BlockedAlloc>);
    }

    CATCH_SECTION("Save then load round trip") {
        auto temp_directory = svs_test::prepare_temp_directory_v2();
        auto container = svs::data::VectorSetData<float>(make_tokens(), make_offsets());

        svs::lib::save_to_disk(container, temp_directory);
        auto reloaded =
            svs::lib::load_from_disk<svs::data::VectorSetData<float>>(temp_directory);

        CATCH_REQUIRE(reloaded.size() == container.size());
        CATCH_REQUIRE(reloaded.dimensions() == container.dimensions());
        CATCH_REQUIRE(reloaded.total_tokens() == container.total_tokens());
        auto reloaded_offsets = reloaded.offsets();
        auto original_offsets = container.offsets();
        CATCH_REQUIRE(std::equal(
            reloaded_offsets.begin(), reloaded_offsets.end(), original_offsets.begin()
        ));
        for (size_t i = 0; i < container.size(); ++i) {
            auto a = container.get_datum(i);
            auto b = reloaded.get_datum(i);
            CATCH_REQUIRE(a.cardinality() == b.cardinality());
            for (size_t j = 0; j < a.cardinality(); ++j) {
                auto ta = a.token(j);
                auto tb = b.token(j);
                CATCH_REQUIRE(std::equal(ta.begin(), ta.end(), tb.begin()));
            }
        }
    }

    CATCH_SECTION("Loading a corrupted offsets file throws") {
        auto temp_directory = svs_test::prepare_temp_directory_v2();
        auto container = svs::data::VectorSetData<float>(make_tokens(), make_offsets());
        svs::lib::save_to_disk(container, temp_directory);

        bool found_offsets_file = false;
        for (auto const& entry : std::filesystem::directory_iterator(temp_directory)) {
            if (entry.path().filename().string().starts_with("vector_set_offsets")) {
                found_offsets_file = true;
                // Truncate the file so its size no longer matches the saved metadata.
                std::filesystem::resize_file(entry.path(), 1);
            }
        }
        CATCH_REQUIRE(found_offsets_file);

        CATCH_REQUIRE_THROWS_AS(
            svs::lib::load_from_disk<svs::data::VectorSetData<float>>(temp_directory),
            svs::ANNException
        );
    }

    CATCH_SECTION("Loading with the wrong schema throws") {
        auto temp_directory = svs_test::prepare_temp_directory_v2();
        // SimpleData has a different, incompatible schema ("uncompressed_data").
        auto other = svs::data::SimpleData<float>(2, 2);
        svs::lib::save_to_disk(other, temp_directory);

        CATCH_REQUIRE_THROWS(
            svs::lib::load_from_disk<svs::data::VectorSetData<float>>(temp_directory)
        );
    }
}
