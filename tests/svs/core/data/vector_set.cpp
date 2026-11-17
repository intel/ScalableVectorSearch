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
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
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

// Every document's tokens, element-wise, must match between two equivalent datasets.
template <typename A, typename B> void require_same_contents(const A& a, const B& b) {
    CATCH_REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        auto da = a.get_datum(i);
        auto db = b.get_datum(i);
        CATCH_REQUIRE(da.cardinality() == db.cardinality());
        for (size_t j = 0; j < da.cardinality(); ++j) {
            auto ta = da.token(j);
            auto tb = db.token(j);
            CATCH_REQUIRE(std::equal(ta.begin(), ta.end(), tb.begin()));
        }
    }
}
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
        auto offsets = container.offsets();
        CATCH_REQUIRE(offsets.size() == 4);
        CATCH_REQUIRE(offsets[3] == 5);

        // cardinality() and get_datum() against the ground truth, one pass.
        std::vector<std::vector<std::array<float, 2>>> expected{
            {{1.0f, 2.0f}, {3.0f, 4.0f}}, {}, {{5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}}};
        for (size_t i = 0; i < expected.size(); ++i) {
            auto set = container.get_datum(i);
            CATCH_REQUIRE(set.cardinality() == expected[i].size());
            CATCH_REQUIRE(set.empty() == expected[i].empty());
            for (size_t j = 0; j < expected[i].size(); ++j) {
                auto t = set.token(j);
                CATCH_REQUIRE(t[0] == expected[i][j][0]);
                CATCH_REQUIRE(t[1] == expected[i][j][1]);
            }
            container.prefetch(i); // exercised for the empty document (i==1) too
        }
    }

    CATCH_SECTION("view() matches the container element-wise") {
        auto container = svs::data::VectorSetData<float>(make_tokens(), make_offsets());
        auto view = container.view();
        CATCH_REQUIRE(view.dimensions() == container.dimensions());
        require_same_contents(container, view);
    }

    CATCH_SECTION("copy_from takes ownership of a copy of a borrowed view") {
        auto tokens = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        auto offsets = std::vector<uint64_t>{0, 1, 3};
        auto view = svs::data::ConstVectorSetView<float>(tokens, offsets, 2);

        auto container = svs::data::VectorSetData<float>::copy_from(view);
        CATCH_REQUIRE(container.tokens().data() != tokens.data());
        CATCH_REQUIRE(container.cardinality(0) == 1);
        CATCH_REQUIRE(container.cardinality(1) == 2);
        CATCH_REQUIRE(container.get_datum(1).token(1)[0] == 5.0f);
        require_same_contents(container, view);
    }

    CATCH_SECTION("Constructor delegates to the shared offsets validator") {
        // The constructor adds no validation beyond calling validate_vector_set_layout,
        // whose branches are already covered against ConstVectorSetView in
        // vector_set_view.cpp.
        CATCH_REQUIRE_THROWS_AS(
            svs::data::VectorSetData<float>(
                svs::data::SimpleData<float>(2, 2), std::vector<uint64_t>{}
            ),
            svs::ANNException
        );
    }

    CATCH_SECTION("Static extent, integer element type") {
        auto tokens = svs::data::SimpleData<uint32_t, 1>(3, 1);
        tokens.set_datum(0, std::vector<uint32_t>{7});
        tokens.set_datum(1, std::vector<uint32_t>{8});
        tokens.set_datum(2, std::vector<uint32_t>{9});

        auto container = svs::data::VectorSetData<uint32_t, 1>(
            std::move(tokens), std::vector<uint64_t>{0, 2, 3}
        );
        CATCH_REQUIRE(container.dimensions() == 1);
        CATCH_REQUIRE(container.get_datum(0).token(1)[0] == 8);
        static_assert(svs::data::ImmutableVectorSetDataset<decltype(container)>);
    }

    CATCH_SECTION("Float16 element type") {
        auto tokens = svs::data::SimpleData<svs::Float16>(2, 3);
        tokens.set_datum(0, std::vector<float>{1.0f, 2.0f, 3.0f});
        tokens.set_datum(1, std::vector<float>{4.0f, 5.0f, 6.0f});

        auto container = svs::data::VectorSetData<svs::Float16>(
            std::move(tokens), std::vector<uint64_t>{0, 1, 2}
        );
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

    CATCH_SECTION("Save/load") {
        auto temp_directory = svs_test::prepare_temp_directory_v2();
        auto container = svs::data::VectorSetData<float>(make_tokens(), make_offsets());
        svs::lib::save_to_disk(container, temp_directory);

        CATCH_SECTION("Round trip") {
            auto reloaded =
                svs::lib::load_from_disk<svs::data::VectorSetData<float>>(temp_directory);
            CATCH_REQUIRE(reloaded.dimensions() == container.dimensions());
            CATCH_REQUIRE(reloaded.total_tokens() == container.total_tokens());
            auto a = container.offsets();
            auto b = reloaded.offsets();
            CATCH_REQUIRE(std::equal(a.begin(), a.end(), b.begin()));
            require_same_contents(container, reloaded);
        }

        CATCH_SECTION("A corrupted offsets file throws") {
            bool found_offsets_file = false;
            for (auto const& entry : std::filesystem::directory_iterator(temp_directory)) {
                if (entry.path().filename().string().starts_with("vector_set_offsets")) {
                    found_offsets_file = true;
                    // Truncate so its size no longer matches the saved metadata.
                    std::filesystem::resize_file(entry.path(), 1);
                }
            }
            CATCH_REQUIRE(found_offsets_file);
            CATCH_REQUIRE_THROWS_AS(
                svs::lib::load_from_disk<svs::data::VectorSetData<float>>(temp_directory),
                svs::ANNException
            );
        }
    }

    CATCH_SECTION("Loading with the wrong schema throws") {
        auto temp_directory = svs_test::prepare_temp_directory_v2();
        // SimpleData has a different, incompatible schema ("uncompressed_data").
        svs::lib::save_to_disk(svs::data::SimpleData<float>(2, 2), temp_directory);
        CATCH_REQUIRE_THROWS(
            svs::lib::load_from_disk<svs::data::VectorSetData<float>>(temp_directory)
        );
    }
}
