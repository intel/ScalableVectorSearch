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
#include "svs/core/data/vector_set_view.h"

// other misc SVS stuff
#include "svs/concepts/vector_set.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <array>
#include <cstdint>
#include <vector>

CATCH_TEST_CASE("Vector Set View", "[core][data][vector_set_view]") {
    // 3 sets with cardinalities 2, 0, 3 over dimension 2.
    auto tokens =
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    auto offsets = std::vector<uint64_t>{0, 2, 2, 5};

    CATCH_SECTION("Construction and contents") {
        auto view = svs::data::ConstVectorSetView<float>(tokens, offsets, 2);
        CATCH_REQUIRE(view.size() == 3);
        CATCH_REQUIRE(view.dimensions() == 2);
        CATCH_REQUIRE(view.total_tokens() == 5);

        // size(), cardinality() and get_datum() against the ground truth, one pass.
        std::vector<std::vector<std::array<float, 2>>> expected{
            {{1.0f, 2.0f}, {3.0f, 4.0f}}, {}, {{5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}}};
        for (size_t i = 0; i < expected.size(); ++i) {
            auto set = view.get_datum(i);
            CATCH_REQUIRE(set.cardinality() == expected[i].size());
            CATCH_REQUIRE(set.dimensions() == 2);
            CATCH_REQUIRE(set.empty() == expected[i].empty());
            for (size_t j = 0; j < expected[i].size(); ++j) {
                auto t = set.token(j);
                CATCH_REQUIRE(t[0] == expected[i][j][0]);
                CATCH_REQUIRE(t[1] == expected[i][j][1]);
            }
            view.prefetch(i); // exercised for the empty set (i==1) too
        }
    }

    CATCH_SECTION("Validation failures: one per throw in validate_vector_set_layout") {
        CATCH_REQUIRE_THROWS_AS(
            (svs::data::ConstVectorSetView<float>(
                std::span<const float>(), std::span<const uint64_t>(), 2
            )),
            svs::ANNException
        ); // offsets.empty()
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(tokens, std::vector<uint64_t>{1, 2}, 2),
            svs::ANNException
        ); // offsets.front() != 0
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(tokens, std::vector<uint64_t>{0, 2, 1}, 2),
            svs::ANNException
        ); // offsets not non-decreasing
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(tokens, std::vector<uint64_t>{0, 1}, 2),
            svs::ANNException
        ); // offsets.back() * dimensions != token_elements
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(
                std::span<const float>(), std::vector<uint64_t>{0, uint64_t{1} << 63}, 2
            ),
            svs::ANNException
        ); // offsets.back() * dimensions overflows size_t
    }

    CATCH_SECTION("Static extent") {
        auto static_tokens = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
        auto static_offsets = std::vector<uint64_t>{0, 1, 2};

        auto view =
            svs::data::ConstVectorSetView<float, 2>(static_tokens, static_offsets, 2);
        CATCH_REQUIRE(view.dimensions() == 2);
        CATCH_REQUIRE(view.get_datum(0).token(0)[0] == 1.0f);

        CATCH_REQUIRE_THROWS_AS(
            (svs::data::ConstVectorSetView<float, 3>(static_tokens, static_offsets, 2)),
            svs::ANNException
        );
    }

    CATCH_SECTION("Non-float element type, concept checks") {
        auto byte_tokens = std::vector<uint8_t>{1, 2, 3, 4, 5, 6};
        auto byte_offsets = std::vector<uint64_t>{0, 1, 3};
        auto view = svs::data::ConstVectorSetView<uint8_t>(byte_tokens, byte_offsets, 2);

        CATCH_REQUIRE(view.size() == 2);
        CATCH_REQUIRE(view.cardinality(0) == 1);
        CATCH_REQUIRE(view.cardinality(1) == 2);
        CATCH_REQUIRE(view.get_datum(1).token(1)[1] == 6);

        static_assert(svs::data::VectorSet<decltype(view)::const_value_type>);
        static_assert(svs::data::ImmutableVectorSetDataset<decltype(view)>);
    }
}
