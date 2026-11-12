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
#include <cstdint>
#include <vector>

CATCH_TEST_CASE("Vector Set View", "[core][data][vector_set_view]") {
    CATCH_SECTION("Construction and contents") {
        // 3 sets with cardinalities 2, 0, 3 over dimension 2.
        auto tokens = std::vector<float>{
            1.0f,
            2.0f, // set 0, token 0
            3.0f,
            4.0f, // set 0, token 1
            5.0f,
            6.0f, // set 2, token 0
            7.0f,
            8.0f, // set 2, token 1
            9.0f,
            10.0f // set 2, token 2
        };
        auto offsets = std::vector<uint64_t>{0, 2, 2, 5};
        auto view = svs::data::ConstVectorSetView<float>(tokens, offsets, 2);

        CATCH_REQUIRE(view.size() == 3);
        CATCH_REQUIRE(view.dimensions() == 2);
        CATCH_REQUIRE(view.total_tokens() == 5);

        CATCH_REQUIRE(view.cardinality(0) == 2);
        CATCH_REQUIRE(view.cardinality(1) == 0);
        CATCH_REQUIRE(view.cardinality(2) == 3);

        auto set0 = view.get_datum(0);
        CATCH_REQUIRE(set0.cardinality() == 2);
        CATCH_REQUIRE(set0.dimensions() == 2);
        CATCH_REQUIRE(!set0.empty());
        CATCH_REQUIRE(set0.token(0)[0] == 1.0f);
        CATCH_REQUIRE(set0.token(0)[1] == 2.0f);
        CATCH_REQUIRE(set0.token(1)[0] == 3.0f);
        CATCH_REQUIRE(set0.token(1)[1] == 4.0f);

        auto set1 = view.get_datum(1);
        CATCH_REQUIRE(set1.cardinality() == 0);
        CATCH_REQUIRE(set1.empty());

        auto set2 = view.get_datum(2);
        CATCH_REQUIRE(set2.cardinality() == 3);
        CATCH_REQUIRE(set2.token(2)[0] == 9.0f);
        CATCH_REQUIRE(set2.token(2)[1] == 10.0f);

        view.prefetch(0);
        view.prefetch(1);
        view.prefetch(2);
    }

    CATCH_SECTION("Validation failures") {
        auto tokens = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};

        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(
                std::span<const float>(tokens), std::span<const uint64_t>(), 2
            ),
            svs::ANNException
        );

        auto bad_front = std::vector<uint64_t>{1, 2};
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(tokens, bad_front, 2), svs::ANNException
        );

        auto decreasing = std::vector<uint64_t>{0, 2, 1};
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(tokens, decreasing, 2), svs::ANNException
        );

        auto mismatched = std::vector<uint64_t>{0, 1};
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(tokens, mismatched, 2), svs::ANNException
        );

        auto overflowing = std::vector<uint64_t>{0, uint64_t{1} << 63};
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstVectorSetView<float>(std::span<const float>(), overflowing, 2),
            svs::ANNException
        );
    }

    CATCH_SECTION("Static extent") {
        auto tokens = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
        auto offsets = std::vector<uint64_t>{0, 1, 2};

        auto view = svs::data::ConstVectorSetView<float, 2>(tokens, offsets, 2);
        CATCH_REQUIRE(view.dimensions() == 2);
        CATCH_REQUIRE(view.get_datum(0).token(0)[0] == 1.0f);

        CATCH_REQUIRE_THROWS_AS(
            (svs::data::ConstVectorSetView<float, 3>(tokens, offsets, 2)), svs::ANNException
        );
    }

    CATCH_SECTION("Non-float element type") {
        auto tokens = std::vector<uint8_t>{1, 2, 3, 4, 5, 6};
        auto offsets = std::vector<uint64_t>{0, 1, 3};
        auto view = svs::data::ConstVectorSetView<uint8_t>(tokens, offsets, 2);

        CATCH_REQUIRE(view.size() == 2);
        CATCH_REQUIRE(view.cardinality(0) == 1);
        CATCH_REQUIRE(view.cardinality(1) == 2);
        CATCH_REQUIRE(view.get_datum(1).token(1)[1] == 6);

        static_assert(svs::data::VectorSet<decltype(view)::const_value_type>);
        static_assert(svs::data::ImmutableVectorSetDataset<decltype(view)>);
    }
}
