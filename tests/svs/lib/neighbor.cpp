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

// stdlib
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <type_traits>
#include <unordered_set>

// third-party
#include "tsl/robin_set.h"

// svs
#include "svs/lib/neighbor.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/generators.h"

CATCH_TEST_CASE("Testing Neighbors", "[core]") {
    CATCH_SECTION("Neighbor") {
        CATCH_REQUIRE(std::is_default_constructible_v<svs::Neighbor<uint32_t>> == true);
        svs::Neighbor<uint32_t> neighbor{uint32_t{10}, 2.0f};
        CATCH_REQUIRE(neighbor.id() == 10);
        CATCH_REQUIRE(neighbor.distance() == 2.0f);

        svs::IDHash neighbor_hash{};
        for (uint32_t i = 0; i < 10; i++) {
            svs::Neighbor temp{i, -100.0f};
            CATCH_REQUIRE(std::hash<decltype(i)>{}(i) == neighbor_hash(temp));
        }
        for (size_t i = 0; i < 10; i++) {
            svs::Neighbor temp{i, -100.0f};
            CATCH_REQUIRE(std::hash<decltype(i)>{}(i) == neighbor_hash(temp));
        }

        // Test Equality
        svs::IDEqual neighbor_equal{};
        auto a = svs::Neighbor<uint32_t>{1, 2};
        auto b = svs::Neighbor<uint32_t>{1, 100};
        auto c = svs::Neighbor<uint32_t>{2, 1};
        CATCH_REQUIRE(svs::equal_id(a, b));
        CATCH_REQUIRE(neighbor_equal(a, b));

        CATCH_REQUIRE(!svs::equal(a, b));

        CATCH_REQUIRE(!svs::equal_id(a, c));
        CATCH_REQUIRE(!neighbor_equal(a, c));

        CATCH_REQUIRE(
            svs::equal(svs::Neighbor<uint32_t>(1, 20), svs::Neighbor<uint32_t>(1, 20))
        );

        // Make sure this passes through a `robin_set` properly.
        auto test_length = 100;
        auto id_generator = svs_test::make_generator<uint32_t>(0, 10000);
        auto ids_set = std::unordered_set<uint32_t>();
        svs_test::populate(ids_set, id_generator, test_length);
        std::vector<uint32_t> ids(ids_set.begin(), ids_set.end());

        // Make sure we only get unique its.
        auto last = std::unique(ids.begin(), ids.end());
        ids.erase(last, ids.end());

        auto distances = std::vector<float>();
        svs_test::populate(distances, svs_test::make_generator<float>(0, 1000), ids.size());
        CATCH_REQUIRE(distances.size() == ids.size());
        auto neighbors = std::vector<svs::Neighbor<uint32_t>>();
        for (size_t i = 0; i < ids.size(); ++i) {
            neighbors.push_back({ids[i], distances[i]});
        }

        auto robin_set =
            tsl::robin_set<svs::Neighbor<uint32_t>, svs::IDHash, svs::IDEqual>();
        auto in_set = [&robin_set](svs::Neighbor<uint32_t> pair) {
            return robin_set.find(pair) != robin_set.end();
        };

        for (size_t i = 0; i < neighbors.size(); ++i) {
            robin_set.insert(neighbors[i]);
            for (size_t j = 0; j < neighbors.size(); ++j) {
                bool should_be_in = (j <= i);
                bool passed = (in_set(neighbors[j]) == should_be_in);
                if (passed != true) {
                    auto x = neighbors[i];
                    std::cout << "id = " << x.id_ << ", distance = " << x.distance_
                              << std::endl;
                }
                CATCH_REQUIRE(passed == true);
            }
        }
    }

    CATCH_SECTION("Total Order") {
        using N = svs::Neighbor<int32_t>;
        CATCH_SECTION("Less") {
            auto cmp = svs::TotalOrder(std::less<>());
            CATCH_REQUIRE(cmp(N(0, 100), N(10, 120)));
            CATCH_REQUIRE(!cmp(N(10, 120), N(0, 100)));

            CATCH_REQUIRE(cmp(N(0, 100), N(10, 100)));
            CATCH_REQUIRE(!cmp(N(10, 100), N(0, 100)));
        }

        CATCH_SECTION("Greater") {
            auto cmp = svs::TotalOrder(std::greater<>());
            CATCH_REQUIRE(!cmp(N(0, 100), N(10, 120)));
            CATCH_REQUIRE(cmp(N(10, 120), N(0, 100)));

            CATCH_REQUIRE(cmp(N(0, 100), N(10, 100)));
            CATCH_REQUIRE(!cmp(N(10, 100), N(0, 100)));
        }
    }

    CATCH_SECTION("SearchNeighbor") {
        using SN = svs::SearchNeighbor<uint32_t>;
        CATCH_REQUIRE(std::is_default_constructible_v<SN> == true);

        // Test copy assignment.
        SN a{1, 2};
        a.set_visited();
        CATCH_REQUIRE(a.visited() == true);
        SN b{};
        CATCH_REQUIRE(b.visited() == false);
        b = a;
        CATCH_REQUIRE(b.id() == 1);
        CATCH_REQUIRE(b.distance() == 2);
        CATCH_REQUIRE(b.visited() == true);

        SN neighbor{100, 1000};
        CATCH_REQUIRE(neighbor.id() == 100);
        CATCH_REQUIRE(neighbor.distance() == 1000);
        CATCH_REQUIRE(neighbor.visited() == false);
        neighbor.set_visited();
        CATCH_REQUIRE(neighbor.visited() == true);
        CATCH_REQUIRE(SN(100, 1000) < SN(100, 10000));
        CATCH_REQUIRE(SN(100, 1000) < SN(0, 10000));
        CATCH_REQUIRE(SN(100, 1000) < SN(1000, 10000));

        CATCH_REQUIRE(SN(10, 10000) > SN(100, 1000));
        CATCH_REQUIRE(SN(10, 10000) > SN(0, 1000));
        CATCH_REQUIRE(SN(10, 10000) > SN(1000, 1000));
        CATCH_REQUIRE(std::greater<>{}(SN(10, 10000), SN(1000, 1000)));

        // Equality.
        a = SN(1, 100);
        b = SN(1, 100, true);
        auto c = SN(1, 200);
        auto d = SN(1, 100);
        auto e = SN(1, 100, true);

        CATCH_REQUIRE(!svs::equal(a, b));
        CATCH_REQUIRE(!svs::equal(a, c));
        CATCH_REQUIRE(svs::equal(a, d));
        CATCH_REQUIRE(svs::equal(b, e));
    }

    CATCH_SECTION("Neighbor Conversion") {
        using SN = svs::SearchNeighbor<uint32_t>;
        using N = svs::Neighbor<uint32_t>;

        SN sn{1, 2};
        CATCH_REQUIRE(sn.visited() == false);
        sn.set_visited();
        CATCH_REQUIRE(sn.visited() == true);

        N np{sn};
        CATCH_REQUIRE(np.id_ == sn.id_);
        CATCH_REQUIRE(np.distance_ == sn.distance_);

        SN sn2{np.id(), np.distance()};
        CATCH_REQUIRE(np.id() == sn2.id());
        CATCH_REQUIRE(np.distance() == sn2.distance());
        CATCH_REQUIRE(sn2.visited() == false);
    }

    CATCH_SECTION("SkipVisit") {
        svs::SkipVisit metadata{false};
        CATCH_REQUIRE(metadata.visited() == false);
        CATCH_REQUIRE(metadata.skipped() == false);

        // Set `visited` then `skipped`.
        metadata.set_visited();
        CATCH_REQUIRE(metadata.visited() == true);
        CATCH_REQUIRE(metadata.skipped() == false);

        metadata.set_skipped();
        CATCH_REQUIRE(metadata.visited() == true);
        CATCH_REQUIRE(metadata.skipped() == true);

        // Set `skipped` then `visited`.
        metadata = svs::SkipVisit{};
        CATCH_REQUIRE(metadata.visited() == false);
        CATCH_REQUIRE(metadata.skipped() == false);

        metadata.set_skipped();
        CATCH_REQUIRE(metadata.visited() == false);
        CATCH_REQUIRE(metadata.skipped() == true);

        metadata.set_visited();
        CATCH_REQUIRE(metadata.visited() == true);
        CATCH_REQUIRE(metadata.skipped() == true);

        // Constructor initializing to skipped
        metadata = svs::SkipVisit{true};
        CATCH_REQUIRE(metadata.visited() == false);
        CATCH_REQUIRE(metadata.skipped() == true);
    }
}
