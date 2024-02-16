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

// Header under test
#include "svs/core/distance.h"

// catch 2
#include "catch2/catch_test_macros.hpp"

namespace {

std::string_view test_table = R"(
__schema__ = 'distance_checker'
__version__ = 'v1.2.3'
euclidean = 'L2'
inner_product = "MIP"
cosine = "Cosine"
)";

std::string_view invalid_table = R"(
__schema__ = 'distance_checker'
__version__ = 'v1.2.3'
euclidean = 'L22'
)";

struct DistanceChecker {
  public:
    svs::DistanceType euclidean;
    svs::DistanceType inner_product;
    svs::DistanceType cosine;

    static bool
    check_load_compatibility(std::string_view schema, svs::lib::Version version) {
        return schema == "distance_checker" && version == svs::lib::Version{1, 2, 3};
    }

    static DistanceChecker load(const svs::lib::ContextFreeLoadTable& table) {
        CATCH_REQUIRE(table.version() == svs::lib::Version{1, 2, 3});
        CATCH_REQUIRE(table.schema() == "distance_checker");

        return DistanceChecker{
            .euclidean = SVS_LOAD_MEMBER_AT(table, euclidean),
            .inner_product = SVS_LOAD_MEMBER_AT(table, inner_product),
            .cosine = SVS_LOAD_MEMBER_AT(table, cosine),
        };
    }
};

} // namespace

CATCH_TEST_CASE("Distance Utils", "[core][distance][distance_type]") {
    CATCH_SECTION("Saving and Loading") {
        // First, check that the expected test table parses correctly.
        CATCH_SECTION("Loading pre-saved table") {
            auto table = toml::parse(test_table);
            auto checker =
                svs::lib::load<DistanceChecker>(svs::lib::ContextFreeLoadTable(table));
            CATCH_REQUIRE(checker.euclidean == svs::DistanceType::L2);
            CATCH_REQUIRE(checker.inner_product == svs::DistanceType::MIP);
            CATCH_REQUIRE(checker.cosine == svs::DistanceType::Cosine);
        }

        CATCH_SECTION("Testing Round-trip") {
            auto f = [](svs::DistanceType t) {
                auto saved = svs::lib::save(t);
                auto loaded = svs::lib::load<svs::DistanceType>(svs::lib::node_view(saved));
                CATCH_REQUIRE(loaded == t);
            };
            f(svs::DistanceType::L2);
            f(svs::DistanceType::MIP);
            f(svs::DistanceType::Cosine);
        }

        CATCH_SECTION("Invalid Checking") {
            auto table = toml::parse(invalid_table);
            auto node_view = svs::lib::node_view(table);
            CATCH_REQUIRE_THROWS_AS(
                svs::lib::load_at<svs::DistanceType>(node_view, "euclidean"),
                svs::ANNException
            );
        }
    }
}
