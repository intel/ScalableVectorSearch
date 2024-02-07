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
#include "svs/core/io/vecs.h"

// svs_test
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

// stl
#include <iostream>
#include <numeric>
#include <vector>

CATCH_TEST_CASE("Testing Vecs Reader Iterator", "[core][io]") {
    auto reference = test_dataset::reference_file_contents();
    auto reference_ndims = reference.at(0).size();

    CATCH_SECTION("Loading") {
        auto vecs_file = test_dataset::reference_vecs_file();
        auto loader = svs::io::vecs::VecsReader<float>(vecs_file, 1);

        CATCH_REQUIRE(loader.ndims() == reference.at(0).size());
        CATCH_REQUIRE(loader.nvectors() == reference.size());

        auto v = std::vector<float>{};
        for (auto i : loader) {
            v.insert(v.end(), i.begin(), i.end());
        }
        CATCH_REQUIRE(v.size() == reference.at(0).size());
        CATCH_REQUIRE(std::equal(v.begin(), v.end(), reference.at(0).begin()));

        // Read entire file
        loader.resize();
        auto loader_it = loader.begin();
        auto loader_end = loader.end();
        for (size_t i = 0, imax = reference.size(); i < imax; ++i) {
            v.clear();
            const auto& this_reference = reference.at(i);
            CATCH_REQUIRE(loader_it != loader_end);
            const auto& slice = *loader_it;
            CATCH_REQUIRE(slice.size() == this_reference.size());
            CATCH_REQUIRE(std::equal(slice.begin(), slice.end(), this_reference.begin()));
            ++loader_it;
        }
        CATCH_REQUIRE(!(loader_it != loader_end));
    }

    CATCH_SECTION("Writing") {
        auto vecs_file = test_dataset::reference_vecs_file();
        auto loader = svs::io::vecs::VecsReader<float>(vecs_file);
        CATCH_REQUIRE(loader.ndims() == reference_ndims);
        CATCH_REQUIRE(svs_test::prepare_temp_directory() == true);
        std::string output_file = svs_test::temp_directory() / "output.fvecs";
        // Introduce a scoped section to all the writer's destructor to run.
        // Should have the effect of flushing the file.
        {
            auto writer = svs::io::vecs::VecsWriter(output_file, loader.ndims());
            for (auto i : loader) {
                writer << i;
            }
        }
        CATCH_REQUIRE(svs_test::compare_files(vecs_file, output_file) == true);
    }

    CATCH_SECTION("Error on Incorrect File") {
        auto incorrect_vecs_file = test_dataset::reference_svs_file();
        CATCH_REQUIRE_THROWS_MATCHES(
            svs::io::vecs::VecsReader<float>(incorrect_vecs_file),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring(incorrect_vecs_file) &&
                Catch::Matchers::ContainsSubstring(fmt::format("{}", sizeof(float)))
            )
        );

        // We should also get an error if we supply an incorrectly sized type.
        auto vecs_file = test_dataset::reference_vecs_file();
        CATCH_REQUIRE_THROWS_MATCHES(
            svs::io::vecs::VecsReader<svs::Float16>(incorrect_vecs_file),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring(incorrect_vecs_file) &&
                Catch::Matchers::ContainsSubstring(fmt::format("{}", sizeof(svs::Float16)))
            )
        );
    }
}
