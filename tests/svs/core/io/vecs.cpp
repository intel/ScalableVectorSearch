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
        std::string output_file = svs_test::prepare_temp_directory_v2() / "output.fvecs";
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

    CATCH_SECTION("VecsFile interface") {
        auto vecs_file = test_dataset::reference_vecs_file();

        auto [n_vecs, dims] = svs::io::vecs::VecsFile(vecs_file).get_dims();
        CATCH_REQUIRE(n_vecs == reference.size());
        CATCH_REQUIRE(dims == reference.at(0).size());

        // Create file with a single vectors and check deduced dimensions
        auto loader = svs::io::vecs::VecsReader<float>(vecs_file, 1);
        std::string output_file = svs_test::prepare_temp_directory_v2() / "output.fvecs";
        {
            auto writer = svs::io::vecs::VecsWriter<float>(output_file, loader.ndims());
            for (auto i : loader) {
                writer << i;
            }
        }

        std::tie(n_vecs, dims) = svs::io::vecs::VecsFile(output_file).get_dims();
        CATCH_REQUIRE(n_vecs == 1);
        CATCH_REQUIRE(dims == reference.at(0).size());

        // Create file with two vectors of float16 type
        loader = svs::io::vecs::VecsReader<float>(vecs_file, 2);
        {
            auto writer =
                svs::io::vecs::VecsWriter<svs::Float16>(output_file, loader.ndims());
            for (auto i : loader) {
                writer << i;
            }
        }

        std::tie(n_vecs, dims) = svs::io::vecs::VecsFile(output_file).get_dims();
        CATCH_REQUIRE(n_vecs == 2);
        CATCH_REQUIRE(dims == reference.at(0).size());
    }

    CATCH_SECTION("Error on Incorrect File or Datatype") {
        auto incorrect_vecs_file = test_dataset::reference_svs_file();
        CATCH_REQUIRE_THROWS_MATCHES(
            svs::io::vecs::VecsReader<float>(incorrect_vecs_file),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring(incorrect_vecs_file) &&
                Catch::Matchers::ContainsSubstring(fmt::format("{}", sizeof(float)))
            )
        );

        CATCH_REQUIRE_THROWS_MATCHES(
            svs::io::vecs::VecsFile(incorrect_vecs_file).get_dims(),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring(incorrect_vecs_file)
            )
        );

        // We should also get an error if we supply an incorrectly sized type.
        auto vecs_file = test_dataset::reference_vecs_file();
        CATCH_REQUIRE_THROWS_MATCHES(
            svs::io::vecs::VecsReader<svs::Float16>(vecs_file),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring(vecs_file) &&
                Catch::Matchers::ContainsSubstring(fmt::format("{}", sizeof(svs::Float16)))
            )
        );
    }
}
