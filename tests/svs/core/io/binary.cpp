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
#include "svs/core/io/binary.h"

// Assume Vecs reader is correct. Use fvecs data file to create fbin file
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

CATCH_TEST_CASE("Testing Binary Reader Iterator", "[core][io]") {
    auto reference = test_dataset::reference_file_contents();
    auto reference_ndims = reference.at(0).size();

    // Using Vecs file to create a binary file
    auto vecs_file = test_dataset::reference_vecs_file();
    auto loader_vecs = svs::io::vecs::VecsReader<float>(vecs_file);
    std::string binary_file = svs_test::prepare_temp_directory_v2() / "data_f32.fbin";
    {
        auto writer = svs::io::binary::BinaryWriter<float>(
            binary_file, loader_vecs.nvectors(), loader_vecs.ndims()
        );
        for (auto i : loader_vecs) {
            writer << i;
        }
    }
    auto loader_binary = svs::io::binary::BinaryReader<float>(binary_file);
    CATCH_REQUIRE(loader_binary.ndims() == reference_ndims);
    CATCH_REQUIRE(loader_binary.nvectors() == reference.size());

    CATCH_SECTION("Loading") {
        auto loader = svs::io::binary::BinaryReader<float>(binary_file, 1);

        CATCH_REQUIRE(loader.ndims() == reference_ndims);
        CATCH_REQUIRE(loader.nvectors() == reference.size());

        auto v = std::vector<float>{};
        for (auto i : loader) {
            v.insert(v.end(), i.begin(), i.end());
        }
        CATCH_REQUIRE(v.size() == reference_ndims);
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

        // Check get_dims functionality in BinaryFile
        auto [n_vecs, dims] = svs::io::binary::BinaryFile(binary_file).get_dims();
        CATCH_REQUIRE(n_vecs == reference.size());
        CATCH_REQUIRE(dims == reference_ndims);
    }

    CATCH_SECTION("Writing") {
        auto loader = svs::io::binary::BinaryReader<float>(binary_file);
        CATCH_REQUIRE(loader.ndims() == reference_ndims);
        std::string output_file = svs_test::temp_directory() / "output.fbin";
        // Introduce a scoped section to all the writer's destructor to run.
        // Should have the effect of flushing the file.
        {
            auto writer = svs::io::binary::BinaryWriter(
                output_file, loader.nvectors(), loader.ndims()
            );
            for (auto i : loader) {
                writer << i;
            }
        }
        CATCH_REQUIRE(svs_test::compare_files(binary_file, output_file) == true);
    }

    CATCH_SECTION("Error on Incorrect File or Datatype") {
        auto incorrect_binary_file = test_dataset::reference_svs_file();
        CATCH_REQUIRE_THROWS_MATCHES(
            svs::io::binary::BinaryReader<float>(incorrect_binary_file),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring(incorrect_binary_file) &&
                Catch::Matchers::ContainsSubstring(fmt::format("{}", sizeof(float)))
            )
        );

        CATCH_REQUIRE_THROWS_MATCHES(
            svs::io::binary::BinaryFile(incorrect_binary_file).get_dims(),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring(incorrect_binary_file)
            )
        );

        // We should also get an error if we supply an incorrectly sized type.
        CATCH_REQUIRE_THROWS_MATCHES(
            svs::io::binary::BinaryReader<svs::Float16>(binary_file),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring(binary_file) &&
                Catch::Matchers::ContainsSubstring(fmt::format("{}", sizeof(svs::Float16)))
            )
        );
    }
}
