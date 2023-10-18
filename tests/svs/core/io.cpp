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

// svs
#include "svs/concepts/data.h"
#include "svs/core/allocator.h"
#include "svs/core/graph.h"
#include "svs/lib/array.h"
#include "svs/lib/memory.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <span>

template <svs::data::ImmutableMemoryDataset Data1, svs::data::ImmutableMemoryDataset Data2>
bool compare(const Data1& x, const Data2& y) {
    CATCH_REQUIRE(x.size() == y.size());
    CATCH_REQUIRE(x.dimensions() == y.dimensions());
    for (size_t i = 0; i < x.size(); ++i) {
        auto x_datum = x.get_datum(i);
        auto y_datum = y.get_datum(i);
        if (!std::equal(x_datum.begin(), x_datum.end(), y_datum.begin())) {
            return false;
        }
    }
    return true;
}

template <typename Idx>
void compare(
    const svs::graphs::SimpleGraph<Idx>& x, const svs::graphs::SimpleGraph<Idx>& y
) {
    CATCH_REQUIRE(x.n_nodes() == y.n_nodes());
    CATCH_REQUIRE(x.max_degree() == y.max_degree());
    for (size_t i = 0; i < x.n_nodes(); ++i) {
        auto x_neighbors = x.get_node(i);
        auto y_neighbors = y.get_node(i);
        CATCH_REQUIRE(x_neighbors.size() == y_neighbors.size());
        // Test that the pointers actually point to different places.
        // If we pass two instances of the same graph, this will fail.
        // BUT, it's a way to protect against actually doing that.
        CATCH_REQUIRE(x_neighbors.data() != y_neighbors.data());
        CATCH_REQUIRE(
            std::equal(x_neighbors.begin(), x_neighbors.end(), y_neighbors.begin())
        );
    }
}

// Testing mixing and matching dataset loading.
CATCH_TEST_CASE("Testing Dataset Loading and Writing", "[core][integrated_io]") {
    CATCH_REQUIRE(svs_test::prepare_temp_directory());

    const size_t EXPECTED_EXTENT = 6;

    auto reference = test_dataset::reference_file_contents();
    auto reference_ndims = reference.at(0).size();
    auto reference_nvectors = reference.size();
    CATCH_REQUIRE(reference_ndims == EXPECTED_EXTENT);

    auto vecs_file = test_dataset::reference_vecs_file();
    auto native_file_reference = test_dataset::reference_svs_file();
    auto native_file_test = svs_test::temp_directory() / "data_test.svs";

    // Load data into memory from the vecs file.
    auto index_data = svs::data::SimpleData<float, EXPECTED_EXTENT>::load(vecs_file);

    CATCH_REQUIRE(index_data.size() == reference_nvectors);
    CATCH_REQUIRE(index_data.dimensions() == reference_ndims);
    // Make sure static size information is propagated correctly.
    CATCH_REQUIRE(index_data.get_datum(0).extent == reference_ndims);

    CATCH_SECTION("Verifying initial load") {
        for (size_t i = 0, imax = index_data.size(); i < imax; ++i) {
            auto span = index_data.get_datum(i);
            CATCH_REQUIRE(std::equal(span.begin(), span.end(), reference.at(i).begin()));
        }
    }

    // Save directly to file.
    CATCH_SECTION("Standard Saving") {
        svs::io::save(index_data, svs::io::v1::NativeFile{native_file_test});
        // CATCH_REQUIRE(svs_test::compare_files(native_file_reference, native_file_test));

        // Load back from file - make sure what we get back matches.
        // Use a dynamically sized dimension for fun.
        CATCH_SECTION("Dynamic") {
            auto native = svs::data::SimpleData<float>::load(native_file_test);
            CATCH_REQUIRE(compare(index_data, native));
            CATCH_REQUIRE(native.get_datum(0).extent == svs::Dynamic);
        }

        CATCH_SECTION("Static") {
            auto native =
                svs::data::SimpleData<float, EXPECTED_EXTENT>::load(native_file_test);
            CATCH_REQUIRE(compare(index_data, native));
            CATCH_REQUIRE(native.get_datum(0).extent == reference_ndims);
        }
    }
}

CATCH_TEST_CASE("Testing Graph Loading and Saving", "[core][integrated_io]") {
    CATCH_REQUIRE(svs_test::prepare_temp_directory());
    auto native_file_reference = test_dataset::graph_file();
    auto temp_dir = svs_test::temp_directory();

    // Standard graph loading and saving.
    CATCH_REQUIRE(std::filesystem::exists(native_file_reference));
    auto reference_graph = svs::graphs::SimpleGraph<uint32_t>::load(native_file_reference);
    CATCH_REQUIRE(reference_graph.max_degree() == test_dataset::NUM_DIMENSIONS);
    CATCH_REQUIRE(reference_graph.n_nodes() == test_dataset::VECTORS_IN_DATA_SET);

    auto expected_num_neighbors = test_dataset::expected_out_neighbors();
    for (size_t i = 0; i < expected_num_neighbors.size(); ++i) {
        auto neighbors = reference_graph.get_node(i);
        CATCH_REQUIRE(neighbors.size() == expected_num_neighbors.at(i));
    }

    CATCH_SECTION("Standard Saving") {
        svs::lib::save_to_disk(reference_graph, temp_dir);
        auto other = svs::lib::load_from_disk<svs::graphs::SimpleGraph<uint32_t>>(temp_dir);
        compare(reference_graph, other);
    }
}
