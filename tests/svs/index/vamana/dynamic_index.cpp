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

// header under test.
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/consolidate.h"

// stl
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

// svs
#include "svs/core/recall.h"
#include "svs/lib/timing.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

namespace {
template <typename T> auto copy_dataset(const T& data) {
    auto copy = svs::data::SimplePolymorphicData<typename T::element_type, T::extent>{
        data.size(), data.dimensions()};
    for (size_t i = 0; i < data.size(); ++i) {
        copy.set_datum(i, data.get_datum(i));
    }
    return copy;
}

template <typename T, typename U> void check_results(const T& results, const U& deleted) {
    for (size_t i = 0; i < svs::getsize<0>(results); ++i) {
        for (size_t j = 0; j < svs::getsize<1>(results); ++j) {
            CATCH_REQUIRE(!deleted.contains(results.at(i, j)));
        }
    }
}

template <typename T, typename U>
void check_deleted(const T& index, const U& deleted, size_t imax) {
    for (size_t i = 0; i < imax; ++i) {
        if (deleted.contains(i)) {
            CATCH_REQUIRE(index.is_deleted(i));
        } else {
            CATCH_REQUIRE(!index.is_deleted(i));
        }
    }
}

template <typename Left, typename Right>
void check_equal(const Left& left, const Right& right) {
    CATCH_REQUIRE(left.size() == right.size());
    CATCH_REQUIRE(left.dimensions() == right.dimensions());

    for (size_t i = 0, imax = left.size(); i < imax; ++i) {
        const auto& datum_left = left.get_datum(i);
        const auto& datum_right = right.get_datum(i);
        CATCH_REQUIRE(std::equal(datum_left.begin(), datum_left.end(), datum_right.begin())
        );
    }
}

} // namespace

#if defined(NDEBUG)
const double DELETE_PERCENT = 0.3;
#else
const double DELETE_PERCENT = 0.05;
#endif

CATCH_TEST_CASE("MutableVamanaIndex", "[graph_index]") {
    const size_t num_threads = 2;
    const size_t num_neighbors = 10;

    const auto base_data = test_dataset::data_blocked_f32();
    // const auto base_data = test_dataset::data_f32();
    const auto queries = test_dataset::queries();
    const auto groundtruth = test_dataset::groundtruth_euclidean();

    CATCH_SECTION("Soft Deletion") {
        // In this section, we test soft deletion.
        // The idea is as follows:
        //
        // (1) Load the test index.
        // (2) Run a round of queries to ensure that everything loading correctly.
        // (3) Set a target deletion percentage where all the neighbors returned by
        //     all results returned by the previous query plus a random collection of extras
        //     are deleted.
        //
        // (4) Rerun queries, make sure accuracy is still high and that no deleted indices
        //     are present in the results.
        auto entry_point = svs::index::load_entry_point(test_dataset::metadata_file());

        auto index = svs::index::MutableVamanaIndex{
            test_dataset::graph_blocked(),
            base_data.copy(),
            entry_point,
            svs::distance::DistanceL2(),
            svs::threads::UnitRange<size_t>(0, base_data.size()),
            num_threads};

        check_equal(base_data, index);
        index.debug_check_graph_consistency(false);

        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        index.set_search_window_size(num_neighbors);

        auto tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto original_time = svs::lib::time_difference(svs::lib::now(), tic);
        auto original_recall = svs::k_recall_at_n(groundtruth, results);
        CATCH_REQUIRE(index.entry_point() == entry_point);

        std::unordered_set<uint32_t> ids_to_delete{};
        double delete_percent = DELETE_PERCENT;
        for (size_t i = 0; i < groundtruth.size(); ++i) {
            auto slice = groundtruth.get_datum(i);
            for (size_t j = 0; j < num_neighbors; ++j) {
                auto id = slice[j];

                // For now - don't delete the entry point.
                if (id != entry_point) {
                    ids_to_delete.insert(slice[j]);
                }
            }

            if (ids_to_delete.size() > delete_percent * base_data.size()) {
                break;
            }
        }

        std::cout << "Deleting " << ids_to_delete.size() << " entries!" << std::endl;
        index.delete_entries(ids_to_delete);
        check_deleted(index, ids_to_delete, base_data.size());
        index.debug_check_graph_consistency(true);
        CATCH_REQUIRE_THROWS_AS(
            index.debug_check_graph_consistency(false), svs::ANNException
        );
        CATCH_REQUIRE(index.entry_point() == entry_point);
        // Make sure the correct points were deleted.
        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto new_time = svs::lib::time_difference(tic);

        // Make sure none of the returned results are in the deleted list.
        check_results(results.indices(), ids_to_delete);

        auto results_reference = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        index.exhaustive_search(queries.view(), num_neighbors, results_reference.view());
        auto new_recall = svs::k_recall_at_n(results_reference.indices(), results);

        // Perform graph consolidation and see how the results are effected.
        index.set_alpha(1.2);
        index.consolidate();
        index.debug_check_graph_consistency(false);
        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto post_consolidate_time = svs::lib::time_difference(tic);
        auto post_consolidate_recall =
            svs::k_recall_at_n(results_reference.indices(), results);

        // Check deletion again.
        check_deleted(index, ids_to_delete, base_data.size());
        CATCH_REQUIRE(index.entry_point() == entry_point);

        std::cout << "Original recall: " << original_recall
                  << ", New Recall: " << new_recall
                  << ", Post Recall: " << post_consolidate_recall << std::endl;
        std::cout << "Original Time: " << original_time << " (s), New Time: " << new_time
                  << " (s) Post Time: " << post_consolidate_time << std::endl;
        CATCH_REQUIRE(new_recall > original_recall);
        check_results(results.indices(), ids_to_delete);

        // Now - delete the entry point and consolidate.
        ids_to_delete.insert(entry_point);
        std::vector<size_t> entry_point_vector{};
        entry_point_vector.push_back(entry_point);
        index.delete_entries(entry_point_vector);
        index.set_alpha(1.2);
        index.consolidate();
        index.debug_check_graph_consistency(false);

        CATCH_REQUIRE(index.entry_point() != entry_point);
        index.search(queries.view(), num_neighbors, results.view());
        auto post_entrypoint_recall =
            svs::k_recall_at_n(results_reference.indices(), results);
        std::cout << "Post entry-point deletion recall: " << post_entrypoint_recall
                  << std::endl;

        // Add the deleted points back in.
        auto points = svs::data::SimpleData<float, svs::Dynamic>(
            ids_to_delete.size(), base_data.dimensions()
        );

        size_t i = 0;
        for (const auto& j : ids_to_delete) {
            points.set_datum(i, base_data.get_datum(j));
            ++i;
        }

        tic = svs::lib::now();
        index.add_points(points, ids_to_delete);
        auto insert_time = svs::lib::time_difference(tic);
        std::cout << "Insertion took: " << insert_time << " seconds!" << std::endl;

        // Check that the stored dataset and the original dataset are equal.
        check_equal(base_data, index);
        index.debug_check_graph_consistency(false);

        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto post_add_time = svs::lib::time_difference(tic);
        auto post_reinsertion_recall = svs::k_recall_at_n(groundtruth, results);
        std::cout << "Post reinsertion recall: " << post_reinsertion_recall << " in "
                  << post_add_time << " seconds." << std::endl;
    }
}
