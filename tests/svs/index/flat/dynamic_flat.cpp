/*
 * Copyright 2025 Intel Corporation
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

// svs
#include "svs/index/flat/dynamic_flat.h"
#include "svs/core/recall.h"
#include "svs/index/flat/flat.h"
#include "svs/lib/float16.h"
#include "svs/lib/timing.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

using Idx = uint32_t;
using Eltype = float;
using Distance = svs::distance::DistanceL2;
const size_t N = 128;

CATCH_TEST_CASE(
    "Dynamic Flat Index Basic Constructor Test", "[dynamic_flat][constructor]"
) {
    // Load test data
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto num_threads = 4;

    // Create initial data
    auto initial_count = std::min(size_t{100}, data.size());
    auto initial_data = svs::data::SimpleData<Eltype, N>(initial_count, N);
    std::vector<Idx> initial_ids(initial_count);

    for (size_t i = 0; i < initial_count; ++i) {
        initial_data.set_datum(i, data.get_datum(i));
        initial_ids[i] = static_cast<Idx>(i);
    }

    CATCH_SECTION("Auto Dynamic Assemble") {
        // Test the auto_dynamic_assemble function
        auto index = svs::index::flat::auto_dynamic_assemble(data, Distance{}, num_threads);

        CATCH_REQUIRE(index.get_logger() != nullptr);
        CATCH_REQUIRE(index.size() == data.size());
        CATCH_REQUIRE(index.dimensions() == N);
        std::cout << "Auto dynamic assemble successful with " << data.size() << " points\n";
    }

    CATCH_SECTION("Add Points Test") {
        auto index = svs::index::flat::DynamicFlatIndex(
            std::move(initial_data), initial_ids, Distance{}, num_threads
        );

        // Verify initial state
        size_t original_size = index.size();
        CATCH_REQUIRE(original_size == initial_count);

        // Create some additional vectors to add
        size_t add_count = std::min(size_t{20}, data.size() - initial_count);
        if (add_count > 0) {
            auto add_data = svs::data::SimpleData<Eltype, N>(add_count, N);
            std::vector<Idx> add_ids(add_count);

            // Copy vectors from the original data that weren't used initially
            for (size_t i = 0; i < add_count; ++i) {
                add_data.set_datum(i, data.get_datum(initial_count + i));
                add_ids[i] =
                    static_cast<Idx>(initial_count + i + 1000); // Use different ID range
            }

            // Add the vectors
            auto slots_used = index.add_points(add_data, add_ids);

            // Verify the results
            CATCH_REQUIRE(slots_used.size() == add_count);
            CATCH_REQUIRE(index.size() == original_size + add_count);
            CATCH_REQUIRE(index.dimensions() == N); // Dimensions should remain the same

            std::cout << "Successfully added " << add_count
                      << " vectors. New size: " << index.size() << "\n";
        } else {
            std::cout << "Skipping add_points test - not enough additional data\n";
        }
    }

    CATCH_SECTION("Delete Entries Test") {
        auto index = svs::index::flat::DynamicFlatIndex(
            std::move(initial_data), initial_ids, Distance{}, num_threads
        );

        // First, add some points so we have more to work with
        size_t add_count = std::min(size_t{20}, data.size() - initial_count);
        std::vector<Idx> added_ids;

        if (add_count > 0) {
            auto add_data = svs::data::SimpleData<Eltype, N>(add_count, N);
            std::vector<Idx> add_ids(add_count);

            for (size_t i = 0; i < add_count; ++i) {
                add_data.set_datum(i, data.get_datum(initial_count + i));
                add_ids[i] = static_cast<Idx>(initial_count + i + 1000);
            }
            added_ids = add_ids;

            index.add_points(add_data, add_ids);
        }

        // Verify initial state after additions
        size_t size_before_deletion = index.size();
        std::cout << "Size before deletion: " << size_before_deletion << "\n";

        // Test deletion with some of the original IDs
        std::vector<Idx> ids_to_delete;
        size_t num_to_delete = std::min(size_t{5}, size_before_deletion);

        // Delete some original IDs
        for (size_t i = 0; i < num_to_delete && i < initial_ids.size(); ++i) {
            ids_to_delete.push_back(initial_ids[i]);
        }

        // Also delete some added IDs if we have them
        size_t added_to_delete = std::min(size_t{3}, added_ids.size());
        for (size_t i = 0; i < added_to_delete; ++i) {
            ids_to_delete.push_back(added_ids[i]);
        }

        // Verify all IDs exist before deletion
        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(index.has_id(id));
        }

        // Perform deletion
        size_t deleted_count = index.delete_entries(ids_to_delete);
        CATCH_REQUIRE(deleted_count == ids_to_delete.size());

        // Verify size decreased
        CATCH_REQUIRE(index.size() == size_before_deletion - ids_to_delete.size());

        // Verify deleted IDs no longer exist in the index
        for (auto id : ids_to_delete) {
            CATCH_REQUIRE_FALSE(index.has_id(id));
        }

        std::cout << "Successfully deleted " << deleted_count
                  << " entries. New size: " << index.size() << "\n";

        // Test deleting non-existent ID should throw
        std::vector<Idx> non_existent_ids = {99999};
        CATCH_REQUIRE_THROWS(index.delete_entries(non_existent_ids));
    }

    CATCH_SECTION("Compact Test") {
        auto index = svs::index::flat::DynamicFlatIndex(
            std::move(initial_data), initial_ids, Distance{}, num_threads
        );

        // First, add some points
        size_t add_count = std::min(size_t{30}, data.size() - initial_count);
        std::vector<Idx> added_ids;

        if (add_count > 0) {
            auto add_data = svs::data::SimpleData<Eltype, N>(add_count, N);
            std::vector<Idx> add_ids(add_count);

            for (size_t i = 0; i < add_count; ++i) {
                add_data.set_datum(i, data.get_datum(initial_count + i));
                add_ids[i] = static_cast<Idx>(initial_count + i + 1000);
            }
            added_ids = add_ids;
            index.add_points(add_data, add_ids);
        }

        // Delete some entries to create fragmentation
        std::vector<Idx> ids_to_delete;
        size_t num_to_delete = std::min(size_t{10}, index.size() / 2);

        // Delete every other original ID to create fragmentation
        for (size_t i = 0; i < num_to_delete && i * 2 < initial_ids.size(); ++i) {
            ids_to_delete.push_back(initial_ids[i * 2]);
        }

        // Also delete some added IDs
        size_t added_to_delete = std::min(size_t{5}, added_ids.size());
        for (size_t i = 0; i < added_to_delete; ++i) {
            ids_to_delete.push_back(added_ids[i]);
        }

        if (!ids_to_delete.empty()) {
            index.delete_entries(ids_to_delete);
        }

        size_t size_before_compact = index.size();
        std::cout << "Size before compact: " << size_before_compact << "\n";

        // Get all existing IDs before compaction for verification
        std::vector<size_t> ids_before_compact;
        index.on_ids([&ids_before_compact](size_t id) {
            ids_before_compact.push_back(id);
        });

        // Perform compaction
        index.compact();

        // Verify size is preserved
        CATCH_REQUIRE(index.size() == size_before_compact);

        // Verify all IDs still exist after compaction
        for (auto id : ids_before_compact) {
            CATCH_REQUIRE(index.has_id(id));
        }

        // Verify dimensions are preserved
        CATCH_REQUIRE(index.dimensions() == N);

        std::cout << "Successfully compacted. Size after compact: " << index.size() << "\n";
    }

    CATCH_SECTION("Save and Load Test") {
        auto index = svs::index::flat::DynamicFlatIndex(
            std::move(initial_data), initial_ids, Distance{}, num_threads
        );

        // Add some points to make the test more meaningful
        size_t add_count = std::min(size_t{15}, data.size() - initial_count);
        if (add_count > 0) {
            auto add_data = svs::data::SimpleData<Eltype, N>(add_count, N);
            std::vector<Idx> add_ids(add_count);

            for (size_t i = 0; i < add_count; ++i) {
                add_data.set_datum(i, data.get_datum(initial_count + i));
                add_ids[i] = static_cast<Idx>(initial_count + i + 2000);
            }
            index.add_points(add_data, add_ids);
        }

        size_t size_before_save = index.size();
        std::cout << "Size before save: " << size_before_save << "\n";

        // Create temporary directory for saving
        auto temp_dir = std::filesystem::temp_directory_path() / "dynamic_flat_save_test";
        std::filesystem::create_directories(temp_dir);
        auto data_dir = temp_dir / "data";

        // Save the index (data only)
        index.save(data_dir);

        // Load the index back
        auto loaded_index = svs::index::flat::auto_dynamic_assemble(
            SVS_LAZY(svs::data::SimpleData<Eltype>::load(data_dir)), Distance{}, num_threads
        );

        // Verify the loaded index properties
        CATCH_REQUIRE(loaded_index.size() == size_before_save);
        CATCH_REQUIRE(loaded_index.dimensions() == N);

        std::cout << "Successfully saved and loaded index with " << loaded_index.size()
                  << " points\n";

        // Clean up temporary files
        std::filesystem::remove_all(temp_dir);
    }
}
