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
 * See            CATCH_REQUIRE(original_result.index(q, k) == loaded_result.index(q, k));
            CATCH_REQUIRE(
                std::abs(original_result.distance(q, k) - loaded_result.distance(q, k)) <
                eps
            );License for the specific language governing permissions and
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
const size_t NUM_NEIGHBORS = 10;
const double TARGET_RECALL = 0.95;

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
}
