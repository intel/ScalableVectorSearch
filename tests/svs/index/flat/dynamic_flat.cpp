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

    CATCH_SECTION("Basic Construction") {
        auto index = svs::index::flat::DynamicFlatIndex(
            std::move(initial_data), initial_ids, Distance{}, num_threads
        );

        // Verify basic properties
        CATCH_REQUIRE(index.get_logger() != nullptr);
        std::cout << "Dynamic Flat Index constructed successfully with " << initial_count
                  << " points\n";
    }

    CATCH_SECTION("Construction with Custom Logger") {
        auto logger = svs::logging::get();
        auto initial_data_copy = svs::data::SimpleData<Eltype, N>(initial_count, N);
        for (size_t i = 0; i < initial_count; ++i) {
            initial_data_copy.set_datum(i, data.get_datum(i));
        }

        auto index = svs::index::flat::DynamicFlatIndex(
            std::move(initial_data_copy), initial_ids, Distance{}, num_threads, logger
        );

        CATCH_REQUIRE(index.get_logger() == logger);
        std::cout << "Dynamic Flat Index constructed with custom logger\n";
    }

    CATCH_SECTION("Auto Dynamic Assemble") {
        // Test the auto_dynamic_assemble function
        auto index = svs::index::flat::auto_dynamic_assemble(data, Distance{}, num_threads);

        CATCH_REQUIRE(index.get_logger() != nullptr);
        std::cout << "Auto dynamic assemble successful with " << data.size() << " points\n";
    }
}
