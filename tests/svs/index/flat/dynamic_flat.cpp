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
#include "svs/misc/dynamic_helper.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

using Idx = uint32_t;
using Eltype = float;
using QueryEltype = float;
using Distance = svs::distance::DistanceL2;
const size_t N = 128;

const size_t NUM_NEIGHBORS = 10;
const double TARGET_RECALL = 0.95;

///
/// Utility Methods
///

template <std::integral I> I div(I i, float fraction) {
    return svs::lib::narrow<I>(std::floor(svs::lib::narrow<float>(i) * fraction));
}

template <typename... Args> std::string stringify(Args&&... args) {
    std::ostringstream stream{};
    ((stream << args), ...);
    return stream.str();
}

///
/// Main Loop.
///

template <typename MutableIndex, typename Queries>
void test_loop(
    MutableIndex& index,
    svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>& reference,
    const Queries& queries,
    size_t num_points,
    size_t consolidate_every,
    size_t iterations
) {
    // Suppress unused variable warnings for now
    (void)queries;
    (void)consolidate_every;

    size_t consolidate_count = 0;
    for (size_t i = 0; i < iterations; ++i) {
        // Add Points
        {
            auto [points, time] = reference.add_points(index, num_points);
            (void)time; // Suppress unused warning
            CATCH_REQUIRE(points <= num_points);
            CATCH_REQUIRE(points > num_points - reference.bucket_size());
        }

        // Delete Points
        {
            auto [points, time] = reference.delete_points(index, num_points);
            (void)time; // Suppress unused warning
            CATCH_REQUIRE(points <= num_points);
            CATCH_REQUIRE(points > num_points - reference.bucket_size());
        }
    }
    (void)consolidate_count; // Suppress unused warning
}

CATCH_TEST_CASE("Testing Flat Index", "[dynamic_flat]") {
#if defined(NDEBUG)
    const float initial_fraction = 0.25;
    const float modify_fraction = 0.05;
#else
    const float initial_fraction = 0.05;
    const float modify_fraction = 0.005;
#endif
    const size_t num_threads = 10;

    // Load the base dataset and queries.
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto num_points = data.size();
    auto queries = test_dataset::queries();

    auto reference = svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>(
        std::move(data),
        Distance(),
        num_threads,
        div(num_points, 0.5 * modify_fraction),
        NUM_NEIGHBORS,
        queries,
        0x12345678
    );

    auto num_indices_to_add = div(reference.size(), initial_fraction);

    // Construct a blocked dataset consisting of initial fraction of the base dataset.
    auto data_mutable = svs::data::BlockedData<Eltype, N>(num_indices_to_add, N);
    std::vector<Idx> initial_indices{};
    {
        auto [vectors, indices] = reference.generate(num_indices_to_add);
        // Copy assign ``initial_indices``
        auto num_points_added = indices.size();
        CATCH_REQUIRE(vectors.size() == num_points_added);
        CATCH_REQUIRE(num_points_added <= num_indices_to_add);
        CATCH_REQUIRE(num_points_added > num_indices_to_add - reference.bucket_size());

        initial_indices = indices;
        if (vectors.size() != num_indices_to_add || indices.size() != num_indices_to_add) {
            throw ANNEXCEPTION("Something when horribly wrong!");
        }

        for (size_t i = 0; i < num_indices_to_add; ++i) {
            data_mutable.set_datum(i, vectors.get_datum(i));
        }
    }

    auto tic = svs::lib::now();
    auto index = svs::index::flat::DynamicFlatIndex(
        std::move(data_mutable), initial_indices, Distance(), num_threads
    );
    double build_time = svs::lib::time_difference(tic);
    (void)build_time; // Suppress unused warning

    reference.configure_extra_checks(true);
    CATCH_REQUIRE(reference.extra_checks_enabled());

    test_loop(index, reference, queries, div(reference.size(), modify_fraction), 2, 6);
}
