/*
 * Copyright 2026 Intel Corporation
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

#pragma once

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/index/vamana/dynamic_search_buffer.h"
#include "svs/lib/exception.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace svs_test {

///
/// @brief Check the ``range_search`` of an orchestrator built with L2 over ``data`` and
/// queried with float queries, with external IDs equal to the row in ``data``.
///
template <typename Index, typename Data>
void check_orchestrator_range_search(
    Index& index, const Data& data, const svs::data::SimpleData<float>& all_queries
) {
    const size_t num_queries = 20;
    auto queries = svs::data::SimpleData<float>(num_queries, all_queries.dimensions());
    for (size_t q = 0; q < num_queries; ++q) {
        queries.set_datum(q, all_queries.get_datum(q));
    }
    auto distance = svs::distance::DistanceL2();

    // Brute-force distances and a radius that holds about 30 neighbors per query.
    auto distances = std::vector<std::vector<float>>(num_queries);
    float radius = 0;
    for (size_t q = 0; q < num_queries; ++q) {
        for (size_t i = 0; i < data.size(); ++i) {
            distances[q].push_back(
                svs::distance::compute(distance, queries.get_datum(q), data.get_datum(i))
            );
        }
        auto sorted = distances[q];
        std::nth_element(sorted.begin(), sorted.begin() + 30, sorted.end());
        radius += sorted[30];
    }
    radius /= static_cast<float>(num_queries);

    auto check = [&](const auto& results, size_t max_results) {
        CATCH_REQUIRE(results.size() == num_queries);
        size_t num_expected = 0;
        size_t num_found = 0;
        for (size_t q = 0; q < num_queries; ++q) {
            const auto& result = results[q];
            CATCH_REQUIRE(result.size() <= max_results);
            auto seen = std::unordered_set<size_t>{};
            float last = 0;
            for (const auto& neighbor : result) {
                CATCH_REQUIRE(neighbor.id() < data.size());
                CATCH_REQUIRE(neighbor.distance() < radius);
                CATCH_REQUIRE(neighbor.distance() >= last);
                CATCH_REQUIRE(seen.insert(neighbor.id()).second);
                last = neighbor.distance();
            }
            for (size_t i = 0; i < data.size(); ++i) {
                if (distances[q][i] < radius) {
                    ++num_expected;
                    num_found += seen.contains(i);
                }
            }
        }
        return static_cast<double>(num_found) / static_cast<double>(num_expected);
    };

    using buffer_type = svs::index::vamana::MutableBuffer<uint32_t>;
    auto sp = index.get_search_parameters();

    // Unbounded. The shared test dataset is hard for graph search (top-30 at window 20);
    // the index tests check recall more strictly.
    sp.buffer_config({20, buffer_type::unbounded_capacity});
    CATCH_REQUIRE(
        check(index.range_search(queries.cview(), radius, sp), data.size()) >= 0.85
    );

    // Bounded by the buffer capacity.
    sp.buffer_config({10, 20});
    auto bounded = index.range_search(queries.cview(), radius, sp);
    check(bounded, 20);
    CATCH_REQUIRE(std::any_of(bounded.begin(), bounded.end(), [](const auto& r) {
        return r.size() == 20;
    }));

    // Cancelled before starting.
    sp.buffer_config({20, buffer_type::unbounded_capacity});
    auto none = index.range_search(queries.cview(), radius, sp, []() { return true; });
    CATCH_REQUIRE(none.size() == num_queries);
    for (const auto& result : none) {
        CATCH_REQUIRE(result.empty());
    }

    // Unsupported query type.
    auto bytes = svs::data::SimpleData<uint8_t>(1, queries.dimensions());
    CATCH_REQUIRE_THROWS_AS(
        index.range_search(bytes.cview(), radius, sp), svs::ANNException
    );
}

} // namespace svs_test
