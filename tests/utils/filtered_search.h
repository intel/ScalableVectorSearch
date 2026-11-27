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
#include "svs/concepts/distance.h"
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/query_result.h"
#include "svs/index/vamana/build_params.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/type_traits.h"

// tests
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cstddef>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace svs_test {

/// @brief Checks filtered search results against an exhaustive scan of the test dataset.
///
/// Row ``i`` of the dataset is labeled ``first_id + i``. Only the first ``num_queries``
/// test queries are used, to keep the scan cheap.
template <typename Distance> class FilteredSearchReference {
  public:
    /// The neighbor a filtered search writes to result slots it could not fill.
    static constexpr svs::Neighbor<size_t> padding = svs::type_traits::
        sentinel_v<svs::Neighbor<size_t>, svs::distance::compare_t<Distance>>;

    explicit FilteredSearchReference(size_t num_queries, size_t first_id = 0)
        : data_{test_dataset::data_f32()}
        , queries_{num_queries, data_.dimensions()}
        , ids_(data_.size()) {
        auto all_queries = test_dataset::queries();
        for (size_t i = 0; i < num_queries; ++i) {
            queries_.set_datum(i, all_queries.get_datum(i));
        }
        for (size_t i = 0; i < ids_.size(); ++i) {
            ids_[i] = first_id + i;
            rows_.emplace(ids_[i], i);
        }
    }

    const svs::data::SimpleData<float>& queries() const { return queries_; }

    /// The initial labels of the dataset rows, in row order.
    const std::vector<size_t>& ids() const { return ids_; }

    /// Parameters for a quick build over the test dataset with ``Distance``.
    static svs::index::vamana::VamanaBuildParameters build_parameters() {
        // Distances where larger is better need alpha below one.
        const bool smaller_is_better = svs::distance::comparator(Distance{})(0.0f, 1.0f);
        return {smaller_is_better ? 1.2f : 0.95f, 64, 10, 20, 10, true};
    }

    /// Recall floor for a filter passing one label in ten: such a filter makes the search
    /// look much further than the plain search does, even on a quick build.
    static constexpr double min_selective_recall = 0.9;

    /// Forget labels that were deleted from the index under test.
    void erase(std::span<const size_t> ids) {
        for (auto id : ids) {
            rows_.erase(id);
        }
    }

    /// Follow a relabel made in the index under test.
    void relabel(size_t old_id, size_t new_id) {
        auto node = rows_.extract(old_id);
        node.key() = new_id;
        rows_.insert(std::move(node));
    }

    /// @brief Require well-formed results for a search with ``filter``; return the recall.
    ///
    /// Each row must hold as many neighbors as there are live labels passing ``filter``
    /// (up to the row length), best first and each with its true distance, followed only
    /// by padding.
    template <typename Filter>
    double check(const svs::QueryResult<size_t>& results, const Filter& filter) const {
        CATCH_REQUIRE(results.n_queries() == queries_.size());
        const size_t num_neighbors = results.n_neighbors();
        const auto compare = svs::distance::comparator(Distance{});

        size_t hits = 0;
        size_t total = 0;
        auto exact = std::vector<std::pair<float, size_t>>{};
        auto seen = std::unordered_set<size_t>{};
        for (size_t q = 0; q < queries_.size(); ++q) {
            const auto query = queries_.get_datum(q);
            auto distance = Distance{};
            svs::distance::maybe_fix_argument(distance, query);
            auto true_distance = [&](size_t row) {
                return svs::distance::compute(distance, query, data_.get_datum(row));
            };

            exact.clear();
            for (const auto& [id, row] : rows_) {
                if (filter(id)) {
                    exact.emplace_back(true_distance(row), id);
                }
            }
            std::sort(exact.begin(), exact.end(), [&](const auto& a, const auto& b) {
                return compare(a.first, b.first);
            });
            exact.resize(std::min(num_neighbors, exact.size()));

            seen.clear();
            size_t found = 0;
            for (; found < num_neighbors; ++found) {
                const size_t id = results.index(q, found);
                if (id == padding.id()) {
                    break;
                }
                CATCH_INFO("query " << q << ", neighbor " << found << ", id " << id);
                CATCH_REQUIRE(filter(id));
                CATCH_REQUIRE(seen.insert(id).second);
                auto row = rows_.find(id);
                CATCH_REQUIRE(row != rows_.end());
                const float returned = results.distance(q, found);
                CATCH_REQUIRE(
                    returned ==
                    Catch::Approx(true_distance(row->second)).epsilon(1e-4).margin(1e-3)
                );
                if (found > 0) {
                    CATCH_REQUIRE(!compare(returned, results.distance(q, found - 1)));
                }
                hits += std::any_of(exact.begin(), exact.end(), [&](const auto& e) {
                    return e.second == id;
                });
            }
            CATCH_INFO("query " << q << " found " << found << " neighbors");
            CATCH_REQUIRE(found == exact.size());
            for (size_t j = found; j < num_neighbors; ++j) {
                CATCH_REQUIRE(results.index(q, j) == padding.id());
                CATCH_REQUIRE(results.distance(q, j) == padding.distance());
            }
            total += exact.size();
        }
        return total == 0 ? 1.0 : static_cast<double>(hits) / static_cast<double>(total);
    }

  private:
    svs::data::SimpleData<float> data_;
    svs::data::SimpleData<float> queries_;
    std::vector<size_t> ids_;
    std::unordered_map<size_t, size_t> rows_{};
};

/// @brief The checks every filtered search must pass on an index nothing was deleted from.
///
/// ``search(filter)`` returns the results of a filtered search for the reference queries,
/// ``unfiltered`` those of the plain search with the same parameters.
template <typename Distance, typename Search>
void check_filtered_search(
    const FilteredSearchReference<Distance>& reference,
    const Search& search,
    const svs::QueryResult<size_t>& unfiltered
) {
    const auto every_tenth = [](size_t id) { return id % 10 == 0; };
    CATCH_REQUIRE(
        reference.check(search(every_tenth), every_tenth) > reference.min_selective_recall
    );

    // A filter passing everything must not cost recall against the plain search.
    const auto accept_all = [](size_t) { return true; };
    const double unfiltered_recall = reference.check(unfiltered, accept_all);
    CATCH_REQUIRE(
        reference.check(search(accept_all), accept_all) > unfiltered_recall - 0.05
    );

    // Fewer passing labels than requested neighbors: all of them, then padding.
    const auto& ids = reference.ids();
    const auto few = [&](size_t id) {
        return id == ids[100] || id == ids[2000] || id == ids[7000];
    };
    CATCH_REQUIRE(reference.check(search(few), few) == 1.0);

    const auto none = [](size_t) { return false; };
    CATCH_REQUIRE(reference.check(search(none), none) == 1.0);
}

} // namespace svs_test
