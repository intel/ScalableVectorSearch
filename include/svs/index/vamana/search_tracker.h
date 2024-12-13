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

#pragma once

#include "svs/lib/neighbor.h"

#include "tsl/robin_set.h"

#include <type_traits>

namespace svs::index::vamana {

template <typename Idx>
    requires std::is_unsigned_v<Idx>
class SearchTracker {
  public:
    SearchTracker()
        : accessed_points_{100}
        , accessed_search_neighbors_{100}
        , n_distance_computations_{0} {}

    // Satisfy the `GreedySearchTracker` concept.
    void visited(SearchNeighbor<Idx> neighbor, size_t n_computations) {
        add_visited_point(neighbor.id_);
        add_visited_neighbor(neighbor);
        add_distance_computations(n_computations);
    }

    void add_distance_computations(size_t n = 1) { n_distance_computations_ += n; }

    void add_visited_point(Idx idx) { accessed_points_.insert(idx); }

    void add_visited_neighbor(Neighbor<Idx> pair) {
        accessed_search_neighbors_.insert(pair);
    }

    size_t n_distance_computations() const { return n_distance_computations_; }

    const tsl::robin_set<Idx>& accessed_points() const { return accessed_points_; }

    const tsl::robin_set<Neighbor<Idx>>& accessed_search_neighbors() const {
        return accessed_search_neighbors_;
    }

  private:
    tsl::robin_set<Idx> accessed_points_;
    tsl::robin_set<Neighbor<Idx>, IDHash, IDEqual> accessed_search_neighbors_;
    size_t n_distance_computations_;
};
} // namespace svs::index::vamana
