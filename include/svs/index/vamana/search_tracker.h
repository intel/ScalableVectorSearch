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
