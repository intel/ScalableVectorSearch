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

#include "svs/runtime/api_defs.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svs::runtime {

// Predict how many more items need to be processed to reach the goal,
// based on the observed hit rate so far.
// If no hits yet, returns `hint` unchanged.
// Result is capped at `max_value` (e.g., number of vectors in the index).
inline size_t predict_further_processing(
    size_t processed, size_t hits, size_t goal, size_t hint, size_t max_value
) {
    if (hits == 0 || hits >= goal) {
        return std::min(hint, max_value);
    }
    float batch_size = static_cast<float>(goal - hits) * processed / hits;
    return std::min(std::max(static_cast<size_t>(batch_size), size_t{1}), max_value);
}

// Check if the filtered search should stop early based on the observed hit rate.
// Returns true if the hit rate is below the threshold, meaning the caller should
// give up and let the caller fall back to exact search.
inline bool
should_stop_filtered_search(size_t total_checked, size_t found, float filter_stop) {
    if (filter_stop <= 0 || total_checked == 0) {
        return false;
    }
    if (found == 0) {
        return true;
    }
    float hit_rate = static_cast<float>(found) / total_checked;
    return hit_rate < filter_stop;
}

template <typename Neighbor> class GroupedBest {
  public:
    explicit GroupedBest(size_t reserve) { items_.reserve(reserve); }

    size_t size() const { return items_.size(); }

    template <typename Compare>
    void offer(const Neighbor& candidate, size_t group, const Compare& closer) {
        if (group == IDGrouper::NO_GROUP) {
            items_.push_back(candidate);
            return;
        }
        auto [slot, inserted] = slots_.try_emplace(group, items_.size());
        if (inserted) {
            items_.push_back(candidate);
        } else if (closer(candidate.distance(), items_[slot->second].distance())) {
            items_[slot->second] = candidate;
        }
    }

    std::vector<Neighbor> take() {
        auto out = std::move(items_);
        items_.clear();
        slots_.clear();
        return out;
    }

  private:
    std::vector<Neighbor> items_;
    std::unordered_map<size_t, size_t> slots_;
};

// Batches are sorted individually, not across next() calls, so each is consumed in full.
template <typename BatchIterator, typename Compare>
auto grouped_topk(
    BatchIterator& iterator,
    size_t k,
    IDFilter* filter,
    float filter_stop,
    const IDGrouper& grouper,
    const Compare& closer,
    size_t initial_batch_size,
    size_t max_batch_size
) {
    using Neighbor = std::remove_cvref_t<decltype(*std::begin(iterator.results()))>;
    GroupedBest<Neighbor> best(k);
    size_t total_checked = 0;
    size_t filter_hits = 0;
    auto batch_size = initial_batch_size;
    do {
        batch_size = predict_further_processing(
            total_checked, best.size(), k, batch_size, max_batch_size
        );
        iterator.next(batch_size);
        total_checked += iterator.size();
        for (const auto& neighbor : iterator.results()) {
            if (filter != nullptr && !filter->is_member(neighbor.id())) {
                continue;
            }
            filter_hits++;
            best.offer(neighbor, grouper.get_group(neighbor.id()), closer);
        }
        if (best.size() < k &&
            should_stop_filtered_search(total_checked, filter_hits, filter_stop)) {
            return std::vector<Neighbor>{};
        }
    } while (best.size() < k && !iterator.done());

    auto out = best.take();
    std::stable_sort(out.begin(), out.end(), [&](const auto& a, const auto& b) {
        return closer(a.distance(), b.distance());
    });
    if (out.size() > k) {
        out.resize(k);
    }
    return out;
}

} // namespace svs::runtime
