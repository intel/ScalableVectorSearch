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

#include "svs/index/vamana/dynamic_search_buffer.h"

namespace svs::index::vamana::concurrent {

///
/// A search buffer that allows entries to be predicated out.
/// The search buffer will still navigate through those entries, but it won't include
/// them in its final result.
///
template <typename Idx, typename Cmp = std::less<>>
class MutableBuffer : public vamana::MutableBuffer<Idx, Cmp> {
    using base_type = vamana::MutableBuffer<Idx, Cmp>;

  public:
    using base_type::base_type;

    MutableBuffer() = default;

    /// Copy the portions of the MutableBuffer that matter for the purposes of scratch
    /// space.
    ///
    /// Preserves the sizes of various containers but not necessarily the values.
    MutableBuffer shallow_copy() const {
        // We don't care about the contents of the buffer - just its size.
        // Therefore, we can construct a new buffer from scratch.
        return MutableBuffer{
            base_type::config(), this->compare_, base_type::visited_set_enabled()};
    }

    // Sort all stored elements in the buffer.
    void sort() {
        assert(base_type::valid() <= this->valid_capacity_);

        // Put the elements in ascending order.
        std::sort(base_type::begin(), base_type::end(), this->compare_);

        // Now - fix our invariants.
        if (this->valid_ < this->target_valid_) {
            // Maintain 5A
            // Invariant 6 has not activated.
            this->roi_end_ = base_type::size();

            // Maintain best_unvisited_ invariant.
            // Recompute it from scratch, mirroring the 5B branch below.
            for (this->best_unvisited_ = 0; this->best_unvisited_ < this->roi_end_;
                 ++this->best_unvisited_) {
                if (!this->candidates_[this->best_unvisited_].visited()) {
                    break;
                }
            }
            return;
        }

        // Maintain 5B
        size_t valid_count = 0;
        for (size_t i = 0; i < base_type::size(); ++i) {
            if (this->candidates_[i].valid()) {
                ++valid_count;
                if (valid_count == this->target_valid_) {
                    // One past the last valid element.
                    this->roi_end_ = i + 1;
                    break;
                }
            }
        }

        // TODO: Switch over to using iterators for the return values to avoid this.
        // Maintain best_unvisited_ invariant
        // cleanup() and std::sort() may invalidate best_unvisited_.
        // so, have to lookup the next unvisited from scratch.
        for (this->best_unvisited_ = 0; this->best_unvisited_ < this->roi_end_;
             ++this->best_unvisited_) {
            if (!this->candidates_[this->best_unvisited_].visited()) {
                break;
            }
        }

        // Check if invariant 6 is active.
        // If so, drop invalid elements off the end until a valid element is found.
        if (base_type::slack() == 0) {
            assert(!this->candidates_.empty());
            while (!base_type::back().valid()) {
                this->candidates_.pop_back();
                assert(!this->candidates_.empty());
            }
        }
    }
};

} // namespace svs::index::vamana::concurrent
