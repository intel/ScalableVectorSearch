/*
 * Copyright 2024 Intel Corporation
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
#include "svs/concurrent/dynamic_index.h"
#include "svs/index/index.h"
#include "svs/index/vamana/index.h"
#include "svs/index/vamana/iterator.h"
#include "svs/lib/scopeguard.h"

// stl
#include <unordered_set>
#include <vector>

namespace svs::index::vamana::concurrent {

/// @brief A batch iterator for retrieving neighbors from the index in batches.
///
/// This iterator abstracts the process of retrieving neighbors in fixed-size batches
/// while maintaining internal state for efficient graph traversal.
template <typename Index, typename QueryType>
class BatchIterator
    : public vamana::detail::
          BatchIteratorBase<BatchIterator<Index, QueryType>, Index, QueryType, size_t> {
    using base_type = vamana::detail::
        BatchIteratorBase<BatchIterator<Index, QueryType>, Index, QueryType, size_t>;
    friend base_type;

    using base_type::is_exhausted_;
    using base_type::parent_;
    using base_type::restart_search_;
    using base_type::scratchspace_;
    using base_type::yielded_;

  public:
    using typename base_type::external_id_type;

    /// @brief Constructs a batch iterator for the given query over the index.
    /// @param parent The index to search.
    /// @param query The query data.
    /// @param extra_search_buffer_capacity Additional buffer capacity for the search.
    ///     When not provided, ``svs::ITERATOR_EXTRA_BUFFER_CAPACITY_DEFAULT = 100`` is
    ///     used.
    BatchIterator(
        const Index& parent,
        std::span<const QueryType> query,
        size_t extra_search_buffer_capacity = svs::UNSIGNED_INTEGER_PLACEHOLDER
    )
        : base_type{parent, query, extra_search_buffer_capacity} {}

    /// @brief Adapts an internal neighbor to an external neighbor.
    template <NeighborLike N> svs::Neighbor<external_id_type> adapt(N internal) const {
        if constexpr (Index::needs_id_translation) {
            // `copy_from_scratch` -- our only caller -- runs under
            // `parent_->lock_for_translation()`, so use the non-locking translation:
            // `std::shared_mutex` is not recursive, and re-acquiring it per neighbor would
            // deadlock against a writer that arrives in between.
            return Neighbor<external_id_type>{
                parent_->unsafe_translate_internal_id(internal.id()), internal.distance()};
        } else {
            return internal;
        }
    }

    /// @brief Returns whether iterator can find more neighbors or not for the given query.
    ///
    /// The iterator is considered done when all the available nodes have been yielded or
    /// when the search can not find any more neighbors. Deleted IDs in the yielded set
    /// do not count as coverage of the current live IDs.
    bool done() const {
        if (is_exhausted_) {
            return true;
        }
        if (yielded_.size() < parent_->size()) {
            return false;
        }
        bool all_yielded = true;
        parent_->on_ids([&](size_t id) {
            all_yielded = all_yielded && yielded_.contains(id);
        });
        return all_yielded;
    }

  private:
    auto begin_search() {
        // Keep internal IDs stable through traversal, reranking, and translation.
        [[maybe_unused]] auto search_guard = parent_->lock_storage_shared();
        const auto epoch = parent_->compaction_epoch_;
        if (compaction_epoch_ != epoch) {
            // Reranking touches cached IDs before the restart initializer runs.
            // Preserve the grown window so the restart can reach unreturned neighbors.
            scratchspace_.buffer.clear();
            restart_search_ = true;
            compaction_epoch_ = epoch;
        }
        return search_guard;
    }

    auto lock_for_translation() const {
        // Hold the translation lock (translator_mutex_) so a
        // concurrent consolidate/compact cannot erase or remap entries mid-copy.
        return parent_->lock_for_translation();
    }

    template <NeighborLike N> auto prepare_neighbor(N internal) const {
        return adapt(internal);
    }

    template <NeighborLike N> bool accept_neighbor(const N& neighbor) const {
        // Erased by a concurrent consolidation.
        if (neighbor.id() == no_external_id) {
            return false;
        }
        return true;
    }

    template <NeighborLike N> N finish_neighbor(N neighbor) const { return neighbor; }

    template <typename... Args> static void run_search(Args&&... args) {
        concurrent::greedy_search(SVS_FWD(args)...);
    }

    size_t compaction_epoch_ = 0; // Epoch of scratchspace_ IDs.
};

// Deduction Guides
template <typename Index, typename QueryType>
BatchIterator(const Index*, std::span<const QueryType>) -> BatchIterator<Index, QueryType>;

} // namespace svs::index::vamana::concurrent
