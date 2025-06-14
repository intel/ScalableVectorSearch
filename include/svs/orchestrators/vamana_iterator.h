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

// svs-vamana
#include "svs/index/vamana/iterator.h"

// stl
#include <memory>

namespace svs {

/// @brief Type-erased wrapper for the low-level Vamana iterator.
class VamanaIterator {
  private:
    struct Interface {
        virtual svs::index::vamana::VamanaSearchParameters
        parameters_for_current_iteration() const = 0;
        virtual svs::DataType query_type() const = 0;
        virtual size_t batch_number() const = 0;
        virtual size_t size() const = 0;
        virtual std::span<const svs::Neighbor<size_t>> results() const = 0;
        virtual void restart_next_search() = 0;
        virtual void next(
            size_t batch_size,
            const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
        ) = 0;
        virtual bool done() const = 0;
        virtual void update(svs::AnonymousArray<1> newquery) = 0;

        virtual ~Interface() = default;
    };

    template <typename Index, typename QueryType> struct Implementation : Interface {
        // For the type-erased implementation - require the schedule to be type-erased as
        // well.
        using type = svs::index::vamana::BatchIterator<Index, QueryType>;

        Implementation(
            const Index& index,
            std::span<const QueryType> query,
            size_t extra_search_buffer_capacity = svs::UNSIGNED_INTEGER_PLACEHOLDER
        )
            : impl_{index, query, extra_search_buffer_capacity} {}

        svs::index::vamana::VamanaSearchParameters
        parameters_for_current_iteration() const override {
            return impl_.parameters_for_current_iteration();
        }

        svs::DataType query_type() const override { return svs::datatype_v<QueryType>; }

        size_t batch_number() const override { return impl_.batch_number(); }
        size_t size() const override { return impl_.size(); }

        std::span<const svs::Neighbor<size_t>> results() const override {
            return impl_.contents();
        }

        void restart_next_search() override { impl_.restart_next_search(); }
        void next(
            size_t batch_size,
            const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
        ) override {
            impl_.next(batch_size, cancel);
        }
        bool done() const override { return impl_.done(); }

        // Query Updates
        void update(svs::AnonymousArray<1> newquery) override {
            if (newquery.type() == svs::datatype_v<QueryType>) {
                impl_.update(
                    std::span<const QueryType>(get<QueryType>(newquery), newquery.size(0))
                );
            }
        }

        // Member
        type impl_;
    };

    std::unique_ptr<Interface> impl_;

  public:
    /// @brief Construct a new batch iterator for the query over the index.
    ///
    /// Argument ``extra_search_buffer_capacity`` is the extra search buffer capacity to use
    /// for the next search. This is used to ensure that we have few extra neighbors in the
    /// search buffer to accommodate the next search (when not provided,
    /// ``svs::ITERATOR_EXTRA_BUFFER_CAPACITY_DEFAULT = 100`` is used.
    template <typename Index, typename QueryType>
    VamanaIterator(
        const Index& parent,
        std::span<const QueryType> query,
        size_t extra_search_buffer_capacity = svs::UNSIGNED_INTEGER_PLACEHOLDER
    )
        : impl_{std::make_unique<Implementation<Index, QueryType>>(
              parent, query, extra_search_buffer_capacity
          )} {}

    /// @brief Return the search parameters used for the current batch.
    [[nodiscard]] svs::index::vamana::VamanaSearchParameters
    parameters_for_current_iteration() const {
        return impl_->parameters_for_current_iteration();
    }

    /// @brief Return the element type of the captured query.
    [[nodiscard]] svs::DataType query_type() const { return impl_->query_type(); }

    /// @brief Return the current batch number.
    [[nodiscard]] size_t batch_number() const { return impl_->batch_number(); }

    /// @brief Return the number of results for the current batch.
    [[nodiscard]] size_t size() const { return impl_->size(); }

    /// @brief Return a span of the results for the current batch.
    [[nodiscard]] std::span<const svs::Neighbor<size_t>> results() const {
        return impl_->results();
    }

    /// @brief Prepare a new batch of results.
    ///
    /// After calling this method, previous results will no longer be available.
    /// This method invalidates previous values return by ``results()``.
    /// @param batch_size The number of results to return in the next batch.
    ///     In some scenarios (like when all entries are returned or if search is
    ///     cancelled), results size can be lower than the ``batch_size``.
    /// @param cancel A predicate called during the search to determine if the search should
    /// be cancelled.
    ///
    void next(
        size_t batch_size,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        impl_->next(batch_size, cancel);
    }

    /// @brief Signal that the next batch search should begin entirely from scratch.
    ///
    /// The iterator records some internal state to accelerate future calls to ``next()``.
    /// This caching of results may yield slightly different results than beginning index
    /// search completely over from the original entry points.
    ///
    /// Calling this method signals the iterator to abandon its cached state.
    ///
    /// This can be helpful for measuring performance and verifying recall values.
    void restart_next_search() const { impl_->restart_next_search(); }

    /// @brief Returns whether iterator can find more neighbors or not for the given query.
    ///
    /// The iterator is considered done when all the available nodes have been yielded or
    /// when the search can not find any more neighbors. The transition from not done to
    /// done will be triggered by a call to ``next()``. The contents of ``batch_number()``
    /// and ``parameters_for_current_iteration()`` will then remain unchanged by subsequent
    /// invocations of ``next()``.
    bool done() const { return impl_->done(); }

    /// @brief Update the iterator with a new query.
    ///
    template <typename QueryType> void update(std::span<const QueryType> newquery) {
        impl_->update(svs::AnonymousArray<1>{newquery.data(), newquery.size()});
    }
};

} // namespace svs
