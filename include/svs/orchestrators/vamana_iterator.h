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
#include "svs/index/vamana/iterator_schedule.h"

// stl
#include <memory>

namespace svs {

/// @brief Type-erased wrapper for the low-level Vamana iterator.
class VamanaIterator {
  private:
    struct Interface {
        virtual svs::index::vamana::VamanaSearchParameters
        parameters_for_current_batch() const = 0;
        virtual svs::DataType query_type() const = 0;
        virtual size_t batch() const = 0;
        virtual size_t size() const = 0;
        virtual std::span<const svs::Neighbor<size_t>> results() const = 0;
        virtual void restart_next_search() = 0;
        virtual void next() = 0;
        virtual bool done() const = 0;
        virtual void update(svs::AnonymousArray<1>) = 0;
        virtual void
        update(svs::AnonymousArray<1>, const svs::index::vamana::AbstractIteratorSchedule&) = 0;

        virtual ~Interface() = default;
    };

    template <typename Index, typename QueryType> struct Implementation : Interface {
        // For the type-erased implementation - require the schedule to be type-erased as
        // well.
        using type = svs::index::vamana::
            BatchIterator<Index, QueryType, svs::index::vamana::AbstractIteratorSchedule>;

        Implementation(
            const Index& index,
            std::span<const QueryType> query,
            svs::index::vamana::AbstractIteratorSchedule schedule
        )
            : impl_{index, query, std::move(schedule)} {}

        svs::index::vamana::VamanaSearchParameters
        parameters_for_current_batch() const override {
            return impl_.parameters_for_current_iteration();
        }

        svs::DataType query_type() const override { return svs::datatype_v<QueryType>; }

        size_t batch() const override { return impl_.batch(); }
        size_t size() const override { return impl_.size(); }

        std::span<const svs::Neighbor<size_t>> results() const override {
            return impl_.contents();
        }

        void restart_next_search() override { impl_.restart_next_search(); }
        void next() override { impl_.next(); }
        bool done() const override { return impl_.done(); }

        // Query Updates
        void update(svs::AnonymousArray<1> newquery) override {
            if (newquery.type() == svs::datatype_v<QueryType>) {
                impl_.update(
                    std::span<const QueryType>(get<QueryType>(newquery), newquery.size(0))
                );
            }
        }

        void update(
            svs::AnonymousArray<1> newquery,
            const svs::index::vamana::AbstractIteratorSchedule& newschedule
        ) override {
            if (newquery.type() == svs::datatype_v<QueryType>) {
                impl_.update(
                    std::span<const QueryType>(get<QueryType>(newquery), newquery.size(0)),
                    newschedule
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
    /// The results for the first batch will be available once the constructor completes.
    ///
    /// Argument ``schedule`` will be used to adjust the search parameters for each batch.
    /// An internal copy of the query will be made.
    template <typename Index, typename QueryType>
    VamanaIterator(
        const Index& parent,
        std::span<const QueryType> query,
        svs::index::vamana::AbstractIteratorSchedule schedule
    )
        : impl_{std::make_unique<Implementation<Index, QueryType>>(
              parent, query, std::move(schedule)
          )} {}

    /// @brief Return the search parameters used for the current batch.
    [[nodiscard]] svs::index::vamana::VamanaSearchParameters
    parameters_for_current_batch() const {
        return impl_->parameters_for_current_batch();
    }

    /// @brief Return the element type of the captured query.
    [[nodiscard]] svs::DataType query_type() const { return impl_->query_type(); }

    /// @brief Return the current batch number.
    [[nodiscard]] size_t batch() const { return impl_->batch(); }

    /// @brief Return the number of results for the current batch.
    [[nodiscard]] size_t size() const { return impl_->size(); }

    /// @brief Return a span of the results for the current batch.
    [[nodiscard]] std::span<const svs::Neighbor<size_t>> results() const {
        return impl_->results();
    }

    /// @brief Retrieve a new batch of results.
    ///
    /// After calling this method, previous results will no longer be available.
    /// This method invalidates previous values return by ``results()``.
    void next() { impl_->next(); }

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

    /// @brief Return whether or not all entries in the index have been yielded.
    ///
    /// This transition is triggered by an invocation of ``next()``.
    /// After ``done() == true``, future calls to ``next()`` will yield an empty set of
    /// candidates and will not change the results of ``batch()`` or
    /// ``parameters_for_current_batch()``.
    bool done() const { return impl_->done(); }

    /// @brief Update the query contained in the iterator and begin a new search.
    ///
    ///
    /// Thus function provides the following exception guarentee:
    ///
    /// * If the new search fails, the state of the iterator is valid for the old query.
    /// * If an exception is thrown after search, the contents of the iterator are valid
    ///   for the new query.
    ///
    /// Throws an ``svs::ANNException`` if the provided query type does not match
    /// ``query_type()``.
    template <typename QueryType> void update(std::span<const QueryType> newquery) {
        impl_->update(svs::AnonymousArray<1>{newquery.data(), newquery.size()}, newquery);
    }

    /// @brief Update the query contained in the iterator and begin a new search.
    ///
    /// Thus function provides the following exception guarentee:
    ///
    /// * If the new search fails, the state of the iterator is valid for the old query.
    /// * If an exception is thrown after search, the contents of the iterator are valid
    ///   for the new query.
    /// * The copy constructor for ``newschedule`` may be invoked. If it is and it throws
    ///   an exception, the state of the iterator is valid for the old query.
    ///
    /// Throws an ``svs::ANNException`` if the provided query type does not match
    /// ``query_type()``.
    template <typename QueryType, svs::index::vamana::IteratorSchedule Schedule>
    void update(std::span<const QueryType> newquery, const Schedule& newschedule) {
        impl_->update(
            svs::AnonymousArray<1>{newquery.data(), newquery.size()},
            svs::index::vamana::AbstractIteratorSchedule(newschedule)
        );
    }
};

} // namespace svs
