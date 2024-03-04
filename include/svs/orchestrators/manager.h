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

#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/core/query_result.h"
#include "svs/lib/datatype.h"
#include "svs/lib/threads/threadpool.h"

#include "svs/index/index.h"

#include <concepts>
#include <memory>
#include <numeric>
#include <type_traits>

namespace svs::manager {

///
/// Top level Manager
///

template <typename IFace> class ManagerInterface : public IFace {
  public:
    static_assert(
        std::is_abstract_v<IFace> || std::is_empty_v<IFace>,
        "Manager Interface must be templated with only pure abstract parameters!"
    );

    // Default constructor. Not much needed for pure-abstract classes.
    ManagerInterface() = default;

    // Interface specific search parameters.
    using search_parameters_type = typename IFace::search_parameters_type;

    // Search interface
    virtual search_parameters_type get_search_parameters() const = 0;
    virtual void set_search_parameters(const search_parameters_type&) = 0;

    virtual void search(
        svs::QueryResultView<size_t> results,
        AnonymousArray<2> data,
        const search_parameters_type& search_parameters
    ) = 0;

    // Data Interface
    virtual size_t size() const = 0;
    virtual size_t dimensions() const = 0;

    // Threading interface
    virtual bool can_change_threads() const = 0;
    virtual size_t get_num_threads() const = 0;
    virtual void set_num_threads(size_t) = 0;

    // Delete the special member functions.
    ManagerInterface(const ManagerInterface&) = delete;
    ManagerInterface& operator=(const ManagerInterface&) = delete;
    ManagerInterface(ManagerInterface&&) = delete;
    ManagerInterface& operator=(ManagerInterface&&) = delete;

    virtual ~ManagerInterface() = default;
};

///
/// The base implementation for types meant to implement polymorphic Manager interface.
/// The goal of this type is to wrap a concrete implementation of type `T` with the
///
template <typename QueryType, typename Impl, typename IFace>
class ManagerImpl : public ManagerInterface<IFace> {
  public:
    // Inherit the search parameters type from the interface.
    using search_parameters_type = typename IFace::search_parameters_type;

    explicit ManagerImpl(Impl implementation)
        : ManagerInterface<IFace>{}
        , implementation_{std::move(implementation)} {}

    ///
    /// Construct a `ManagerImpl` piecewise by calling it's implementation's constructor
    /// directly.
    ///
    template <typename... Args>
    explicit ManagerImpl(Args&&... args)
        : ManagerInterface<IFace>{}
        , implementation_{std::forward<Args>(args)...} {}

    /// Return the default search parameters for the index.
    search_parameters_type get_search_parameters() const override {
        return implementation_.get_search_parameters();
    }

    /// Set the default search parameters for the index.
    void set_search_parameters(const search_parameters_type& search_parameters) override {
        implementation_.set_search_parameters(search_parameters);
    }

    // Search Interface
    // At this level - simply dispatch over the supported query types.
    void search(
        QueryResultView<size_t> result,
        AnonymousArray<2> data,
        const search_parameters_type& search_parameters
    ) override {
        // TODO (Mark) For now - only allow implementations to support a single query
        // type.
        //
        // Generalizing this to multiple query types will require some metaprogramming
        // dances.
        if (data.type() == datatype_v<QueryType>) {
            const auto view = data::ConstSimpleDataView<QueryType>(data);
            svs::index::search_batch_into_with(
                implementation_, result, view, search_parameters
            );
        } else {
            throw ANNEXCEPTION(
                "Unsupported datatype! Got: {}. Expected: {}.",
                data.type(),
                (datatype_v<QueryType>)
            );
        }
    }

    // Data Interface
    size_t size() const override { return implementation_.size(); }
    size_t dimensions() const override { return implementation_.dimensions(); }

    // Threading interface.
    bool can_change_threads() const override {
        return implementation_.can_change_threads();
    }
    size_t get_num_threads() const override { return implementation_.get_num_threads(); }
    void set_num_threads(size_t num_threads) override {
        implementation_.set_num_threads(num_threads);
    }

  protected:
    Impl& impl() { return implementation_; }
    const Impl& impl() const { return implementation_; }

  private:
    Impl implementation_;
};

///
/// @brief Do I need to document this also?
///
template <typename IFace> class IndexManager {
  public:
    /// Require the presence of the `search_parameters_type` alias in the instantiating
    /// interface.
    using search_parameters_type = typename IFace::search_parameters_type;

    template <typename Impl>
        requires std::is_base_of_v<ManagerInterface<IFace>, Impl>
    explicit IndexManager(std::unique_ptr<Impl> impl)
        : impl_{std::move(impl)} {}

    search_parameters_type get_search_parameters() const {
        return impl_->get_search_parameters();
    }

    void set_search_parameters(const search_parameters_type& search_parameters) {
        impl_->set_search_parameters(search_parameters);
    }

    template <typename QueryType>
    void search(
        QueryResultView<size_t> result,
        data::ConstSimpleDataView<QueryType> queries,
        const search_parameters_type& search_parameters
    ) {
        impl_->search(result, AnonymousArray<2>(queries), search_parameters);
    }

    // This is an API compatibility trick.
    // If called with just the queries and number of neighbors - bounce into the dispatch
    // pipeline which will end-up calling the appropriate search above.
    template <typename Queries>
    QueryResult<size_t> search(const Queries& queries, size_t num_neighbors) {
        return svs::index::search_batch(*this, queries.cview(), num_neighbors);
    }

    ///// Data Interface

    /// @brief Return the number of elements in the indexed dataset.
    size_t size() const { return impl_->size(); }

    /// @brief Return the logical number of dimensions of each vector in the indexed
    /// dataset.
    size_t dimensions() const { return impl_->dimensions(); }

    ///// Threading Interface

    ///
    /// @brief Return whether the back-end implementation can change the number of threads.
    ///
    bool can_change_threads() const { return impl_->can_change_threads(); }

    ///
    /// @brief Return the current number of worker threads used by this index for searches.
    ///
    size_t get_num_threads() const { return impl_->get_num_threads(); }

    ///
    /// @brief Set the number of threads to use for searching.
    ///
    /// @param num_threads The number of threads to use. If set to ``0``, will implicitly
    ///     default to ``1``.
    ///
    /// Only effective if ``can_change_threads()`` returns ``true``.
    ///
    void set_num_threads(size_t num_threads) {
        impl_->set_num_threads(std::max(size_t(1), num_threads));
    }

    // The implementation is `protected` instead of private because derived classes
    // should extent the interface provided by the base Manager.
    //
    // TODO: This could be exposed via a `get_impl()` accessor - but is that really
    // any different than just grabbing the interface pointer directly?
  protected:
    std::unique_ptr<ManagerInterface<IFace>> impl_;
};

} // namespace svs::manager
