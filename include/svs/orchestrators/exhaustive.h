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

// Pull in flat-index related functionaliry
#include "svs/index/flat/flat.h"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/threads.h"

#include "svs/orchestrators/manager.h"

// stdlib
#include <memory>

namespace svs {

/////
///// Type-erased stack implementation.
/////

// Additional API exposed by exhaustive search.
class FlatInterface : public manager::ManagerInterface {
  public:
    // Batch size adjustment
    virtual void set_data_batch_size(size_t batch_size) = 0;
    virtual size_t get_data_batch_size() const = 0;
    virtual void set_query_batch_size(size_t batch_size) = 0;
    virtual size_t get_query_batch_size() const = 0;
};

template <typename QueryType, typename Impl, typename IFace = FlatInterface>
class FlatImpl : public manager::ManagerImpl<QueryType, Impl, FlatInterface> {
  public:
    using base_type = manager::ManagerImpl<QueryType, Impl, FlatInterface>;
    using base_type::impl;

    ///
    /// Construct a FlatImpl implementation from the implementation's move constructor.
    ///
    explicit FlatImpl(Impl impl)
        : base_type{std::move(impl)} {}

    ///
    /// Construct a FlatImpl implementation by calling its actual implementation's
    /// constructor directly.
    ///
    template <typename... Args>
    explicit FlatImpl(Args&&... args)
        : base_type{std::forward<Args>(args)...} {}

    // Batch size management.
    void set_data_batch_size(size_t batch_size) override {
        impl().set_data_batch_size(batch_size);
    }
    size_t get_data_batch_size() const override { return impl().get_data_batch_size(); }

    void set_query_batch_size(size_t batch_size) override {
        impl().set_query_batch_size(batch_size);
    }
    size_t get_query_batch_size() const override { return impl().get_query_batch_size(); }
};

// Forward Declarations
class Flat;
template <typename QueryType, typename... Args> Flat make_flat(Args&&... args);

/// @brief Type erased container for the Flat index.
class Flat : public manager::IndexManager<FlatInterface, FlatImpl> {
  public:
    /// Internal dispatch tag.
    struct AssembleTag {};
    using base_type = manager::IndexManager<FlatInterface, FlatImpl>;

    template <typename Impl>
    explicit Flat(std::unique_ptr<Impl> impl)
        : base_type{std::move(impl)} {}

    ///// Flat interface

    Flat& set_data_batch_size(size_t batch_size) {
        impl_->set_data_batch_size(batch_size);
        return *this;
    }

    size_t get_data_batch_size() const { return impl_->get_data_batch_size(); }

    Flat& set_query_batch_size(size_t batch_size) {
        impl_->set_query_batch_size(batch_size);
        return *this;
    }

    size_t get_query_batch_size() const { return impl_->get_query_batch_size(); }

    ///// Loading

    ///
    /// @brief Load a Flat Index from an existing dataset.
    ///
    /// @tparam QueryType The element type of the vectors that will be used for querying.
    ///
    /// @param data_loader A compatible class capable of load data. See expanded notes.
    /// @param distance A distance functor to use or a ``svs::DistanceType`` enum.
    /// @param num_threads The number of threads to use for index searches.
    ///
    /// @copydoc hidden_flat_auto_assemble
    ///
    template <typename QueryType, typename DataLoader, typename Distance>
    static Flat assemble(DataLoader&& data_loader, Distance distance, size_t num_threads) {
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            auto dispatcher = DistanceDispatcher{distance};
            return dispatcher([&, num_threads](auto distance_function) {
                return make_flat<QueryType>(
                    AssembleTag(),
                    std::forward<DataLoader>(data_loader),
                    std::move(distance_function),
                    num_threads
                );
            });
        } else {
            return make_flat<QueryType>(
                AssembleTag(),
                std::forward<DataLoader>(data_loader),
                std::move(distance),
                num_threads
            );
        }
    }
};

template <typename QueryType, typename... Args> Flat make_flat(Args&&... args) {
    using Impl = decltype(index::flat::FlatIndex(std::forward<Args>(args)...));
    return Flat{std::make_unique<FlatImpl<QueryType, Impl>>(std::forward<Args>(args)...)};
}

template <typename QueryType, typename... Args>
Flat make_flat(Flat::AssembleTag SVS_UNUSED(tag) /*unused*/, Args&&... args) {
    return make_flat<QueryType>(index::flat::auto_assemble(std::forward<Args>(args)...));
}
} // namespace svs
