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

///
/// @defgroup orchestrator_vamana_entry Main API for the Vamana orchestrator.
/// @brief Main API for the Vamana type-erased orchestrator.
///

#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/core/medioid.h"
#include "svs/index/vamana/index.h"
#include "svs/index/vamana/vamana_build.h"
#include "svs/lib/datatype.h"
#include "svs/lib/readwrite.h"
#include "svs/lib/threads.h"
#include "svs/orchestrators/manager.h"
#include "svs/orchestrators/vamana_iterator.h"

// stdlib
#include <filesystem>
#include <string>
#include <string_view>

namespace svs {

class VamanaInterface {
  public:
    using search_parameters_type = svs::index::vamana::VamanaSearchParameters;

    virtual void set_alpha(float alpha) = 0;
    virtual float get_alpha() const = 0;

    virtual size_t get_graph_max_degree() const = 0;

    virtual void set_construction_window_size(size_t window_size) = 0;
    virtual size_t get_construction_window_size() const = 0;

    virtual void set_max_candidates(size_t max_candidates) = 0;
    virtual size_t get_max_candidates() const = 0;

    virtual void set_prune_to(size_t prune_to) = 0;
    virtual size_t get_prune_to() const = 0;

    virtual void set_full_search_history(bool enable) = 0;
    virtual bool get_full_search_history() const = 0;

    ///// Backend Information Interface
    virtual std::string experimental_backend_string() const = 0;

    // Saving
    virtual void save(
        const std::filesystem::path& config_dir,
        const std::filesystem::path& graph_dir,
        const std::filesystem::path& data_dir
    ) = 0;

    ///// Reconstruction
    // TODO: Allow threadpools to be const-invocable.
    virtual void
    reconstruct_at(data::SimpleDataView<float> dst, std::span<const uint64_t> ids) = 0;

    ///// Iterator
    virtual VamanaIterator batch_iterator(
        svs::AnonymousArray<1> query, svs::index::vamana::AbstractIteratorSchedule schedule
    ) const = 0;

    ///// Calibrations
    virtual index::vamana::VamanaSearchParameters experimental_calibrate(
        ConstErasedPointer queries,
        size_t query_size_0,
        size_t query_size_1,
        ConstErasedPointer groundtruth,
        size_t groundtruth_size_0,
        size_t groundtruth_size_1,
        size_t num_neighbors,
        double target_recall,
        const index::vamana::CalibrationParameters& calibration_parameters
    ) = 0;
    virtual void reset_performance_parameters() = 0;
};

template <lib::TypeList QueryTypes, typename Impl, typename IFace = VamanaInterface>
class VamanaImpl : public manager::ManagerImpl<QueryTypes, Impl, IFace> {
  private:
    // Null-terimated array of characters.
    static constexpr auto typename_impl = lib::generate_typename<Impl>();

  public:
    // type aliases
    using base_type = manager::ManagerImpl<QueryTypes, Impl, IFace>;
    using base_type::impl;
    using VamanaSearchParameters = index::vamana::VamanaSearchParameters;

    explicit VamanaImpl(Impl impl)
        : base_type{std::move(impl)} {}

    template <typename... Args>
    explicit VamanaImpl(Args&&... args)
        : base_type{std::forward<Args>(args)...} {}

    void set_alpha(float alpha) override { impl().set_alpha(alpha); }
    float get_alpha() const override { return impl().get_alpha(); }

    size_t get_graph_max_degree() const override { return impl().get_graph_max_degree(); }

    void set_construction_window_size(size_t window_size) override {
        impl().set_construction_window_size(window_size);
    }
    size_t get_construction_window_size() const override {
        return impl().get_construction_window_size();
    }

    void set_max_candidates(size_t max_candidates) override {
        impl().set_max_candidates(max_candidates);
    }
    size_t get_max_candidates() const override { return impl().get_max_candidates(); }

    void set_prune_to(size_t prune_to) override { impl().set_prune_to(prune_to); }
    size_t get_prune_to() const override { return impl().get_prune_to(); }

    void set_full_search_history(bool enable) override {
        impl().set_full_search_history(enable);
    }
    bool get_full_search_history() const override {
        return impl().get_full_search_history();
    }

    ///// Backend Information Interface
    std::string experimental_backend_string() const override {
        return std::string{typename_impl.begin(), typename_impl.end() - 1};
    }

    // Saving.
    void save(
        const std::filesystem::path& config_dir,
        const std::filesystem::path& graph_dir,
        const std::filesystem::path& data_dir
    ) override {
        // TODO (MH): Add a compile-time switch to select whether we want saving failure
        // to be a compile-time or run-time error.
        //
        // For development, we want it to be run-time to allow us to construct indexes
        // that don't fully support saving.
        //
        // However, for deployment, it would be nice to turn it into a compile error to
        // catch any weird corner cases.
        if constexpr (Impl::supports_saving) {
            impl().save(config_dir, graph_dir, data_dir);
        } else {
            throw ANNEXCEPTION("The current Vamana backend doesn't support saving!");
        }
    }

    ///// Reconstruction
    void reconstruct_at(data::SimpleDataView<float> data, std::span<const uint64_t> ids)
        override {
        impl().reconstruct_at(data, ids);
    }

    ///// Iterator
    SVS_TEMPORARY_DISABLE_SINGLE_SEARCH VamanaIterator batch_iterator(
        svs::AnonymousArray<1> query, svs::index::vamana::AbstractIteratorSchedule schedule
    ) const override {
        if constexpr (!Impl::temporary_disable_batch_iterator()) {
            // Match the query type.
            return svs::lib::match(
                QueryTypes{},
                query.type(),
                [&]<typename T>(svs::lib::Type<T> SVS_UNUSED(type)) {
                    return VamanaIterator{
                        impl(),
                        std::span<const T>(svs::get<T>(query), query.size(0)),
                        std::move(schedule)};
                }
            );
        } else {
            // TODO: Enable single-search for LeanVec to remove this run-time error.
            throw ANNEXCEPTION("The current index backend does not support batch iteration."
            );
        }
    }

    ///// Calibration
    VamanaSearchParameters experimental_calibrate(
        ConstErasedPointer queries,
        size_t query_size_0,
        size_t query_size_1,
        ConstErasedPointer groundtruth,
        size_t groundtruth_size_0,
        size_t groundtruth_size_1,
        size_t num_neighbors,
        double target_recall,
        const index::vamana::CalibrationParameters& calibration_parameters
    ) override {
        if (!lib::in(queries.type(), QueryTypes{})) {
            throw ANNEXCEPTION(
                "Unsupported query type! Got {}, expected {}.",
                queries.type(),
                fmt::join(QueryTypes::data_types(), ", ")
            );
        }
        if (groundtruth.type() != DataType::uint32) {
            throw ANNEXCEPTION(
                "Unsupported groundtruth type! Got {}, expected {}.",
                groundtruth.type(),
                DataType::uint32
            );
        }

        // Reassemble and call the real implementation.
        return lib::match(
            QueryTypes{},
            queries.type(),
            [&]<typename Q>(lib::Type<Q> SVS_UNUSED(tag)) {
                return impl().calibrate(
                    data::ConstSimpleDataView<Q>(
                        get<Q>(queries), query_size_0, query_size_1
                    ),
                    data::ConstSimpleDataView<uint32_t>(
                        get<uint32_t>(groundtruth), groundtruth_size_0, groundtruth_size_1
                    ),
                    num_neighbors,
                    target_recall,
                    calibration_parameters
                );
            }
        );
    }

    void reset_performance_parameters() override { impl().reset_performance_parameters(); }
};

///// Forward declarations

// Type-erased wrapper around the VamanaIndex
class Vamana;
// Deducing constructor.
template <lib::TypeList QueryTypes, typename... Args> Vamana make_vamana(Args&&... args);

///
/// Vamana Manager
///

///
/// @ingroup orchestrator_vamana_entry
/// @brief Type erased container for the Vamana index.
///
class Vamana : public manager::IndexManager<VamanaInterface> {
  public:
    using base_type = manager::IndexManager<VamanaInterface>;
    using VamanaSearchParameters = index::vamana::VamanaSearchParameters;

    /// @private
    struct BuildTag {};
    /// @private
    struct AssembleTag {};

    explicit Vamana(std::unique_ptr<manager::ManagerInterface<VamanaInterface>> impl)
        : base_type{std::move(impl)} {}

    void experimental_reset_performance_parameters() {
        impl_->reset_performance_parameters();
    }

    ///// Vamana Interface.

    Vamana& set_search_window_size(size_t search_window_size) {
        auto parameters = get_search_parameters();
        parameters.buffer_config_ = index::vamana::SearchBufferConfig{search_window_size};
        set_search_parameters(parameters);
        return *this;
    }

    size_t get_search_window_size() const {
        return get_search_parameters().buffer_config_.get_search_window_size();
    }

    /// @copydoc svs::index::vamana::VamanaIndex::get_alpha
    float get_alpha() const { return impl_->get_alpha(); }
    void set_alpha(float alpha) { impl_->set_alpha(alpha); }

    /// @copydoc svs::index::vamana::VamanaIndex::get_graph_max_degree
    size_t get_graph_max_degree() const { return impl_->get_graph_max_degree(); }

    /// @copydoc svs::index::vamana::VamanaIndex::set_construction_window_size
    size_t get_construction_window_size() const {
        return impl_->get_construction_window_size();
    }
    void set_construction_window_size(size_t window_size) {
        impl_->set_construction_window_size(window_size);
    }

    /// @copydoc svs::index::vamana::VamanaIndex::get_max_candidates
    size_t get_max_candidates() const { return impl_->get_max_candidates(); }
    void set_max_candidates(size_t max_candidates) {
        impl_->set_max_candidates(max_candidates);
    }

    /// @copydoc svs::index::vamana::VamanaIndex::get_prune_to
    size_t get_prune_to() const { return impl_->get_prune_to(); }
    void set_prune_to(size_t prune_to) { impl_->set_prune_to(prune_to); }

    /// @copydoc svs::index::vamana::VamanaIndex::get_full_search_history
    bool get_full_search_history() const { return impl_->get_full_search_history(); }
    void set_full_search_history(bool enable) { impl_->set_full_search_history(enable); }

    bool visited_set_enabled() const {
        return get_search_parameters().search_buffer_visited_set_;
    }
    void enable_visited_set() {
        auto parameters = get_search_parameters();
        parameters.search_buffer_visited_set_ = true;
        set_search_parameters(parameters);
    }
    void disable_visited_set() {
        auto parameters = get_search_parameters();
        parameters.search_buffer_visited_set_ = false;
        set_search_parameters(parameters);
    }

    std::string experimental_backend_string() const {
        return impl_->experimental_backend_string();
    }

    ///
    /// @copydoc svs::index::vamana::VamanaIndex::save
    ///
    /// @sa assemble, build
    ///
    void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& graph_directory,
        const std::filesystem::path& data_directory
    ) {
        impl_->save(config_directory, graph_directory, data_directory);
    }

    void reconstruct_at(data::SimpleDataView<float> data, std::span<const uint64_t> ids) {
        impl_->reconstruct_at(data, ids);
    }

    ///
    /// @brief Load a Vamana Index from a previously saved index.
    ///
    /// @tparam QueryType The element type of queries that will be used when requesting
    ///     searches over the index.
    ///
    /// @param config_path Path to the directory where the index configuration was saved.
    ///     This corresponds to the ``config_dir`` argument of ``svs::Vamana::save``.
    /// @param graph_loader The loader for the graph to use. See ``svs::GraphLoader``.
    ///     The file path corresponds to the directory given as the ``graph_dir`` argument
    ///     of ``svs::Vamana::save``.
    /// @param data_loader An acceptable data loader.
    ///     See the documentation below for details.
    /// @param distance The distance functor or ``svs::DistanceType`` enum to use for
    ///     similarity search computations.
    /// @param num_threads The number of threads to use to process queries.
    ///     May be changed at run-time.
    ///
    /// The data loader should be any object loadable via ``svs::detail::dispatch_load``
    /// returning a Vamana compatible dataset. Concrete examples include:
    ///
    /// * An instance of ``VectorDataLoader``.
    /// * An implementation of ``svs::data::ImmutableMemoryDataset`` (passed by value).
    ///
    /// @sa save, build
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename GraphLoaderType,
        typename DataLoader,
        typename Distance>
    static Vamana assemble(
        const std::filesystem::path& config_path,
        const GraphLoaderType& graph_loader,
        DataLoader&& data_loader,
        const Distance& distance,
        size_t num_threads = 1
    ) {
        // If given an `enum` for the distance type, than we need to dispatch over that
        // enum.
        //
        // Otherwise, we can directly forward the provided distance function.
        if constexpr (std::is_same_v<Distance, DistanceType>) {
            auto dispatcher = DistanceDispatcher(distance);
            return dispatcher([&, num_threads](auto distance_function) {
                return make_vamana<manager::as_typelist<QueryTypes>>(
                    AssembleTag(),
                    config_path,
                    graph_loader,
                    std::forward<DataLoader>(data_loader),
                    distance_function,
                    num_threads
                );
            });
        } else {
            return make_vamana<manager::as_typelist<QueryTypes>>(
                AssembleTag(),
                config_path,
                graph_loader,
                std::forward<DataLoader>(data_loader),
                distance,
                num_threads
            );
        }
    }

    ///
    /// @brief Construct the a Vamana Index for the given dataset.
    ///
    /// @tparam QueryType The element type of the queries that will be given to this index.
    ///
    /// @param parameters The build parameters for the search graph constructed over the
    ///     data.
    /// @param data_loader Either a data loader from disk or a dataset by value.
    ///     See detailed notes below.
    /// @param distance The distance functor to use or a ``svs::DistanceType`` enum.
    /// @param num_threads The number of threads to use for query processing (may be
    ///     changed after class construction).
    /// @param graph_allocator The allocator to use for the backing graph.
    ///
    /// The data loader should be any object loadable via ``svs::detail::dispatch_load``
    /// returning a Vamana compatible dataset. Concrete examples include:
    ///
    /// * An instance of ``VectorDataLoader``.
    /// * An implementation of ``svs::data::ImmutableMemoryDataset`` (passed by value).
    ///
    /// @sa assemble, save
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename DataLoader,
        typename Distance,
        typename Allocator = HugepageAllocator<uint32_t>>
    static Vamana build(
        const index::vamana::VamanaBuildParameters& parameters,
        DataLoader&& data_loader,
        Distance distance,
        size_t num_threads = 1,
        const Allocator& graph_allocator = {}
    ) {
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            auto dispatcher = DistanceDispatcher(distance);
            return dispatcher([&](auto distance_function) {
                return make_vamana<manager::as_typelist<QueryTypes>>(
                    BuildTag(),
                    parameters,
                    std::forward<DataLoader>(data_loader),
                    std::move(distance_function),
                    num_threads,
                    graph_allocator
                );
            });
        } else {
            return make_vamana<manager::as_typelist<QueryTypes>>(
                BuildTag(),
                parameters,
                std::forward<DataLoader>(data_loader),
                distance,
                num_threads,
                graph_allocator
            );
        }
    }

    ///// Iterator

    /// @brief Return a new batch iterator for the query using the provided schedule.
    ///
    /// The parameter `QueryType` must be an element of  ``svs::Vamana::query_types()``.
    ///
    /// The returned iterator will maintain an internal copy of the query.
    template <typename QueryType, size_t N, svs::index::vamana::IteratorSchedule Schedule>
    svs::VamanaIterator
    batch_iterator(std::span<const QueryType, N> query, Schedule schedule) const {
        return impl_->batch_iterator(
            svs::AnonymousArray<1>(query.data(), query.size()),
            svs::index::vamana::AbstractIteratorSchedule(std::move(schedule))
        );
    }

    ///// Experimental Calibration
    template <
        data::ImmutableMemoryDataset Queries,
        data::ImmutableMemoryDataset GroundTruth>
    VamanaSearchParameters experimental_calibrate(
        const Queries& queries,
        const GroundTruth& groundtruth,
        size_t num_neighbors,
        double target_recall,
        const index::vamana::CalibrationParameters calibration_parameters = {}
    ) {
        return experimental_calibrate_impl(
            queries.cview(),
            groundtruth.cview(),
            num_neighbors,
            target_recall,
            calibration_parameters
        );
    }

    template <typename QueryType>
    VamanaSearchParameters experimental_calibrate_impl(
        data::ConstSimpleDataView<QueryType> queries,
        data::ConstSimpleDataView<uint32_t> groundtruth,
        size_t num_neighbors,
        double target_recall,
        const index::vamana::CalibrationParameters calibration_parameters
    ) {
        return impl_->experimental_calibrate(
            ConstErasedPointer{queries.data()},
            queries.size(),
            queries.dimensions(),
            ConstErasedPointer{groundtruth.data()},
            groundtruth.size(),
            groundtruth.dimensions(),
            num_neighbors,
            target_recall,
            calibration_parameters
        );
    }
};

///
/// @ingroup orchestrator_vamana_entry
/// @brief Construct an instance of Vamana using the VamanaIndex constructor.
///
/// @tparam QueryType The element type of queries that will be supported by the
///     underlying implementation.
///
/// Uses the VamanaIndex constructor (both those defined explicitly as well as
/// compiler-generated move constructors) to construct an instance of Vamana.
///
/// Due to the limitations of type-erasure, the query type that will be accepted by
/// the resulting index needs to be declared up-front as the ``QueryType`` type parameter.
///
/// @sa svs::index::vamana::VamanaIndex::VamanaIndex
///
template <lib::TypeList QueryTypes, typename... Args> Vamana make_vamana(Args&&... args) {
    using Impl = decltype(index::vamana::VamanaIndex{std::forward<Args>(args)...});
    return Vamana{
        std::make_unique<VamanaImpl<QueryTypes, Impl>>(std::forward<Args>(args)...)};
}

///
/// @ingroup orchestrator_vamana_entry
/// @brief Construct an instance of Vamana using the factory function.
///
/// @tparam QueryType The element type of queries that will be supported by the
///     underlying implementation.
///
/// Uses ``vamana::auto_build`` to create the type-erased Vamana orchestrator.
/// All arguments after the dispatch tag are forwarded to that overload set.
/// Due to the limitations of type-erasure, the query type that will be accepted by
/// the resulting index needs to be declared up-front as the ``QueryType`` type parameter.
///
/// @sa svs::index::vamana::vamana_index_factory
///
template <lib::TypeList QueryTypes, typename... Args>
Vamana make_vamana(Vamana::BuildTag SVS_UNUSED(tag), Args&&... args) {
    return make_vamana<QueryTypes>(index::vamana::auto_build(std::forward<Args>(args)...));
}

template <lib::TypeList QueryTypes, typename... Args>
Vamana make_vamana(Vamana::AssembleTag SVS_UNUSED(tag), Args&&... args) {
    return make_vamana<QueryTypes>(index::vamana::auto_assemble(std::forward<Args>(args)...)
    );
}

} // namespace svs
