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

#include "svs/index/vamana/dynamic_index.h"

#include "svs/orchestrators/manager.h"
#include "svs/orchestrators/vamana.h"

namespace svs {

///
/// @brief Type-erased wrapper for DynamicVamana.
///
/// Implementation details: The DynamicVamana implementation implements a superset of the
/// operations supported by the VamanaInterface,
///
class DynamicVamanaInterface : public VamanaInterface {
  public:
    // TODO: For now - only accept floating point entries.
    virtual void add_points(
        const float* data, size_t dim0, size_t dim1, std::span<const size_t> ids
    ) = 0;

    virtual void delete_points(std::span<const size_t> ids) = 0;
    virtual void consolidate() = 0;
    virtual void compact(size_t batchsize = 1'000'000) = 0;

    // ID inspection.
    virtual bool has_id(size_t id) const = 0;
    virtual void all_ids(std::vector<size_t>& ids) const = 0;
};

template <lib::TypeList QueryTypes, typename Impl>
class DynamicVamanaImpl : public VamanaImpl<QueryTypes, Impl, DynamicVamanaInterface> {
  public:
    using base_type = VamanaImpl<QueryTypes, Impl, DynamicVamanaInterface>;
    using base_type::impl;

    explicit DynamicVamanaImpl(Impl impl)
        : base_type{std::move(impl)} {}

    template <typename... Args>
    explicit DynamicVamanaImpl(Args&&... args)
        : base_type{std::forward<Args>(args)...} {}

    // Implement the interface.
    void add_points(
        const float* data, size_t dim0, size_t dim1, std::span<const size_t> ids
    ) override {
        auto points = data::ConstSimpleDataView<float>(data, dim0, dim1);
        impl().add_points(points, ids);
    }

    void delete_points(std::span<const size_t> ids) override { impl().delete_entries(ids); }
    void consolidate() override { impl().consolidate(); }
    void compact(size_t batchsize) override { impl().compact(batchsize); }

    // ID inspection.
    bool has_id(size_t id) const override { return impl().has_id(id); }

    void all_ids(std::vector<size_t>& ids) const override {
        ids.clear();
        impl().on_ids([&ids](size_t id) { ids.push_back(id); });
    }
};

// Forward Declaractions.
class DynamicVamana;

template <lib::TypeList QueryTypes, typename... Args>
DynamicVamana make_dynamic_vamana(Args&&... args);

///
/// DynamicVamana
///
class DynamicVamana : public manager::IndexManager<DynamicVamanaInterface> {
  public:
    using base_type = manager::IndexManager<DynamicVamanaInterface>;
    using VamanaSearchParameters = index::vamana::VamanaSearchParameters;

    struct AssembleTag {};

    ///
    /// @brief Construct a new DynamicVamana instance.
    ///
    /// @param impl A pointer to a concrete implementation of the full
    ///     DynamicVamanaInteface.
    ///
    explicit DynamicVamana(
        std::unique_ptr<manager::ManagerInterface<DynamicVamanaInterface>> impl
    )
        : base_type{std::move(impl)} {}

    template <lib::TypeList QueryTypes, typename Impl>
    explicit DynamicVamana(
        AssembleTag SVS_UNUSED(tag), QueryTypes SVS_UNUSED(type), Impl impl
    )
        : base_type{
              std::make_unique<DynamicVamanaImpl<QueryTypes, Impl>>(std::move(impl))} {}

    ///// Vamana Interface
    void experimental_reset_performance_parameters() {
        impl_->reset_performance_parameters();
    }

    ///
    /// @brief Set the search window size used to process queries.
    ///
    /// @param search_window_size The new search window size.
    ///
    DynamicVamana& set_search_window_size(size_t search_window_size) {
        auto parameters = get_search_parameters();
        parameters.buffer_config_ = {search_window_size};
        set_search_parameters(parameters);
        return *this;
    }

    ///
    /// @brief The current search window size used to process queries.
    ///
    size_t get_search_window_size() const {
        return get_search_parameters().buffer_config_.get_search_window_size();
    }

    // Mutable Interface.
    DynamicVamana& consolidate() {
        impl_->consolidate();
        return *this;
    }

    DynamicVamana& compact(size_t batchsize = 1'000'000) {
        impl_->compact(batchsize);
        return *this;
    }

    DynamicVamana&
    add_points(data::ConstSimpleDataView<float> points, std::span<const size_t> ids) {
        impl_->add_points(points.data(), points.size(), points.dimensions(), ids);
        return *this;
    }

    DynamicVamana& delete_points(std::span<const size_t> ids) {
        impl_->delete_points(ids);
        return *this;
    }

    // Accessors
    /// @copydoc svs::index::vamana::MutableVamanaIndex::get_alpha
    float get_alpha() const { return impl_->get_alpha(); }
    void set_alpha(size_t alpha) { impl_->set_alpha(alpha); }

    /// @copydoc svs::index::vamana::MutableVamanaIndex::get_graph_max_degree
    size_t get_graph_max_degree() const { return impl_->get_graph_max_degree(); }

    /// @copydoc svs::index::vamana::MutableVamanaIndex::set_construction_window_size
    size_t get_construction_window_size() const {
        return impl_->get_construction_window_size();
    }
    void set_construction_window_size(size_t window_size) {
        impl_->set_construction_window_size(window_size);
    }

    /// @copydoc svs::index::vamana::MutableVamanaIndex::get_max_candidates
    size_t get_max_candidates() const { return impl_->get_max_candidates(); }
    void set_max_candidates(size_t max_candidates) {
        impl_->set_max_candidates(max_candidates);
    }

    /// @copydoc svs::index::vamana::MutableVamanaIndex::get_prune_to
    size_t get_prune_to() const { return impl_->get_prune_to(); }
    void set_prune_to(size_t prune_to) { impl_->set_prune_to(prune_to); }

    /// @copydoc svs::index::vamana::MutableVamanaIndex::get_full_search_history
    bool get_full_search_history() const { return impl_->get_full_search_history(); }
    void set_full_search_history(bool enable) { impl_->set_full_search_history(enable); }

    // Backend String
    std::string experimental_backend_string() const {
        return impl_->experimental_backend_string();
    }

    // ID Inspection

    ///
    /// @brief Return whether ``id`` is in the index.
    ///
    bool has_id(size_t id) const { return impl_->has_id(id); }

    ///
    /// @brief Return all ``ids`` currently in the index.
    ///
    /// Note: If the stored index is large, the returned container may result in a
    /// significant memory allocation.
    ///
    /// If more precise handling is required, please work with the lower level C++ class
    /// directly.
    ///
    std::vector<size_t> all_ids() const {
        auto v = std::vector<size_t>();
        impl_->all_ids(v);
        return v;
    }

    void save(
        const std::filesystem::path& config_dir,
        const std::filesystem::path& graph_dir,
        const std::filesystem::path& data_dir
    ) {
        impl_->save(config_dir, graph_dir, data_dir);
    }

    /// Reconstruction
    void reconstruct_at(data::SimpleDataView<float> data, std::span<const uint64_t> ids) {
        impl_->reconstruct_at(data, ids);
    }

    // Building
    template <
        manager::QueryTypeDefinition QueryTypes,
        data::ImmutableMemoryDataset Data,
        typename Distance>
    static DynamicVamana build(
        const index::vamana::VamanaBuildParameters& parameters,
        Data data,
        std::span<const size_t> ids,
        Distance distance,
        size_t num_threads
    ) {
        return make_dynamic_vamana<manager::as_typelist<QueryTypes>>(
            parameters, std::move(data), ids, std::move(distance), num_threads
        );
    }

    // Assembly
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename GraphLoader,
        typename DataLoader,
        typename Distance>
    static DynamicVamana assemble(
        const std::filesystem::path& config_path,
        const GraphLoader& graph_loader,
        const DataLoader& data_loader,
        const Distance& distance,
        size_t num_threads,
        bool debug_load_from_static = false
    ) {
        return DynamicVamana(
            AssembleTag(),
            manager::as_typelist<QueryTypes>(),
            index::vamana::auto_dynamic_assemble(
                config_path,
                graph_loader,
                data_loader,
                distance,
                num_threads,
                debug_load_from_static
            )
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
/// @brief Construct a ``DynamicVamana`` by calling the default implementations'
/// constructor.
///
template <lib::TypeList QueryTypes, typename... Args>
DynamicVamana make_dynamic_vamana(Args&&... args) {
    using Impl = decltype(index::vamana::MutableVamanaIndex{std::forward<Args>(args)...});
    return DynamicVamana{
        std::make_unique<DynamicVamanaImpl<QueryTypes, Impl>>(std::forward<Args>(args)...)};
}

} // namespace svs
