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

template <typename QueryType, typename Impl>
class DynamicVamanaImpl : public VamanaImpl<QueryType, Impl, DynamicVamanaInterface> {
  public:
    using base_type = VamanaImpl<QueryType, Impl, DynamicVamanaInterface>;
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

template <typename QueryType, typename... Args>
DynamicVamana make_dynamic_vamana(Args&&... args);

///
/// DynamicVamana
///
class DynamicVamana
    : public manager::IndexManager<DynamicVamanaInterface, DynamicVamanaImpl> {
  public:
    using base_type = manager::IndexManager<DynamicVamanaInterface, DynamicVamanaImpl>;

    struct AssembleTag {};

    ///
    /// @brief Construct a new DynamicVamana instance.
    ///
    /// @param impl A pointer to a concrete implementation of the full
    ///     DynamicVamanaInteface.
    ///
    template <typename Impl>
    explicit DynamicVamana(std::unique_ptr<Impl> impl)
        : base_type{std::move(impl)} {}

    template <typename QueryType, typename Impl>
    explicit DynamicVamana(
        AssembleTag SVS_UNUSED(tag), lib::meta::Type<QueryType> SVS_UNUSED(type), Impl impl
    )
        : base_type{std::make_unique<DynamicVamanaImpl<QueryType, Impl>>(std::move(impl))} {
    }

    ///// Vamana Interface

    ///
    /// @brief Set the search window size used to process queries.
    ///
    /// @param search_window_size The new search window size.
    ///
    DynamicVamana& set_search_window_size(size_t search_window_size) {
        impl_->set_search_window_size(search_window_size);
        return *this;
    }

    ///
    /// @brief The current search window size used to process queries.
    ///
    size_t get_search_window_size() const { return impl_->get_search_window_size(); }

    bool visited_set_enabled() const { return impl_->visited_set_enabled(); }
    void enable_visited_set() { impl_->enable_visited_set(); }
    void disable_visited_set() { impl_->disable_visited_set(); }

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
    float get_alpha() const { return impl_->get_alpha(); }
    void set_alpha(size_t alpha) { impl_->set_alpha(alpha); }
    size_t get_construction_window_size() const {
        return impl_->get_construction_window_size();
    }
    void set_construction_window_size(size_t window_size) {
        impl_->set_construction_window_size(window_size);
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

    // Building
    template <typename QueryType, data::ImmutableMemoryDataset Data, typename Distance>
    static DynamicVamana build(
        const index::vamana::VamanaBuildParameters& parameters,
        Data data,
        std::span<const size_t> ids,
        Distance distance,
        size_t num_threads
    ) {
        fmt::print("Entering build!\n");
        return make_dynamic_vamana<QueryType>(
            parameters, std::move(data), ids, std::move(distance), num_threads
        );
    }

    // Assembly
    template <
        typename QueryType,
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
            lib::meta::Type<QueryType>(),
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
};

///
/// @brief Construct a ``DynamicVamana`` by calling the default implementations'
/// constructor.
///
template <typename QueryType, typename... Args>
DynamicVamana make_dynamic_vamana(Args&&... args) {
    using Impl = decltype(index::vamana::MutableVamanaIndex{std::forward<Args>(args)...});
    return DynamicVamana{
        std::make_unique<DynamicVamanaImpl<QueryType, Impl>>(std::forward<Args>(args)...)};
}

} // namespace svs
