/*
 * Copyright 2025 Intel Corporation
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

#include "svs/index/flat/dynamic_flat.h"

#include "svs/orchestrators/exhaustive.h"
#include "svs/orchestrators/manager.h"

namespace svs {

///
/// @brief Type-erased wrapper for DynamicFlat.
///
/// Implementation details: The DynamicFlat implementation implements a superset of the
/// operations supported by the FlatInterface.
///
class DynamicFlatInterface {
  public:
    using search_parameters_type = svs::index::flat::FlatParameters;

    // Non-templated virtual method for distance calculation
    virtual double get_distance(size_t id, const AnonymousArray<1>& query) const = 0;

    // TODO: For now - only accept floating point entries.
    virtual void add_points(
        const float* data,
        size_t dim0,
        size_t dim1,
        std::span<const size_t> ids,
        bool reuse_empty = false
    ) = 0;

    virtual void delete_points(std::span<const size_t> ids) = 0;
    virtual void consolidate() = 0;
    virtual void compact(size_t batchsize = 1'000) = 0;

    // ID inspection.
    virtual bool has_id(size_t id) const = 0;
    virtual void all_ids(std::vector<size_t>& ids) const = 0;

    // Saving - 2 parameter version
    virtual void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) = 0;
    virtual void save(std::ostream& stream) = 0;
};

template <lib::TypeList QueryTypes, typename Impl>
class DynamicFlatImpl
    : public manager::ManagerImpl<QueryTypes, Impl, DynamicFlatInterface> {
  public:
    using base_type = manager::ManagerImpl<QueryTypes, Impl, DynamicFlatInterface>;
    using base_type::impl;

    explicit DynamicFlatImpl(Impl impl)
        : base_type{std::move(impl)} {}

    template <typename... Args>
    explicit DynamicFlatImpl(Args&&... args)
        : base_type{std::forward<Args>(args)...} {}

    // Implement the dynamic interface.
    void add_points(
        const float* data,
        size_t dim0,
        size_t dim1,
        std::span<const size_t> ids,
        bool reuse_empty = false
    ) override {
        auto points = data::ConstSimpleDataView<float>(data, dim0, dim1);
        impl().add_points(points, ids, reuse_empty);
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

    ///// Distance
    double get_distance(size_t id, const AnonymousArray<1>& query) const override {
        return svs::lib::match(
            QueryTypes{},
            query.type(),
            [&]<typename T>(svs::lib::Type<T>) {
                auto query_span = std::span<const T>(get<T>(query), query.size(0));
                return impl().get_distance(id, query_span);
            }
        );
    }

    ///// Saving
    // 2-parameter save implementation
    void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) override {
        impl().save(config_directory, data_directory);
    }

    // Stream-based save implementation
    void save(std::ostream& stream) override {
        if constexpr (Impl::supports_saving) {
            lib::UniqueTempDirectory tempdir{"svs_dynflat_save"};
            const auto config_dir = tempdir.get() / "config";
            const auto data_dir = tempdir.get() / "data";
            std::filesystem::create_directories(config_dir);
            std::filesystem::create_directories(data_dir);
            save(config_dir, data_dir);
            lib::DirectoryArchiver::pack(tempdir, stream);
        } else {
            throw ANNEXCEPTION("The current DynamicFlat backend doesn't support saving!");
        }
    }
};

// Forward Declarations.
class DynamicFlat;

template <lib::TypeList QueryTypes, typename... Args>
DynamicFlat make_dynamic_flat(Args&&... args);

///
/// DynamicFlat
///
class DynamicFlat : public manager::IndexManager<DynamicFlatInterface> {
  public:
    using base_type = manager::IndexManager<DynamicFlatInterface>;
    using FlatSearchParameters = index::flat::FlatParameters;

    struct AssembleTag {};

    ///
    /// @brief Construct a new DynamicFlat instance.
    ///
    /// @param impl A pointer to a concrete implementation of the full
    ///     DynamicFlatInterface.
    ///
    explicit DynamicFlat(
        std::unique_ptr<manager::ManagerInterface<DynamicFlatInterface>> impl
    )
        : base_type{std::move(impl)} {}

    template <lib::TypeList QueryTypes, typename Impl>
    explicit DynamicFlat(
        AssembleTag SVS_UNUSED(tag), QueryTypes SVS_UNUSED(type), Impl impl
    )
        : base_type{std::make_unique<DynamicFlatImpl<QueryTypes, Impl>>(std::move(impl))} {}

    // Mutable Interface.
    DynamicFlat& add_points(
        data::ConstSimpleDataView<float> points,
        std::span<const size_t> ids,
        bool reuse_empty = false
    ) {
        impl_->add_points(
            points.data(), points.size(), points.dimensions(), ids, reuse_empty
        );
        return *this;
    }

    DynamicFlat& delete_points(std::span<const size_t> ids) {
        impl_->delete_points(ids);
        return *this;
    }

    DynamicFlat& consolidate() {
        impl_->consolidate();
        return *this;
    }

    DynamicFlat& compact(size_t batchsize = 1'000) {
        impl_->compact(batchsize);
        return *this;
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
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) const {
        impl_->save(config_directory, data_directory);
    }

    // Building
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename Data,
        typename Distance,
        typename ThreadPoolProto>
    static DynamicFlat build(
        Data data,
        std::span<const size_t> ids,
        Distance distance,
        ThreadPoolProto threadpool_proto
    ) {
        return make_dynamic_flat<manager::as_typelist<QueryTypes>>(
            std::move(data),
            ids,
            std::move(distance),
            threads::as_threadpool(std::move(threadpool_proto))
        );
    }

    // Assembly
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename DataLoader,
        typename Distance,
        typename ThreadPoolProto>
    static DynamicFlat assemble(
        const std::filesystem::path& config_directory,
        DataLoader&& data_loader,
        const Distance& distance,
        ThreadPoolProto threadpool_proto
    ) {
        return DynamicFlat(
            AssembleTag(),
            manager::as_typelist<QueryTypes>(),
            index::flat::auto_dynamic_assemble(
                config_directory,
                std::forward<DataLoader>(data_loader),
                distance,
                threads::as_threadpool(std::move(threadpool_proto))
            )
        );
    }

    // Assembly from stream
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename Data,
        typename Distance,
        typename ThreadPoolProto,
        typename... DataLoaderArgs>
    static DynamicFlat assemble(
        std::istream& stream,
        const Distance& distance,
        ThreadPoolProto threadpool_proto,
        DataLoaderArgs&&... data_args
    ) {
        namespace fs = std::filesystem;
        lib::UniqueTempDirectory tempdir{"svs_dynflat_load"};
        lib::DirectoryArchiver::unpack(stream, tempdir);

        const auto config_path = tempdir.get() / "config";
        if (!fs::is_directory(config_path)) {
            throw ANNEXCEPTION(
                "Invalid Dynamic Flat index archive: missing config directory!"
            );
        }

        const auto data_path = tempdir.get() / "data";
        if (!fs::is_directory(data_path)) {
            throw ANNEXCEPTION("Invalid Dynamic Flat index archive: missing data directory!"
            );
        }

        return assemble<QueryTypes>(
            config_path,
            lib::load_from_disk<Data>(data_path, SVS_FWD(data_args)...),
            distance,
            threads::as_threadpool(std::move(threadpool_proto))
        );
    }

    ///// Distance
    /// @brief Get the distance between a vector in the index and a query vector
    /// @tparam Query The query vector type
    /// @param id The ID of the vector in the index
    /// @param query The query vector
    template <typename Query> double get_distance(size_t id, const Query& query) const {
        // Create AnonymousArray from the query
        AnonymousArray<1> query_array{query.data(), query.size()};
        return impl_->get_distance(id, query_array);
    }
};

///
/// @brief Construct a ``DynamicFlat`` by calling the default implementations'
/// constructor.
///
template <lib::TypeList QueryTypes, typename... Args>
DynamicFlat make_dynamic_flat(Args&&... args) {
    using Impl = decltype(index::flat::DynamicFlatIndex{std::forward<Args>(args)...});
    return DynamicFlat{
        std::make_unique<DynamicFlatImpl<QueryTypes, Impl>>(std::forward<Args>(args)...)
    };
}

} // namespace svs
