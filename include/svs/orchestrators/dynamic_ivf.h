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

#include "svs/index/ivf/dynamic_ivf.h"
#include "svs/index/ivf/index.h"
#include "svs/lib/bfloat16.h"
#include "svs/orchestrators/ivf.h"
#include "svs/orchestrators/manager.h"

namespace svs {

///
/// @brief Type-erased wrapper for DynamicIVF.
///
/// Implementation details: The DynamicIVF implementation implements a superset of the
/// operations supported by the IVFInterface.
///
class DynamicIVFInterface : public IVFInterface {
  public:
    // TODO: For now - only accept floating point entries.
    virtual void add_points(
        const float* data,
        size_t dim0,
        size_t dim1,
        std::span<const size_t> ids,
        bool reuse_empty = false
    ) = 0;

    virtual size_t delete_points(std::span<const size_t> ids) = 0;
    virtual void consolidate() = 0;
    virtual void compact(size_t batchsize = 1'000'000) = 0;

    // ID inspection.
    virtual bool has_id(size_t id) const = 0;
    virtual void all_ids(std::vector<size_t>& ids) const = 0;

    // Distance calculation
    virtual double get_distance(size_t id, const AnonymousArray<1>& query) const = 0;

    // Saving
    virtual void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) = 0;
};

template <lib::TypeList QueryTypes, typename Impl>
class DynamicIVFImpl : public IVFImpl<QueryTypes, Impl, DynamicIVFInterface> {
  public:
    using base_type = IVFImpl<QueryTypes, Impl, DynamicIVFInterface>;
    using base_type::impl;

    explicit DynamicIVFImpl(Impl impl)
        : base_type{std::move(impl)} {}

    template <typename... Args>
    explicit DynamicIVFImpl(Args&&... args)
        : base_type{std::forward<Args>(args)...} {}

    // Implement the interface.
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

    size_t delete_points(std::span<const size_t> ids) override {
        return impl().delete_entries(ids);
    }

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
    void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) override {
        impl().save(config_directory, data_directory);
    }
};

// Forward Declarations.
class DynamicIVF;

template <lib::TypeList QueryTypes, typename... Args>
DynamicIVF make_dynamic_ivf(Args&&... args);

///
/// DynamicIVF
///
class DynamicIVF : public manager::IndexManager<DynamicIVFInterface> {
  public:
    using base_type = manager::IndexManager<DynamicIVFInterface>;
    using IVFSearchParameters = index::ivf::IVFSearchParameters;

    struct AssembleTag {};

    ///
    /// @brief Construct a new DynamicIVF instance.
    ///
    /// @param impl A pointer to a concrete implementation of the full
    ///     DynamicIVFInterface.
    ///
    explicit DynamicIVF(std::unique_ptr<manager::ManagerInterface<DynamicIVFInterface>> impl
    )
        : base_type{std::move(impl)} {}

    template <lib::TypeList QueryTypes, typename Impl>
    explicit DynamicIVF(AssembleTag SVS_UNUSED(tag), QueryTypes SVS_UNUSED(type), Impl impl)
        : base_type{std::make_unique<DynamicIVFImpl<QueryTypes, Impl>>(std::move(impl))} {}

    // Mutable Interface.
    DynamicIVF& add_points(
        data::ConstSimpleDataView<float> points,
        std::span<const size_t> ids,
        bool reuse_empty = false
    ) {
        impl_->add_points(
            points.data(), points.size(), points.dimensions(), ids, reuse_empty
        );
        return *this;
    }

    size_t delete_points(std::span<const size_t> ids) { return impl_->delete_points(ids); }

    DynamicIVF& consolidate() {
        impl_->consolidate();
        return *this;
    }

    DynamicIVF& compact(size_t batchsize = 1'000'000) {
        impl_->compact(batchsize);
        return *this;
    }

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
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) {
        impl_->save(config_directory, data_directory);
    }

    ///// Distance
    template <typename Query> double get_distance(size_t id, const Query& query) const {
        // Create AnonymousArray from the query
        AnonymousArray<1> query_array{query.data(), query.size()};
        return impl_->get_distance(id, query_array);
    }

    // Building
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename DataProto,
        typename Distance,
        typename ThreadPoolProto>
    static DynamicIVF build(
        const index::ivf::IVFBuildParameters& build_parameters,
        const DataProto& data_proto,
        std::span<const size_t> ids,
        Distance distance,
        ThreadPoolProto threadpool_proto,
        size_t intra_query_threads = 1,
        size_t num_clustering_threads = 0
    ) {
        // Handle DistanceType enum by dispatching to concrete distance types
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            auto dispatcher = DistanceDispatcher(distance);
            return dispatcher([&](auto distance_function) {
                return build<QueryTypes>(
                    build_parameters,
                    data_proto,
                    ids,
                    distance_function,
                    std::move(threadpool_proto),
                    intra_query_threads,
                    num_clustering_threads
                );
            });
        } else {
            auto threadpool = threads::as_threadpool(std::move(threadpool_proto));

            // Load the data (handles both loaders and views)
            auto data = svs::detail::dispatch_load(data_proto, threadpool);

            // Build clustering first
            // Choose build type for clustering to leverage AMX instructions:
            // - Float32 data -> BFloat16 (AMX supports BFloat16)
            // - Float16 data -> Float16 (AMX supports Float16)
            // - BFloat16 data -> BFloat16 (already optimal)
            using DataElementType = typename decltype(data)::element_type;
            using BuildType = std::conditional_t<
                std::is_same_v<DataElementType, float>,
                BFloat16,
                DataElementType>;
            // Note: build_clustering takes threadpool by value, so we need a copy
            auto clustering = [&]() {
                size_t clustering_threads = (num_clustering_threads == 0)
                                                ? threadpool.size()
                                                : num_clustering_threads;
                auto threadpool_copy = threads::NativeThreadPool(clustering_threads);
                return index::ivf::build_clustering<BuildType>(
                    build_parameters, data, distance, std::move(threadpool_copy), false
                );
            }();

            // Now build the dynamic IVF index from the clustering
            auto impl = index::ivf::build_dynamic_ivf(
                std::move(clustering.centroids_),
                clustering,
                data, // Pass by const reference - build_dynamic_ivf will create BlockedData
                ids,
                distance,
                std::move(threadpool),
                intra_query_threads
            );

            return DynamicIVF(
                AssembleTag(), manager::as_typelist<QueryTypes>{}, std::move(impl)
            );
        }
    }

    // Assembly
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename Centroids,
        typename Data,
        typename Distance,
        typename ThreadPoolProto>
    static DynamicIVF assemble(
        Centroids centroids,
        Data data,
        std::span<const size_t> ids,
        Distance distance,
        ThreadPoolProto threadpool_proto
    ) {
        using I = uint32_t;
        using Cluster = index::ivf::DynamicDenseCluster<Data, I>;

        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));

        // For assembly, create empty clusters - user will add points later
        std::vector<Cluster> clusters;
        clusters.reserve(centroids.size());

        for (size_t i = 0; i < centroids.size(); ++i) {
            auto cluster_data = Data(0, data.dimensions());
            std::vector<I> cluster_ids;
            clusters.emplace_back(std::move(cluster_data), std::move(cluster_ids));
        }

        // Create the index with empty clusters
        auto impl =
            index::ivf::DynamicIVFIndex<Centroids, Cluster, Distance, decltype(threadpool)>(
                std::move(centroids),
                std::move(clusters),
                ids,
                std::move(distance),
                std::move(threadpool)
            );

        return DynamicIVF(
            AssembleTag(), manager::as_typelist<QueryTypes>{}, std::move(impl)
        );
    }
};

} // namespace svs
