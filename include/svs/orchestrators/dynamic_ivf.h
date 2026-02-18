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

    virtual void save(std::ostream& stream) = 0;
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

    void save(std::ostream& stream) override {
        lib::UniqueTempDirectory tempdir{"svs_dynamic_ivf_save"};
        const auto config_dir = tempdir.get() / "config";
        const auto data_dir = tempdir.get() / "data";
        std::filesystem::create_directories(config_dir);
        std::filesystem::create_directories(data_dir);
        save(config_dir, data_dir);
        lib::DirectoryArchiver::pack(tempdir, stream);
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

    ///
    /// @brief Save the DynamicIVF index to a stream.
    ///
    /// @param stream Output stream to save the index to.
    ///
    /// The index is saved in a binary format that can be loaded using the
    /// stream-based ``assemble`` method.
    ///
    /// @sa assemble
    ///
    void save(std::ostream& stream) const { impl_->save(stream); }

    ///// Distance
    template <typename Query> double get_distance(size_t id, const Query& query) const {
        // Create AnonymousArray from the query
        AnonymousArray<1> query_array{query.data(), query.size()};
        return impl_->get_distance(id, query_array);
    }

    ///
    /// @brief Return a new iterator (an instance of `svs::IVFIterator`) for the query.
    ///
    /// @tparam QueryType The element type of the query that will be given to the iterator.
    /// @tparam N The dimension of the query.
    ///
    /// @param query The query to use for the iterator.
    /// @param extra_search_buffer_capacity An optional extra search buffer capacity.
    ///     For IVF, the default of 0 means the buffer will be sized based on the first
    ///     batch_size passed to next().
    ///
    /// The returned iterator will maintain an internal copy of the query.
    ///
    template <typename QueryType, size_t N>
    svs::IVFIterator batch_iterator(
        std::span<const QueryType, N> query, size_t extra_search_buffer_capacity = 0
    ) {
        return impl_->batch_iterator(
            svs::AnonymousArray<1>(query.data(), query.size()), extra_search_buffer_capacity
        );
    }

    ///// Assembly - Assemble from clustering and data
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename Clustering,
        typename Data,
        typename Distance,
        typename ThreadPoolProto>
    static DynamicIVF assemble_from_clustering(
        Clustering clustering,
        Data data,
        std::span<const size_t> ids,
        Distance distance,
        ThreadPoolProto threadpool_proto,
        size_t intra_query_threads = 1
    ) {
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));

        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            auto dispatcher = DistanceDispatcher(distance);
            return dispatcher([&](auto distance_function) {
                auto impl = index::ivf::assemble_dynamic_from_clustering(
                    std::move(clustering),
                    data,
                    ids,
                    std::move(distance_function),
                    std::move(threadpool),
                    intra_query_threads
                );
                return DynamicIVF(
                    AssembleTag(), manager::as_typelist<QueryTypes>{}, std::move(impl)
                );
            });
        } else {
            auto impl = index::ivf::assemble_dynamic_from_clustering(
                std::move(clustering),
                data,
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

    ///// Assembly - Assemble from file (load clustering from disk)
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename BuildType,
        typename Data,
        typename Distance,
        typename ThreadPoolProto>
    static DynamicIVF assemble_from_file(
        const std::filesystem::path& cluster_path,
        Data data,
        std::span<const size_t> ids,
        Distance distance,
        ThreadPoolProto threadpool_proto,
        size_t intra_query_threads = 1
    ) {
        using centroids_type = data::SimpleData<BuildType>;
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
        auto clustering =
            lib::load_from_disk<index::ivf::Clustering<centroids_type, uint32_t>>(
                cluster_path, threadpool
            );
        return assemble_from_clustering<QueryTypes>(
            std::move(clustering),
            data,
            ids,
            distance,
            std::move(threadpool),
            intra_query_threads
        );
    }

    ///
    /// @brief Load a saved DynamicIVF index from disk.
    ///
    /// This method restores a DynamicIVF index that was previously saved using `save()`.
    ///
    /// @tparam QueryTypes The query types supported by the returned index.
    /// @tparam CentroidType Element type of centroids (e.g., float, BFloat16).
    /// @tparam DataType Full cluster data type (e.g., BlockedData<float>).
    ///
    /// @param config_path Path to the saved configuration directory.
    /// @param data_path Path to the saved data directory (centroids and clusters).
    /// @param distance Distance metric for searching.
    /// @param threadpool_proto Thread pool prototype for parallel processing.
    /// @param intra_query_threads Number of threads for intra-query parallelism.
    ///
    /// @return A fully constructed DynamicIVF ready for searching and modifications.
    ///
    /// @sa save, assemble_from_file
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename CentroidType,
        typename DataType,
        typename Distance,
        typename ThreadPoolProto>
    static DynamicIVF assemble(
        const std::filesystem::path& config_path,
        const std::filesystem::path& data_path,
        Distance distance,
        ThreadPoolProto threadpool_proto,
        size_t intra_query_threads = 1
    ) {
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            auto dispatcher = DistanceDispatcher(distance);
            return dispatcher([&](auto distance_function) {
                return DynamicIVF(
                    AssembleTag(),
                    manager::as_typelist<QueryTypes>{},
                    index::ivf::load_dynamic_ivf_index<CentroidType, DataType>(
                        config_path,
                        data_path,
                        std::move(distance_function),
                        std::move(threadpool),
                        intra_query_threads
                    )
                );
            });
        } else {
            return DynamicIVF(
                AssembleTag(),
                manager::as_typelist<QueryTypes>{},
                index::ivf::load_dynamic_ivf_index<CentroidType, DataType>(
                    config_path,
                    data_path,
                    distance,
                    std::move(threadpool),
                    intra_query_threads
                )
            );
        }
    }

    ///
    /// @brief Load a DynamicIVF index from a stream.
    ///
    /// @tparam QueryTypes The query types supported by the returned index.
    /// @tparam CentroidType Element type of centroids (e.g., float, BFloat16).
    /// @tparam DataType Full cluster data type (e.g., SimpleData<float>).
    ///
    /// @param stream Input stream to load the index from.
    /// @param distance Distance metric for searching.
    /// @param threadpool_proto Thread pool prototype for parallel processing.
    /// @param intra_query_threads Number of threads for intra-query parallelism.
    ///
    /// @return A fully constructed DynamicIVF ready for searching and modifications.
    ///
    /// @sa save
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename CentroidType,
        typename DataType,
        typename Distance,
        typename ThreadPoolProto>
    static DynamicIVF assemble(
        std::istream& stream,
        Distance distance,
        ThreadPoolProto threadpool_proto,
        size_t intra_query_threads = 1
    ) {
        namespace fs = std::filesystem;
        lib::UniqueTempDirectory tempdir{"svs_dynamic_ivf_load"};
        lib::DirectoryArchiver::unpack(stream, tempdir);

        const auto config_path = tempdir.get() / "config";
        if (!fs::is_directory(config_path)) {
            throw ANNEXCEPTION("Invalid DynamicIVF index archive: missing config directory!"
            );
        }

        const auto data_path = tempdir.get() / "data";
        if (!fs::is_directory(data_path)) {
            throw ANNEXCEPTION("Invalid DynamicIVF index archive: missing data directory!");
        }

        return assemble<QueryTypes, CentroidType, DataType>(
            config_path,
            data_path,
            distance,
            std::move(threadpool_proto),
            intra_query_threads
        );
    }
};

} // namespace svs
