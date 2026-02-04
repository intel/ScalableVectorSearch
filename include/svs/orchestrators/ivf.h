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

#include "svs/index/ivf/index.h"
#include "svs/orchestrators/ivf_iterator.h"
#include "svs/orchestrators/manager.h"

namespace svs {

class IVFInterface {
  public:
    using search_parameters_type = svs::index::ivf::IVFSearchParameters;

    ///// Backend information interface
    virtual std::string experimental_backend_string() const = 0;

    ///// Distance calculation
    virtual double get_distance(size_t id, const AnonymousArray<1>& query) const = 0;

    ///// Iterator
    virtual IVFIterator batch_iterator(
        svs::AnonymousArray<1> query, size_t extra_search_buffer_capacity = 0
    ) = 0;

    ///// Saving
    virtual void save(
        const std::filesystem::path& config_dir, const std::filesystem::path& data_dir
    ) = 0;

    virtual void save(std::ostream& stream) = 0;
};

template <lib::TypeList QueryTypes, typename Impl, typename IFace = IVFInterface>
class IVFImpl : public manager::ManagerImpl<QueryTypes, Impl, IFace> {
  private:
    // Null-terminated array of characters.
    static constexpr auto typename_impl = lib::generate_typename<Impl>();

  public:
    using base_type = manager::ManagerImpl<QueryTypes, Impl, IFace>;
    using base_type::impl;
    using search_parameters_type = typename IFace::search_parameters_type;

    explicit IVFImpl(Impl impl)
        : base_type{std::move(impl)} {}

    ///// Parameter Interface
    [[nodiscard]] search_parameters_type get_search_parameters() const override {
        return impl().get_search_parameters();
    }

    void set_search_parameters(const search_parameters_type& search_parameters) override {
        impl().set_search_parameters(search_parameters);
    }

    ///// Backend Information Interface
    [[nodiscard]] std::string experimental_backend_string() const override {
        return std::string{typename_impl.begin(), typename_impl.end() - 1};
    }

    ///// Distance Calculation
    [[nodiscard]] double
    get_distance(size_t id, const AnonymousArray<1>& query) const override {
        return svs::lib::match(
            QueryTypes{},
            query.type(),
            [&]<typename T>(svs::lib::Type<T>) {
                auto query_span = std::span<const T>(get<T>(query), query.size(0));
                return impl().get_distance(id, query_span);
            }
        );
    }

    ///// Iterator
    IVFIterator batch_iterator(
        svs::AnonymousArray<1> query, size_t extra_search_buffer_capacity = 0
    ) override {
        // Match the query type.
        return svs::lib::match(
            QueryTypes{},
            query.type(),
            [&]<typename T>(svs::lib::Type<T> SVS_UNUSED(type)) {
                return IVFIterator{
                    impl(),
                    std::span<const T>(svs::get<T>(query), query.size(0)),
                    extra_search_buffer_capacity};
            }
        );
    }

    ///// Saving
    void save(
        const std::filesystem::path& config_dir, const std::filesystem::path& data_dir
    ) override {
        if constexpr (Impl::supports_saving) {
            impl().save(config_dir, data_dir);
        } else {
            throw ANNEXCEPTION("The current IVF backend doesn't support saving!");
        }
    }

    void save(std::ostream& stream) override {
        if constexpr (Impl::supports_saving) {
            lib::UniqueTempDirectory tempdir{"svs_ivf_save"};
            const auto config_dir = tempdir.get() / "config";
            const auto data_dir = tempdir.get() / "data";
            std::filesystem::create_directories(config_dir);
            std::filesystem::create_directories(data_dir);
            save(config_dir, data_dir);
            lib::DirectoryArchiver::pack(tempdir, stream);
        } else {
            throw ANNEXCEPTION("The current IVF backend doesn't support saving!");
        }
    }
};

/////
///// IVFManager
/////

class IVF : public manager::IndexManager<IVFInterface> {
    // Type Alises
  public:
    using base_type = manager::IndexManager<IVFInterface>;
    using search_parameters_type = typename IVFInterface::search_parameters_type;

    // Constructors
    IVF(std::unique_ptr<manager::ManagerInterface<IVFInterface>> impl)
        : base_type{std::move(impl)} {}

    template <lib::TypeList QueryTypes, typename Impl>
    IVF(std::in_place_t, QueryTypes SVS_UNUSED(type), Impl&& impl)
        : base_type{std::make_unique<IVFImpl<QueryTypes, Impl>>(SVS_FWD(impl))} {}

    ///// Backend String
    std::string experimental_backend_string() const {
        return impl_->experimental_backend_string();
    }

    ///// Distance Calculation
    template <typename QueryType>
    double get_distance(size_t id, const QueryType& query) const {
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

    ///// Saving
    ///
    /// @brief Save the IVF index to disk.
    ///
    /// @param config_directory Directory where the index configuration will be saved.
    /// @param data_directory Directory where the centroids and cluster data will be saved.
    ///
    /// Each directory may be created as a side-effect of this method call provided that
    /// the parent directory exists.
    ///
    /// @sa assemble
    ///
    void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) {
        impl_->save(config_directory, data_directory);
    }

    ///
    /// @brief Save the IVF index to a stream.
    ///
    /// @param stream Output stream to save the index to.
    ///
    /// The index is saved in a binary format that can be loaded using the
    /// stream-based ``assemble`` method.
    ///
    /// @sa assemble
    ///
    void save(std::ostream& stream) const { impl_->save(stream); }

    ///// Assembling
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename Clustering,
        typename DataProto,
        typename Distance,
        typename ThreadpoolProto>
    static IVF assemble_from_clustering(
        Clustering clustering,
        const DataProto& data_proto,
        const Distance& distance,
        ThreadpoolProto threadpool_proto,
        size_t intra_query_threads = 1
    ) {
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            auto dispatcher = DistanceDispatcher(distance);
            return dispatcher([&](auto distance_function) {
                return IVF(
                    std::in_place,
                    manager::as_typelist<QueryTypes>{},
                    index::ivf::assemble_from_clustering(
                        std::move(clustering),
                        data_proto,
                        std::move(distance_function),
                        std::move(threadpool),
                        intra_query_threads
                    )
                );
            });
        } else {
            return IVF(
                std::in_place,
                manager::as_typelist<QueryTypes>{},
                index::ivf::assemble_from_clustering(
                    std::move(clustering),
                    data_proto,
                    distance,
                    std::move(threadpool),
                    intra_query_threads
                )
            );
        }
    }

    template <
        manager::QueryTypeDefinition QueryTypes,
        typename Centroids,
        typename DataProto,
        typename Distance,
        typename ThreadpoolProto>
    static IVF assemble_from_file(
        const std::filesystem::path& clustering_path,
        const DataProto& data_proto,
        const Distance& distance,
        ThreadpoolProto threadpool_proto,
        size_t intra_query_threads = 1
    ) {
        using centroids_type = data::SimpleData<Centroids>;
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
        auto clustering =
            svs::lib::load_from_disk<index::ivf::Clustering<centroids_type, uint32_t>>(
                clustering_path, threadpool
            );
        return assemble_from_clustering<QueryTypes>(
            std::move(clustering),
            data_proto,
            distance,
            std::move(threadpool),
            intra_query_threads
        );
    }

    ///
    /// @brief Load an IVF Index from a previously saved index.
    ///
    /// @tparam QueryTypes The element types of queries that will be used when requesting
    ///     searches over the index. Can be a single type or a ``svs::lib::Types``.
    /// @tparam CentroidType The element type of the centroids.
    /// @tparam DataType The element type of the cluster data.
    ///
    /// @param config_path Path to the directory where the index configuration was saved.
    ///     This corresponds to the ``config_directory`` argument of ``svs::IVF::save``.
    /// @param data_path Path to the directory where the centroids and cluster data were
    ///     saved. This corresponds to the ``data_directory`` argument of
    ///     ``svs::IVF::save``.
    /// @param distance The distance functor or ``svs::DistanceType`` enum to use for
    ///     similarity search computations.
    /// @param threadpool_proto Precursor for the thread pool to use. Can either be an
    ///     acceptable thread pool instance or an integer specifying the number of threads
    ///     to use.
    /// @param intra_query_threads Number of threads for intra-query parallelism.
    ///
    /// @sa save, assemble_from_file
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename CentroidType,
        typename DataType,
        typename Distance,
        typename ThreadpoolProto>
    static IVF assemble(
        const std::filesystem::path& config_path,
        const std::filesystem::path& data_path,
        const Distance& distance,
        ThreadpoolProto threadpool_proto,
        size_t intra_query_threads = 1
    ) {
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            auto dispatcher = DistanceDispatcher(distance);
            return dispatcher([&](auto distance_function) {
                return IVF(
                    std::in_place,
                    manager::as_typelist<QueryTypes>{},
                    index::ivf::load_ivf_index<CentroidType, DataType>(
                        config_path,
                        data_path,
                        std::move(distance_function),
                        std::move(threadpool),
                        intra_query_threads
                    )
                );
            });
        } else {
            return IVF(
                std::in_place,
                manager::as_typelist<QueryTypes>{},
                index::ivf::load_ivf_index<CentroidType, DataType>(
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
    /// @brief Load an IVF Index from a stream.
    ///
    /// @tparam QueryTypes The element types of queries that will be used when requesting
    ///     searches over the index. Can be a single type or a ``svs::lib::Types``.
    /// @tparam CentroidType The element type of the centroids.
    /// @tparam DataType The element type of the cluster data.
    ///
    /// @param stream Input stream to load the index from.
    /// @param distance The distance functor or ``svs::DistanceType`` enum to use for
    ///     similarity search computations.
    /// @param threadpool_proto Precursor for the thread pool to use.
    /// @param intra_query_threads Number of threads for intra-query parallelism.
    ///
    /// @sa save
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename CentroidType,
        typename DataType,
        typename Distance,
        typename ThreadpoolProto>
    static IVF assemble(
        std::istream& stream,
        const Distance& distance,
        ThreadpoolProto threadpool_proto,
        size_t intra_query_threads = 1
    ) {
        namespace fs = std::filesystem;
        lib::UniqueTempDirectory tempdir{"svs_ivf_load"};
        lib::DirectoryArchiver::unpack(stream, tempdir);

        const auto config_path = tempdir.get() / "config";
        if (!fs::is_directory(config_path)) {
            throw ANNEXCEPTION("Invalid IVF index archive: missing config directory!");
        }

        const auto data_path = tempdir.get() / "data";
        if (!fs::is_directory(data_path)) {
            throw ANNEXCEPTION("Invalid IVF index archive: missing data directory!");
        }

        return assemble<QueryTypes, CentroidType, DataType>(
            config_path,
            data_path,
            distance,
            std::move(threadpool_proto),
            intra_query_threads
        );
    }

    ///// Building
    template <typename BuildType, typename DataProto, typename Distance>
    static auto build_clustering(
        const index::ivf::IVFBuildParameters& build_parameters,
        const DataProto& data_proto,
        const Distance& distance,
        size_t num_threads
    ) {
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            auto dispatcher = DistanceDispatcher(distance);
            return dispatcher([&](auto distance_function) {
                return index::ivf::build_clustering<BuildType>(
                    build_parameters, data_proto, std::move(distance_function), num_threads
                );
            });
        } else {
            return index::ivf::build_clustering<BuildType>(
                build_parameters, data_proto, distance, num_threads
            );
        }
    }
};

} // namespace svs
