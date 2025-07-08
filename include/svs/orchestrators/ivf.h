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
#include "svs/orchestrators/manager.h"

namespace svs {

class IVFInterface {
  public:
    using search_parameters_type = svs::index::ivf::IVFSearchParameters;

    ///// Beckend Information Inteface
    virtual std::string experimental_backend_string() const = 0;
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
