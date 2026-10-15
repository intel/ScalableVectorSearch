/*
 * Copyright 2026 Intel Corporation
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

#include "svs/concurrent/blocked_data.h"
#include "svs/concurrent/dynamic_index.h"
#include "svs/concurrent/graph.h"
#include "svs/concurrent/iterator.h"

#include "svs/orchestrators/dynamic_vamana.h"

// stdlib
#include <filesystem>
#include <istream>
#include <span>
#include <type_traits>

namespace svs {

///
/// @brief Type-erased wrapper for ``svs::index::vamana::concurrent::MutableVamanaIndex``.
///
/// Exposes the same API as ``svs::DynamicVamana``; only the factory functions differ.
/// The wrapped index may be searched and mutated from several threads at once: search,
/// batch iteration, ``add_points``, ``delete_points``, ``consolidate``, ``compact``, ID
/// inspection, ``get_distance`` and ``reconstruct_at`` are safe to call concurrently.
/// ``save``, ``set_threadpool`` and the build-parameter setters are not; call them only
/// when no other thread uses the index. See ``include/svs/concurrent/README.md``.
///
/// The dataset must use a ``svs::index::vamana::concurrent::SegmentedBlocked`` allocator so
/// that growing it never relocates storage under concurrent readers.
///
class ConcurrentDynamicVamana : public DynamicVamana {
  public:
    using base_type = DynamicVamana;
    using default_graph_allocator_type =
        index::vamana::concurrent::SegmentedBlocked<HugepageAllocator<uint32_t>>;
    using default_graph_type =
        index::vamana::concurrent::graphs::SimpleBlockedGraph<uint32_t>;

    template <lib::TypeList QueryTypes, typename Impl>
    explicit ConcurrentDynamicVamana(AssembleTag tag, QueryTypes types, Impl impl)
        : base_type{tag, types, std::move(impl)} {}

    ///
    /// @brief Construct a ConcurrentDynamicVamana index from a data loader or dataset.
    ///
    /// @tparam QueryTypes The set of query element types supported by the resulting index.
    ///
    /// @param parameters Build parameters controlling graph construction.
    /// @param data_loader Loader (or dataset) producing ``SegmentedBlocked`` storage.
    /// @param ids External IDs to assign to each row; must be unique.
    /// @param distance Distance functor or ``svs::DistanceType`` enum.
    /// @param threadpool_proto Thread pool or number of threads to use.
    /// @param graph_allocator ``SegmentedBlocked`` allocator used for the graph.
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename DataLoader,
        typename Distance,
        typename ThreadPoolProto,
        typename GraphAllocator = default_graph_allocator_type>
    static ConcurrentDynamicVamana build(
        const index::vamana::VamanaBuildParameters& parameters,
        DataLoader&& data_loader,
        std::span<const size_t> ids,
        Distance distance,
        ThreadPoolProto threadpool_proto,
        const GraphAllocator& graph_allocator = {}
    ) {
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
        auto data =
            svs::detail::dispatch_load(std::forward<DataLoader>(data_loader), threadpool);
        auto make = [&](auto distance_function) {
            return ConcurrentDynamicVamana(
                AssembleTag{},
                manager::as_typelist<QueryTypes>(),
                index::vamana::concurrent::auto_dynamic_build(
                    parameters,
                    std::move(data),
                    ids,
                    std::move(distance_function),
                    std::move(threadpool),
                    graph_allocator
                )
            );
        };
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            return DistanceDispatcher(distance)(make);
        } else {
            return make(std::move(distance));
        }
    }

    ///
    /// @brief Reload a ConcurrentDynamicVamana index from separate config, graph and data.
    ///
    /// @param config_proto Directory holding the saved index configuration, or an
    ///     already-loaded ``index::vamana::concurrent::detail::VamanaStateLoader``.
    /// @param graph_loader Loader (or graph) producing a concurrent ``SimpleBlockedGraph``.
    /// @param data_loader Loader (or dataset) producing ``SegmentedBlocked`` storage.
    /// @param distance Distance functor or ``svs::DistanceType`` enum.
    /// @param threadpool_proto Thread pool or number of threads to use.
    /// @param debug_load_from_static Load a static index config with identity IDs.
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename ConfigProto,
        typename GraphLoader,
        typename DataLoader,
        typename Distance,
        typename ThreadPoolProto>
    static ConcurrentDynamicVamana assemble(
        ConfigProto&& config_proto,
        GraphLoader&& graph_loader,
        DataLoader&& data_loader,
        Distance distance,
        ThreadPoolProto threadpool_proto,
        bool debug_load_from_static = false
    ) {
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
        auto make = [&](auto distance_function) {
            return ConcurrentDynamicVamana(
                AssembleTag{},
                manager::as_typelist<QueryTypes>(),
                index::vamana::concurrent::auto_dynamic_assemble(
                    std::forward<ConfigProto>(config_proto),
                    std::forward<GraphLoader>(graph_loader),
                    std::forward<DataLoader>(data_loader),
                    std::move(distance_function),
                    std::move(threadpool),
                    debug_load_from_static
                )
            );
        };
        if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
            return DistanceDispatcher(distance)(make);
        } else {
            return make(std::move(distance));
        }
    }

    ///
    /// @brief Reload a ConcurrentDynamicVamana index saved with ``save(std::ostream&)``.
    ///
    /// Accepts both the native stream format and the directory-archive format.
    ///
    /// @tparam Data The ``SegmentedBlocked`` dataset type to load.
    ///
    template <
        manager::QueryTypeDefinition QueryTypes,
        typename Data,
        typename Distance,
        typename ThreadPoolProto,
        typename... DataLoaderArgs>
    static ConcurrentDynamicVamana assemble(
        std::istream& stream,
        Distance distance,
        ThreadPoolProto threadpool_proto,
        DataLoaderArgs&&... data_args
    ) {
        static_assert(
            index::vamana::concurrent::is_segmented_blocked_v<
                typename Data::allocator_type>,
            "The concurrent index requires a dataset with a SegmentedBlocked allocator."
        );
        auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
        auto deserializer = svs::lib::detail::Deserializer::build(stream);
        if (deserializer.is_native()) {
            auto make = [&](auto distance_function) {
                return ConcurrentDynamicVamana(
                    AssembleTag{},
                    manager::as_typelist<QueryTypes>(),
                    index::vamana::concurrent::auto_dynamic_assemble(
                        stream,
                        [&]() -> default_graph_type {
                            return default_graph_type::load(stream);
                        },
                        [&]() -> Data {
                            return lib::load_from_stream<Data>(
                                stream, SVS_FWD(data_args)...
                            );
                        },
                        std::move(distance_function),
                        std::move(threadpool)
                    )
                );
            };
            if constexpr (std::is_same_v<std::decay_t<Distance>, DistanceType>) {
                return DistanceDispatcher(distance)(make);
            } else {
                return make(std::move(distance));
            }
        }

        namespace fs = std::filesystem;
        lib::UniqueTempDirectory tempdir{"svs_concurrent_vamana_load"};
        lib::DirectoryArchiver::unpack(stream, tempdir, deserializer.magic());

        const auto config_path = tempdir.get() / "config";
        const auto graph_path = tempdir.get() / "graph";
        const auto data_path = tempdir.get() / "data";
        for (const auto& path : {config_path, graph_path, data_path}) {
            if (!fs::is_directory(path)) {
                throw ANNEXCEPTION(
                    "Invalid Vamana index archive: missing {} directory!",
                    path.filename().string()
                );
            }
        }

        return assemble<QueryTypes>(
            config_path,
            SVS_LAZY(default_graph_type::load(graph_path)),
            lib::load_from_disk<Data>(data_path, SVS_FWD(data_args)...),
            std::move(distance),
            std::move(threadpool)
        );
    }
};

} // namespace svs
