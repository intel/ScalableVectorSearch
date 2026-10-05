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

// Conversion between dynamic Vamana indices, shared by the regular and concurrent
// dispatchers. An index flavor provides:
// - `impl_type<Data, Distance>`: the typed index implementation;
// - `block_kind`: the data storage layout;
// - `status(impl)`: slot states in `svs::index::vamana::SlotMetadata` terms;
// - `assemble(...)`: a new index from copied state.

#include "allocator.hpp"
#include "data_builder.hpp"
#include "storage.hpp"
#include "types_support.hpp"

#include <svs/core/distance.h>
#include <svs/core/translation.h>
#include <svs/index/vamana/build_params.h>
#include <svs/index/vamana/dynamic_index.h>
#include <svs/lib/dispatcher.h>
#include <svs/lib/narrow.h>
#include <svs/lib/threads/threadpool.h>
#include <svs/orchestrators/dynamic_vamana.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace svs::c_runtime::detail {

using CopyDynamicIndexDispatcher = svs::lib::Dispatcher<
    svs::DynamicVamana,
    const svs::index::vamana::VamanaBuildParameters&,
    const svs::DynamicVamana&,
    const Storage*, // src
    const Storage*, // dst
    svs::DistanceType,
    svs::threads::ThreadPoolHandle,
    const AllocatorBuilder&,
    const svs::data::BlockingParameters&>;

struct RegularVamanaFlavor {
    using graph_type =
        svs::graphs::SimpleGraph<uint32_t, svs::data::Blocked<AllocatorHandle<uint32_t>>>;
    static constexpr BlockKind block_kind = BlockKind::Blocked;

    template <typename Data, typename Distance>
    using impl_type = svs::index::vamana::MutableVamanaIndex<graph_type, Data, Distance>;

    template <typename Impl>
    static std::vector<svs::index::vamana::SlotMetadata> status(const Impl& impl) {
        return impl.view_status();
    }

    template <typename Data, typename Distance, typename SrcGraph>
    static svs::DynamicVamana assemble(
        const svs::index::vamana::VamanaIndexParameters& config,
        Data data,
        const SrcGraph& src_graph,
        std::vector<svs::index::vamana::SlotMetadata> status,
        std::span<const size_t> external_ids,
        std::span<const uint32_t> internal_ids,
        Distance distance,
        svs::threads::ThreadPoolHandle pool,
        AllocatorHandle<uint32_t> graph_handle,
        const svs::data::BlockingParameters& block_params
    ) {
        auto graph = graph_type(
            src_graph.n_nodes(),
            config.build_parameters.graph_max_degree,
            svs::data::Blocked{block_params, std::move(graph_handle)}
        );
        svs::data::copy(src_graph.get_data(), graph.get_data());

        auto translator = svs::IDTranslator{};
        translator.insert(external_ids, internal_ids);
        svs::index::vamana::detail::VamanaStateLoader state_loader{
            config, std::move(translator), std::move(status)};

        return svs::DynamicVamana::assemble<float>(
            std::move(state_loader),
            std::move(graph),
            std::move(data),
            std::move(distance),
            std::move(pool)
        );
    }
};

template <
    typename SrcFlavor,
    typename DstFlavor,
    typename SrcDataBuilder,
    typename DstDataBuilder,
    typename Distance>
svs::DynamicVamana copy_dynamic_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const svs::DynamicVamana& src_index,
    SrcDataBuilder src_builder,
    DstDataBuilder dst_builder,
    Distance distance,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params
) {
    auto config = src_index.parameters();

    // A defaulted alpha is resolved per-metric at build time, so it cannot be compared.
    constexpr svs::index::vamana::VamanaBuildParameters default_build_params{};
    const bool alpha_is_default = build_params.alpha == default_build_params.alpha;

    // Validate build parameters match
    if (config.build_parameters.graph_max_degree != build_params.graph_max_degree ||
        (!alpha_is_default && config.build_parameters.alpha != build_params.alpha)) {
        throw not_implemented("Index build parameters mismatch");
    }

    // Other build parameters that are not explicitly checked above are updated here.
    config.build_parameters.apply(build_params);
    verify_and_set_default_index_parameters(config.build_parameters, distance);

    using SrcImpl = typename SrcFlavor::
        template impl_type<typename SrcDataBuilder::data_type, Distance>;
    auto src_index_impl =
        src_index.template get_typed_impl<svs::lib::Types<float>, SrcImpl>();
    if (!src_index_impl) {
        throw std::runtime_error("Failed to get typed index implementation");
    }

    // Only valid slots are translated: the concurrent translator keeps soft-deleted
    // entries until consolidation.
    auto status = SrcFlavor::status(*src_index_impl);
    std::vector<size_t> external_ids;
    std::vector<uint32_t> internal_ids;
    for (auto pair : src_index_impl->view_translator()) {
        if (status.at(pair.second) == svs::index::vamana::SlotMetadata::Valid) {
            external_ids.push_back(pair.first);
            internal_ids.push_back(lib::narrow<uint32_t>(pair.second));
        }
    }

    const auto src_data = src_builder.get_dataset(src_index_impl->view_data());

    using allocator_type = typename DstDataBuilder::allocator_type;
    using value_type = typename allocator_type::value_type;
    auto data_allocator =
        allocator_type{block_params, allocator_builder.build<value_type>()};
    auto data = dst_builder.build(src_data, pool, data_allocator);

    return DstFlavor::assemble(
        config,
        std::move(data),
        src_index_impl->view_graph(),
        std::move(status),
        external_ids,
        internal_ids,
        std::move(distance),
        std::move(pool),
        allocator_builder.build_for_graph<uint32_t>(),
        block_params
    );
}

template <typename SrcFlavor, typename DstFlavor>
void register_copy_specializations(CopyDynamicIndexDispatcher& dispatcher) {
    // TODO: Enable Compressed -> Compressed specializations by making decompressors,
    // decompression accessors and decompression dataset are thread-safe
    auto register_pair = [&dispatcher]<typename SrcDataBuilder>() {
        return [&dispatcher]<typename DstDataBuilder, typename Distance>() {
            dispatcher.register_target(&copy_dynamic_vamana_index<
                                       SrcFlavor,
                                       DstFlavor,
                                       SrcDataBuilder,
                                       DstDataBuilder,
                                       Distance>);
        };
    };

    // Simple -> any; one distance is enough to enumerate the source types.
    auto compression_closure = [&]<typename SrcDataBuilder, typename Dist>() {
        if constexpr (std::is_same_v<Dist, DistanceL2>) {
            auto inner = register_pair.template operator()<SrcDataBuilder>();
            for_simple_specializations<DstFlavor::block_kind>(inner);
            for_sq_specializations<DstFlavor::block_kind>(inner);
            for_lvq_specializations<DstFlavor::block_kind>(inner);
            for_leanvec_specializations<DstFlavor::block_kind>(inner);
        }
    };
    for_simple_specializations<SrcFlavor::block_kind>(compression_closure);

    // Compressed -> Simple
    auto decompression_closure = [&]<typename SrcDataBuilder, typename Dist>() {
        if constexpr (std::is_same_v<Dist, DistanceL2>) {
            auto inner = register_pair.template operator()<SrcDataBuilder>();
            for_simple_specializations<DstFlavor::block_kind>(inner);
        }
    };
    for_sq_specializations<SrcFlavor::block_kind>(decompression_closure);
    for_lvq_specializations<SrcFlavor::block_kind>(decompression_closure);
    for_leanvec_specializations<SrcFlavor::block_kind>(decompression_closure);
}

template <typename SrcFlavor, typename DstFlavor>
const CopyDynamicIndexDispatcher& copy_dynamic_index_dispatcher() {
    static const CopyDynamicIndexDispatcher dispatcher = [] {
        CopyDynamicIndexDispatcher d{};
        register_copy_specializations<SrcFlavor, DstFlavor>(d);
        return d;
    }();
    return dispatcher;
}

} // namespace svs::c_runtime::detail
