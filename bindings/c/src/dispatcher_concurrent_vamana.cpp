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
#include "dispatcher_concurrent_vamana.hpp"

#include "allocator.hpp"
#include "data_builder.hpp"
#include "dispatcher_dynamic_vamana.hpp"
#include "dispatcher_dynamic_vamana_copy.hpp"
#include "storage.hpp"
#include "types_support.hpp"

#include <svs/concurrent/blocked_data.h>
#include <svs/concurrent/dynamic_index.h>
#include <svs/concurrent/graph.h>
#include <svs/concurrent/spinlock.h>
#include <svs/core/distance.h>
#include <svs/index/vamana/build_params.h>
#include <svs/orchestrators/concurrent_dynamic_vamana.h>

#include <filesystem>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace svs::c_runtime {

namespace {

namespace cc = svs::index::vamana::concurrent;
using RegularSlot = svs::index::vamana::SlotMetadata;

using ConcurrentGraph = cc::graphs::SimpleBlockedGraph<uint32_t, AllocatorHandle<uint32_t>>;
using ConcurrentGraphAllocator = cc::SegmentedBlocked<AllocatorHandle<uint32_t>>;

RegularSlot to_regular_slot(cc::SlotMetadata s) {
    switch (s) {
        case cc::SlotMetadata::Empty:
            return RegularSlot::Empty;
        case cc::SlotMetadata::Valid:
            return RegularSlot::Valid;
        case cc::SlotMetadata::Deleted:
            return RegularSlot::Deleted;
        case cc::SlotMetadata::Pending:
            break;
    }
    throw ANNEXCEPTION("Cannot convert an index with in-flight insertions");
}

cc::SlotMetadata to_concurrent_slot(RegularSlot s) {
    switch (s) {
        case RegularSlot::Empty:
            return cc::SlotMetadata::Empty;
        case RegularSlot::Valid:
            return cc::SlotMetadata::Valid;
        case RegularSlot::Deleted:
            return cc::SlotMetadata::Deleted;
    }
    throw ANNEXCEPTION("Unknown slot state");
}

// Copy flavor for svs::index::vamana::concurrent::MutableVamanaIndex.
struct ConcurrentVamanaFlavor {
    static constexpr BlockKind block_kind = BlockKind::Segmented;

    template <typename Data, typename Distance>
    using impl_type = cc::MutableVamanaIndex<ConcurrentGraph, Data, Distance>;

    template <typename Impl> static std::vector<RegularSlot> status(const Impl& impl) {
        auto snapshot = impl.status_snapshot();
        std::vector<RegularSlot> result;
        result.reserve(snapshot.size());
        for (auto s : snapshot) {
            result.push_back(to_regular_slot(s));
        }
        return result;
    }

    template <typename Data, typename Distance, typename SrcGraph>
    static svs::DynamicVamana assemble(
        const svs::index::vamana::VamanaIndexParameters& config,
        Data data,
        const SrcGraph& src_graph,
        std::vector<RegularSlot> status,
        std::span<const size_t> external_ids,
        std::span<const uint32_t> internal_ids,
        Distance distance,
        svs::threads::ThreadPoolHandle pool,
        AllocatorHandle<uint32_t> graph_handle,
        const svs::data::BlockingParameters& block_params
    ) {
        auto graph = ConcurrentGraph(
            src_graph.n_nodes(),
            config.build_parameters.graph_max_degree,
            ConcurrentGraphAllocator{block_params, std::move(graph_handle)}
        );
        svs::data::copy(src_graph.get_data(), graph.get_data());

        auto translator = cc::IDTranslator{};
        translator.insert(external_ids, internal_ids);
        auto concurrent_status = std::vector<cc::SlotMetadata>();
        concurrent_status.reserve(status.size());
        for (auto s : status) {
            concurrent_status.push_back(to_concurrent_slot(s));
        }

        return svs::ConcurrentDynamicVamana(
            svs::DynamicVamana::AssembleTag{},
            svs::lib::Types<float>{},
            impl_type<Data, Distance>(
                config,
                std::move(data),
                std::move(graph),
                std::move(distance),
                concurrent_status,
                std::move(translator),
                std::move(pool)
            )
        );
    }
};

template <typename DataBuilder, typename Distance>
svs::DynamicVamana build_concurrent_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    std::pair<svs::data::ConstSimpleDataView<float>, std::span<const size_t>> src_data,
    DataBuilder builder,
    Distance D,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params
) {
    using allocator_type = typename DataBuilder::allocator_type;
    using value_type = typename allocator_type::value_type;

    auto data_allocator =
        allocator_type{block_params, allocator_builder.build<value_type>()};
    auto data = builder.build(std::move(src_data.first), pool, data_allocator);

    auto graph_allocator = ConcurrentGraphAllocator{
        block_params, allocator_builder.build_for_graph<uint32_t>()};
    return svs::ConcurrentDynamicVamana::build<float>(
        build_params,
        std::move(data),
        src_data.second,
        std::move(D),
        std::move(pool),
        graph_allocator
    );
}

template <typename DataLoader, typename Distance>
svs::DynamicVamana load_concurrent_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& SVS_UNUSED(build_params),
    const std::filesystem::path& directory,
    DataLoader loader,
    Distance D,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params
) {
    using allocator_type = typename DataLoader::allocator_type;
    using value_type = typename allocator_type::value_type;
    auto allocator = allocator_type{block_params, allocator_builder.build<value_type>()};
    auto data = loader.load(directory / "data", allocator);

    auto graph_allocator = ConcurrentGraphAllocator{
        block_params, allocator_builder.build_for_graph<uint32_t>()};
    return svs::ConcurrentDynamicVamana::assemble<float>(
        directory / "config",
        SVS_LAZY(ConcurrentGraph::load(directory / "graph", graph_allocator)),
        std::move(data),
        std::move(D),
        std::move(pool)
    );
}

using ConcurrentVamanaSource = std::variant<
    std::pair<svs::data::ConstSimpleDataView<float>, std::span<const size_t>>,
    std::filesystem::path>;

using BuildConcurrentIndexDispatcher = svs::lib::Dispatcher<
    svs::DynamicVamana,
    const svs::index::vamana::VamanaBuildParameters&,
    ConcurrentVamanaSource,
    const Storage*,
    svs::DistanceType,
    svs::threads::ThreadPoolHandle,
    const AllocatorBuilder&,
    const svs::data::BlockingParameters&>;

const BuildConcurrentIndexDispatcher& build_concurrent_vamana_index_dispatcher() {
    static const BuildConcurrentIndexDispatcher dispatcher = [] {
        BuildConcurrentIndexDispatcher d{};
        auto build_closure = [&d]<typename DataBuilder, typename Distance>() {
            d.register_target(&build_concurrent_vamana_index<DataBuilder, Distance>);
        };
        auto load_closure = [&d]<typename DataLoader, typename Distance>() {
            d.register_target(&load_concurrent_vamana_index<DataLoader, Distance>);
        };

        constexpr auto kind = ConcurrentVamanaFlavor::block_kind;
        for_simple_specializations<kind>(build_closure);
        for_simple_specializations<kind>(load_closure);
        for_leanvec_specializations<kind>(build_closure);
        for_leanvec_specializations<kind>(load_closure);
        for_lvq_specializations<kind>(build_closure);
        for_lvq_specializations<kind>(load_closure);
        for_sq_specializations<kind>(build_closure);
        for_sq_specializations<kind>(load_closure);
        return d;
    }();
    return dispatcher;
}

const detail::CopyDynamicIndexDispatcher&
copy_concurrent_index_dispatcher(bool src_concurrent, bool dst_concurrent) {
    using detail::copy_dynamic_index_dispatcher;
    using detail::RegularVamanaFlavor;
    if (src_concurrent) {
        return dst_concurrent ? copy_dynamic_index_dispatcher<
                                    ConcurrentVamanaFlavor,
                                    ConcurrentVamanaFlavor>()
                              : copy_dynamic_index_dispatcher<
                                    ConcurrentVamanaFlavor,
                                    RegularVamanaFlavor>();
    }
    INVALID_ARGUMENT_IF(!dst_concurrent, "Neither index is concurrent");
    return copy_dynamic_index_dispatcher<RegularVamanaFlavor, ConcurrentVamanaFlavor>();
}

} // namespace

svs::DynamicVamana dispatch_concurrent_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> data,
    std::span<const size_t> ids,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params
) {
    return build_concurrent_vamana_index_dispatcher().invoke(
        build_params,
        ConcurrentVamanaSource{std::make_pair(data, ids)},
        storage,
        distance_type,
        std::move(pool),
        allocator_builder,
        block_params
    );
}

svs::DynamicVamana dispatch_concurrent_vamana_index_load(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const std::filesystem::path& directory,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params
) {
    return build_concurrent_vamana_index_dispatcher().invoke(
        build_params,
        ConcurrentVamanaSource{directory},
        storage,
        distance_type,
        std::move(pool),
        allocator_builder,
        block_params
    );
}

svs::DynamicVamana dispatch_concurrent_vamana_index_copy(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const svs::DynamicVamana& src_index,
    bool src_concurrent,
    const Storage* src_storage,
    const Storage* dst_storage,
    bool dst_concurrent,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params
) {
    return copy_concurrent_index_dispatcher(src_concurrent, dst_concurrent)
        .invoke(
            build_params,
            src_index,
            src_storage,
            dst_storage,
            distance_type,
            std::move(pool),
            allocator_builder,
            block_params
        );
}

svs::index::vamana::MemoryBreakdown dispatch_concurrent_vamana_memory_estimate(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    size_t num_vectors,
    size_t dimension,
    const Storage* storage,
    svs::DistanceType distance_type,
    const svs::data::BlockingParameters& block_params
) {
    auto breakdown = dispatch_dynamic_vamana_memory_estimate(
        build_params, num_vectors, dimension, storage, distance_type, block_params
    );
    // One in-neighbor list per node; each edge is recorded once in reverse.
    using index_type = uint32_t;
    breakdown.reverse_edges_bytes =
        num_vectors * (sizeof(std::vector<index_type>) + sizeof(cc::SpinLock) +
                       build_params.graph_max_degree * sizeof(index_type));
    return breakdown;
}

} // namespace svs::c_runtime
