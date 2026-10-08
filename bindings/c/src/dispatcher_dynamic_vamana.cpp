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
#include "dispatcher_dynamic_vamana.hpp"

#include "algorithm.hpp"
#include "allocator.hpp"
#include "data_builder.hpp"
#include "storage.hpp"
#include "threadpool.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/index/vamana/dynamic_index.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>

#include <filesystem>
#include <istream>
#include <memory>
#include <span>
#include <utility>
#include <variant>

namespace svs::c_runtime {

namespace {
template <typename DataBuilder, typename Distance>
svs::DynamicVamana build_dynamic_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    std::pair<svs::data::ConstSimpleDataView<float>, std::span<const size_t>> src_data,
    DataBuilder builder,
    Distance D,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    size_t blocksize_bytes
) {
    svs::data::BlockingParameters block_params;
    if (blocksize_bytes != 0) {
        block_params.blocksize_bytes = svs::lib::prevpow2(blocksize_bytes);
    }
    using allocator_type = typename DataBuilder::allocator_type;
    using value_type = typename allocator_type::value_type;

    auto data_allocator_handle = allocator_builder.build<value_type>();
    auto data_allocator = allocator_type{block_params, data_allocator_handle};
    auto data = builder.build(std::move(src_data.first), pool, data_allocator);

    auto graph_allocator_handle = allocator_builder.build_for_graph<uint32_t>();
    auto graph_allocator = svs::data::Blocked{block_params, graph_allocator_handle};
    return svs::DynamicVamana::build<float>(
        build_params,
        std::move(data),
        std::move(src_data.second),
        std::move(D),
        std::move(pool),
        graph_allocator
    );
}

template <typename DataLoader, typename Distance>
svs::DynamicVamana load_dynamic_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& SVS_UNUSED(build_params),
    const std::filesystem::path& directory,
    DataLoader loader,
    Distance D,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    size_t blocksize_bytes
) {
    svs::data::BlockingParameters block_params;
    if (blocksize_bytes != 0) {
        block_params.blocksize_bytes = svs::lib::prevpow2(blocksize_bytes);
    }
    using allocator_type = typename DataLoader::allocator_type;
    using value_type = typename allocator_type::value_type;
    auto data_allocator_handle = allocator_builder.build<value_type>();
    auto allocator = allocator_type{block_params, data_allocator_handle};
    auto data = loader.load(directory / "data", allocator);

    auto graph_allocator_handle = allocator_builder.build_for_graph<uint32_t>();
    auto graph_allocator = svs::data::Blocked{block_params, graph_allocator_handle};

    return svs::DynamicVamana::assemble<float>(
        directory / "config",
        svs::GraphLoader{directory / "graph", graph_allocator},
        std::move(data),
        std::move(D),
        std::move(pool)
    );
}

template <typename DataLoader, typename Distance>
svs::DynamicVamana load_stream_dynamic_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& SVS_UNUSED(build_params),
    std::unique_ptr<std::istream> stream,
    DataLoader SVS_UNUSED(loader),
    Distance distance,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    size_t blocksize_bytes
) {
    svs::data::BlockingParameters block_params;
    if (blocksize_bytes != 0) {
        block_params.blocksize_bytes = svs::lib::prevpow2(blocksize_bytes);
    }
    using allocator_type = typename DataLoader::allocator_type;
    using value_type = typename allocator_type::value_type;
    using data_type = typename DataLoader::data_type;
    auto data_allocator_handle = allocator_builder.build<value_type>();
    auto allocator = allocator_type{block_params, data_allocator_handle};

    auto graph_allocator_handle = allocator_builder.build_for_graph<uint32_t>();
    auto graph_allocator = svs::data::Blocked{block_params, graph_allocator_handle};

    // svs_c.h lets the caller drop the stream once loading returns. That holds only while
    // assemble copies data out; a view allocator here would leave the index dangling.
    return svs::DynamicVamana::assemble<float, data_type>(
        *stream, distance, std::move(pool), allocator, graph_allocator
    );
}

template <typename Dispatcher>
void register_dynamic_vamana_index_specializations(Dispatcher& dispatcher) {
    auto build_closure = [&dispatcher]<typename DataBuilder, typename Distance>() {
        dispatcher.register_target(&build_dynamic_vamana_index<DataBuilder, Distance>);
    };
    auto load_closure = [&dispatcher]<typename DataLoader, typename Distance>() {
        dispatcher.register_target(&load_dynamic_vamana_index<DataLoader, Distance>);
    };
    auto load_stream_closure = [&dispatcher]<typename DataLoader, typename Distance>() {
        dispatcher.register_target(&load_stream_dynamic_vamana_index<DataLoader, Distance>);
    };

    for_simple_specializations<true>(build_closure);
    for_simple_specializations<true>(load_closure);
    for_simple_specializations<true>(load_stream_closure);
    for_leanvec_specializations<true>(build_closure);
    for_leanvec_specializations<true>(load_closure);
    for_leanvec_specializations<true>(load_stream_closure);
    for_lvq_specializations<true>(build_closure);
    for_lvq_specializations<true>(load_closure);
    for_lvq_specializations<true>(load_stream_closure);
    for_sq_specializations<true>(build_closure);
    for_sq_specializations<true>(load_closure);
    for_sq_specializations<true>(load_stream_closure);
}

// Stream load alternative, matched via generic variant DispatchConverter like the
// existing build and directory-load alternatives.
using DynamicVamanaSource = std::variant<
    std::pair<svs::data::ConstSimpleDataView<float>, std::span<const size_t>>,
    std::filesystem::path,
    std::unique_ptr<std::istream>>;

using BuildDynamicIndexDispatcher = svs::lib::Dispatcher<
    svs::DynamicVamana,
    const svs::index::vamana::VamanaBuildParameters&,
    DynamicVamanaSource,
    const Storage*,
    svs::DistanceType,
    svs::threads::ThreadPoolHandle,
    const AllocatorBuilder&,
    size_t>;

const BuildDynamicIndexDispatcher& build_dynamic_vamana_index_dispatcher() {
    static BuildDynamicIndexDispatcher dispatcher = [] {
        BuildDynamicIndexDispatcher d{};
        register_dynamic_vamana_index_specializations(d);
        return d;
    }();
    return dispatcher;
}

using CopyDynamicIndexDispatcher = svs::lib::Dispatcher<
    svs::DynamicVamana,
    const svs::index::vamana::VamanaBuildParameters&,
    const svs::DynamicVamana&,
    const Storage*, // src
    const Storage*, // dst
    svs::DistanceType,
    svs::threads::ThreadPoolHandle,
    const AllocatorBuilder&,
    size_t>;

template <typename SrcDataBuilder, typename DstDataBuilder, typename Distance>
svs::DynamicVamana copy_dynamic_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const svs::DynamicVamana& src_index,
    SrcDataBuilder src_builder,
    DstDataBuilder dst_builder,
    Distance distance,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    size_t blocksize_bytes
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

    // Determine the blocking parameters based on the provided block size
    svs::data::BlockingParameters block_params;
    if (blocksize_bytes != 0) {
        block_params.blocksize_bytes = svs::lib::prevpow2(blocksize_bytes);
    }

    // Must match the graph type produced by the build and load paths above.
    using GraphType =
        svs::graphs::SimpleGraph<uint32_t, svs::data::Blocked<AllocatorHandle<uint32_t>>>;

    // Get the typed index implementation from the source index
    using SrcDataType = typename SrcDataBuilder::data_type;
    using IndexImplType =
        svs::index::vamana::MutableVamanaIndex<GraphType, SrcDataType, Distance>;
    auto src_index_impl =
        src_index.template get_typed_impl<svs::lib::Types<float>, IndexImplType>();
    if (!src_index_impl) {
        throw std::runtime_error("Failed to get typed index implementation");
    }

    // Copy the graph structure from the source index to the new graph instance
    const auto& src_graph = src_index_impl->view_graph();
    assert(src_graph.max_degree() == config.build_parameters.graph_max_degree);

    auto graph_allocator_handle = allocator_builder.build_for_graph<uint32_t>();
    auto graph_allocator = svs::data::Blocked{block_params, graph_allocator_handle};
    auto graph =
        GraphType(src_graph.n_nodes(), build_params.graph_max_degree, graph_allocator);
    svs::data::copy(src_graph.get_data(), graph.get_data());

    // Copy/convert the data from the source index to the new data instance
    decltype(auto) src_data = src_builder.get_dataset(src_index_impl->view_data());

    using allocator_type = typename DstDataBuilder::allocator_type;
    using value_type = typename allocator_type::value_type;

    auto data_allocator_handle = allocator_builder.build<value_type>();
    auto data_allocator = allocator_type{block_params, data_allocator_handle};
    auto data = dst_builder.build(src_data, pool, data_allocator);

    svs::index::vamana::detail::VamanaStateLoader state_loader{
        config, src_index_impl->view_translator(), src_index_impl->view_status()};

    return svs::DynamicVamana::assemble<float>(
        std::move(state_loader),
        std::move(graph),
        std::move(data),
        distance,
        std::move(pool)
    );
}

template <typename Dispatcher> void register_copy_specializations(Dispatcher& dispatcher) {
    // TODO: Enable Compressed -> Compressed specializations by making decompressors,
    // decompression accessors and decompression dataset are thread-safe

    // Compression specializations for copy_vamana_index
    // To handle cases Simple -> Compressed
    auto compression_closure = [&dispatcher]<typename SrcDataBuilder, typename Dist>() {
        // Skip all distance specializations except one
        if constexpr (!std::is_same_v<Dist, DistanceL2>) {
            return;
        }
        auto inner_closure = [&dispatcher]<typename DstDataBuilder, typename Distance>() {
            dispatcher.register_target(&copy_dynamic_vamana_index<
                                       SrcDataBuilder,
                                       DstDataBuilder,
                                       Distance>);
        };

        for_simple_specializations<true>(inner_closure);
        for_sq_specializations<true>(inner_closure);
        for_lvq_specializations<true>(inner_closure);
        for_leanvec_specializations<true>(inner_closure);
    };
    for_simple_specializations<true>(compression_closure);

    // Decompression specializations for copy_vamana_index
    // To handle cases Compressed -> Simple
    auto decompression_closure = [&dispatcher]<typename SrcDataBuilder, typename Dist>() {
        // Skip all distance specializations except one
        if constexpr (!std::is_same_v<Dist, DistanceL2>) {
            return;
        }
        auto inner_closure = [&dispatcher]<typename DstDataBuilder, typename Distance>() {
            dispatcher.register_target(&copy_dynamic_vamana_index<
                                       SrcDataBuilder,
                                       DstDataBuilder,
                                       Distance>);
        };

        for_simple_specializations<true>(inner_closure);
    };
    for_sq_specializations<true>(decompression_closure);
    for_lvq_specializations<true>(decompression_closure);
    for_leanvec_specializations<true>(decompression_closure);
}

const CopyDynamicIndexDispatcher& copy_dynamic_index_dispatcher() {
    static CopyDynamicIndexDispatcher dispatcher = [] {
        CopyDynamicIndexDispatcher d{};
        register_copy_specializations(d);
        return d;
    }();
    return dispatcher;
}

} // namespace

svs::DynamicVamana dispatch_dynamic_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> data,
    std::span<const size_t> ids,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    size_t blocksize_bytes
) {
    return build_dynamic_vamana_index_dispatcher().invoke(
        build_params,
        DynamicVamanaSource{std::make_pair(data, ids)},
        storage,
        distance_type,
        std::move(pool),
        allocator_builder,
        blocksize_bytes
    );
}

svs::DynamicVamana dispatch_dynamic_vamana_index_load(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const std::filesystem::path& directory,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    size_t blocksize_bytes
) {
    return build_dynamic_vamana_index_dispatcher().invoke(
        build_params,
        DynamicVamanaSource{directory},
        storage,
        distance_type,
        std::move(pool),
        allocator_builder,
        blocksize_bytes
    );
}

svs::DynamicVamana dispatch_dynamic_vamana_index_load_stream(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    std::unique_ptr<std::istream> stream,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    size_t blocksize_bytes
) {
    return build_dynamic_vamana_index_dispatcher().invoke(
        build_params,
        DynamicVamanaSource{std::move(stream)},
        storage,
        distance_type,
        std::move(pool),
        allocator_builder,
        blocksize_bytes
    );
}

svs::DynamicVamana dispatch_dynamic_vamana_index_copy(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const svs::DynamicVamana& src_index,
    const Storage* src_storage,
    const Storage* dst_storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    size_t blocksize_bytes
) {
    return copy_dynamic_index_dispatcher().invoke(
        build_params,
        src_index,
        src_storage,
        dst_storage,
        distance_type,
        std::move(pool),
        allocator_builder,
        blocksize_bytes
    );
}

svs::index::vamana::MemoryBreakdown dispatch_dynamic_vamana_memory_estimate(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    size_t num_vectors,
    size_t dimension,
    const Storage* storage,
    svs::DistanceType SVS_UNUSED(distance_type),
    size_t blocksize_bytes
) {
    svs::index::vamana::MemoryBreakdown breakdown{};

    // Graph size
    // Graph: SimpleBlockedData<uint32_t> with num_vectors rows and (max_degree + 1)
    // cols; the +1 slot stores the per-node neighbor count.
    using index_type = uint32_t;
    using graph_allocator_type = svs::data::Blocked<svs::lib::Allocator<index_type>>;

    const size_t max_degree = build_params.graph_max_degree;

    svs::data::BlockingParameters block_params;
    if (blocksize_bytes != 0) {
        block_params.blocksize_bytes = svs::lib::prevpow2(blocksize_bytes);
    }
    auto graph_allocator = graph_allocator_type{block_params};
    auto graph_data_builder = svs::SimpleDataBuilder<index_type, graph_allocator_type>{};

    breakdown.graph_bytes =
        graph_data_builder.estimate_size(num_vectors, (max_degree + 1), graph_allocator);

    // Data size
    breakdown.data_bytes =
        estimate_data_size_blocked(storage, num_vectors, dimension, blocksize_bytes);

    // Metadata: single entry point held as Idx, plus the SlotMetadata vector, plus the
    // IDTranslator maps.
    size_t metadata_bytes =
        sizeof(index_type) + sizeof(svs::index::vamana::SlotMetadata) * num_vectors;
    // The IDTranslator holds two tsl::robin_map instances (external->internal and
    // internal->external), neither of which exposes its allocated byte count. We
    // approximate the storage as the id pair held in each of the two directions. This
    // ignores the maps' load-factor slack and control bytes, so it is an estimate of
    // the hash-map overhead that is accurate to within a few percent.
    metadata_bytes +=
        2 * num_vectors *
        (sizeof(IDTranslator::external_id_type) + sizeof(IDTranslator::internal_id_type));
    breakdown.metadata_bytes = metadata_bytes;
    return breakdown;
}
} // namespace svs::c_runtime
