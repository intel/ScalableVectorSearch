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
#include "dispatcher_vamana.hpp"

#include "algorithm.hpp"
#include "data_builder.hpp"
#include "storage.hpp"
#include "threadpool.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/vamana.h>

#include <cassert>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

namespace svs::c_runtime {

namespace {
template <typename DataBuilder, typename Distance>
svs::Vamana build_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> src_data,
    DataBuilder builder,
    Distance distance,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder
) {
    using value_type = typename DataBuilder::allocator_type::value_type;
    auto data =
        builder.build(std::move(src_data), pool, allocator_builder.build<value_type>());
    return svs::Vamana::build<float>(
        build_params,
        std::move(data),
        distance,
        std::move(pool),
        allocator_builder.build_for_graph<uint32_t>()
    );
}

template <typename DataLoader, typename Distance>
svs::Vamana load_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& SVS_UNUSED(build_params),
    const std::filesystem::path& directory,
    DataLoader loader,
    Distance distance,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder
) {
    using value_type = typename DataLoader::allocator_type::value_type;
    auto data = loader.load(directory / "data", allocator_builder.build<value_type>());
    return svs::Vamana::assemble<float>(
        directory / "config",
        svs::GraphLoader<uint32_t, AllocatorHandle<uint32_t>>{
            directory / "graph", allocator_builder.build_for_graph<uint32_t>()},
        std::move(data),
        distance,
        std::move(pool)
    );
}

template <typename Dispatcher>
void register_vamana_index_specializations(Dispatcher& dispatcher) {
    auto build_closure = [&dispatcher]<typename DataBuilder, typename Distance>() {
        dispatcher.register_target(&build_vamana_index<DataBuilder, Distance>);
    };
    auto load_closure = [&dispatcher]<typename DataLoader, typename Distance>() {
        dispatcher.register_target(&load_vamana_index<DataLoader, Distance>);
    };

    for_simple_specializations<false>(build_closure);
    for_simple_specializations<false>(load_closure);
    for_leanvec_specializations<false>(build_closure);
    for_leanvec_specializations<false>(load_closure);
    for_lvq_specializations<false>(build_closure);
    for_lvq_specializations<false>(load_closure);
    for_sq_specializations<false>(build_closure);
    for_sq_specializations<false>(load_closure);
}

using VamanaSource =
    std::variant<svs::data::ConstSimpleDataView<float>, std::filesystem::path>;

using BuildIndexDispatcher = svs::lib::Dispatcher<
    svs::Vamana,
    const svs::index::vamana::VamanaBuildParameters&,
    VamanaSource,
    const Storage*,
    svs::DistanceType,
    svs::threads::ThreadPoolHandle,
    const AllocatorBuilder&>;

const BuildIndexDispatcher& build_vamana_index_dispatcher() {
    static BuildIndexDispatcher dispatcher = [] {
        BuildIndexDispatcher d{};
        register_vamana_index_specializations(d);
        return d;
    }();
    return dispatcher;
}

using StoragePair = std::pair<const Storage*, const Storage*>;

using CopyIndexDispatcher = svs::lib::Dispatcher<
    svs::Vamana,
    const svs::index::vamana::VamanaBuildParameters&,
    const svs::Vamana&,
    const Storage*, // src
    const Storage*, // dst
    svs::DistanceType,
    svs::threads::ThreadPoolHandle,
    const AllocatorBuilder&>;

template <typename SrcDataBuilder, typename DstDataBuilder, typename Distance>
svs::Vamana copy_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const svs::Vamana& src_index,
    SrcDataBuilder src_builder,
    DstDataBuilder dst_builder,
    Distance distance,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder
) {
    using GraphType = svs::graphs::SimpleGraph<uint32_t, AllocatorHandle<uint32_t>>;
    using SrcDataType = typename SrcDataBuilder::data_type;

    using IndexImplType = svs::index::vamana::VamanaIndex<GraphType, SrcDataType, Distance>;

    auto src_index_impl =
        src_index.template get_typed_impl<svs::lib::Types<float>, IndexImplType>();
    if (!src_index_impl) {
        throw std::runtime_error("Failed to get typed index implementation");
    }

    const auto& src_data = src_builder.get_dataset(src_index_impl->view_data());

    using value_type = typename DstDataBuilder::allocator_type::value_type;
    auto data = dst_builder.build(src_data, pool, allocator_builder.build<value_type>());

    auto config = src_index.parameters();
    config.build_parameters = build_params;

    if (config.build_parameters.graph_max_degree != build_params.graph_max_degree) {
        throw not_implemented("Graph max degree mismatch");
    }

    const auto& src_graph = src_index_impl->view_graph();

    assert(src_graph.max_degree() == config.build_parameters.graph_max_degree);

    auto graph = GraphType(
        src_graph.n_nodes(),
        build_params.graph_max_degree,
        allocator_builder.build_for_graph<uint32_t>()
    );
    svs::data::copy(src_graph.get_data(), graph.get_data());

    return svs::Vamana::assemble<float>(
        config, std::move(graph), std::move(data), distance, std::move(pool)
    );
}

template <typename Dispatcher>
void register_copy_vamana_index_specializations(Dispatcher& dispatcher) {
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
            dispatcher.register_target(&copy_vamana_index<
                                       SrcDataBuilder,
                                       DstDataBuilder,
                                       Distance>);
        };

        for_simple_specializations<false>(inner_closure);
        for_sq_specializations<false>(inner_closure);
        for_lvq_specializations<false>(inner_closure);
        for_leanvec_specializations<false>(inner_closure);
    };
    for_simple_specializations<false>(compression_closure);

    // Decompression specializations for copy_vamana_index
    // To handle cases Compressed -> Simple
    auto decompression_closure = [&dispatcher]<typename SrcDataBuilder, typename Dist>() {
        // Skip all distance specializations except one
        if constexpr (!std::is_same_v<Dist, DistanceL2>) {
            return;
        }
        auto inner_closure = [&dispatcher]<typename DstDataBuilder, typename Distance>() {
            dispatcher.register_target(&copy_vamana_index<
                                       SrcDataBuilder,
                                       DstDataBuilder,
                                       Distance>);
        };

        for_simple_specializations<false>(inner_closure);
    };
    for_sq_specializations<false>(decompression_closure);
    for_lvq_specializations<false>(decompression_closure);
    for_leanvec_specializations<false>(decompression_closure);
}

const CopyIndexDispatcher& copy_vamana_index_dispatcher() {
    static CopyIndexDispatcher dispatcher = [] {
        CopyIndexDispatcher d{};
        register_copy_vamana_index_specializations(d);
        return d;
    }();
    return dispatcher;
}

} // namespace

svs::Vamana dispatch_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> data,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder
) {
    return build_vamana_index_dispatcher().invoke(
        build_params,
        VamanaSource{std::move(data)},
        storage,
        distance_type,
        std::move(pool),
        allocator_builder
    );
}

svs::Vamana dispatch_vamana_index_load(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const std::filesystem::path& directory,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder
) {
    return build_vamana_index_dispatcher().invoke(
        build_params,
        VamanaSource{directory},
        storage,
        distance_type,
        std::move(pool),
        allocator_builder
    );
}

svs::Vamana dispatch_vamana_index_copy(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const svs::Vamana& src_index,
    const Storage* src_storage,
    const Storage* dst_storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder
) {
    return copy_vamana_index_dispatcher().invoke(
        build_params,
        src_index,
        src_storage,
        dst_storage,
        distance_type,
        std::move(pool),
        allocator_builder
    );
}

svs::index::vamana::MemoryBreakdown dispatch_vamana_memory_estimate(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    size_t num_vectors,
    size_t dimension,
    const Storage* storage,
    svs::DistanceType SVS_UNUSED(distance_type)
) {
    svs::index::vamana::MemoryBreakdown breakdown{};

    // Graph: SimpleData<uint32_t> with num_vectors rows and (max_degree + 1) cols;
    // the +1 slot stores the per-node neighbor count.
    using index_type = uint32_t;
    const size_t max_degree = build_params.graph_max_degree;
    auto graph_data_builder = SimpleDataBuilder<index_type>{};
    breakdown.graph_bytes = graph_data_builder.estimate_size(num_vectors, (max_degree + 1));

    // Data: SimpleData<T> with num_vectors rows and `dimension` cols.
    breakdown.data_bytes = estimate_data_size(storage, num_vectors, dimension);
    // Metadata: single entry point held as Idx.
    breakdown.metadata_bytes = sizeof(index_type);
    return breakdown;
}
} // namespace svs::c_runtime
