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
#include "index.hpp"
#include "storage.hpp"
#include "threadpool.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>

#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <variant>

namespace svs::c_runtime {

template <typename DataBuilder, typename Distance>
svs::DynamicVamana build_dynamic_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    std::pair<svs::data::ConstSimpleDataView<float>, std::span<const size_t>> src_data,
    DataBuilder builder,
    Distance D,
    svs::threads::ThreadPoolHandle pool,
    size_t blocksize_bytes
) {
    svs::data::BlockingParameters block_params;
    if (blocksize_bytes != 0) {
        block_params.blocksize_bytes = svs::lib::prevpow2(blocksize_bytes);
    }
    using allocator_type = typename DataBuilder::allocator_type;
    auto allocator = allocator_type{block_params};
    auto data = builder.build(std::move(src_data.first), pool, allocator);
    return svs::DynamicVamana::build<float>(
        build_params,
        std::move(data),
        std::move(src_data.second),
        std::move(D),
        std::move(pool)
    );
}

template <typename DataLoader, typename Distance>
svs::DynamicVamana load_dynamic_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& SVS_UNUSED(build_params),
    const std::filesystem::path& directory,
    DataLoader loader,
    Distance D,
    svs::threads::ThreadPoolHandle pool,
    size_t blocksize_bytes
) {
    svs::data::BlockingParameters block_params;
    if (blocksize_bytes != 0) {
        block_params.blocksize_bytes = svs::lib::prevpow2(blocksize_bytes);
    }
    using allocator_type = typename DataLoader::allocator_type;
    auto allocator = allocator_type{block_params};
    auto data = loader.load(directory / "data", allocator);
    return svs::DynamicVamana::assemble<float>(
        directory / "config",
        svs::GraphLoader{directory / "graph"},
        std::move(data),
        std::move(D),
        std::move(pool)
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

    for_simple_specializations<true>(build_closure);
    for_simple_specializations<true>(load_closure);
    for_leanvec_specializations<true>(build_closure);
    for_leanvec_specializations<true>(load_closure);
    for_lvq_specializations<true>(build_closure);
    for_lvq_specializations<true>(load_closure);
    for_sq_specializations<true>(build_closure);
    for_sq_specializations<true>(load_closure);
}

using DynamicVamanaSource = std::variant<
    std::pair<svs::data::ConstSimpleDataView<float>, std::span<const size_t>>,
    std::filesystem::path>;

using BuildDynamicIndexDispatcher = svs::lib::Dispatcher<
    svs::DynamicVamana,
    const svs::index::vamana::VamanaBuildParameters&,
    DynamicVamanaSource,
    const Storage*,
    svs::DistanceType,
    svs::threads::ThreadPoolHandle,
    size_t>;

const BuildDynamicIndexDispatcher& build_dynamic_vamana_index_dispatcher() {
    static BuildDynamicIndexDispatcher dispatcher = [] {
        BuildDynamicIndexDispatcher d{};
        register_dynamic_vamana_index_specializations(d);
        return d;
    }();
    return dispatcher;
}

svs::DynamicVamana dispatch_dynamic_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> data,
    std::span<const size_t> ids,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    size_t blocksize_bytes
) {
    return build_dynamic_vamana_index_dispatcher().invoke(
        build_params,
        DynamicVamanaSource{std::make_pair(data, ids)},
        storage,
        distance_type,
        std::move(pool),
        blocksize_bytes
    );
}

svs::DynamicVamana dispatch_dynamic_vamana_index_load(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const std::filesystem::path& directory,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    size_t blocksize_bytes
) {
    return build_dynamic_vamana_index_dispatcher().invoke(
        build_params,
        DynamicVamanaSource{directory},
        storage,
        distance_type,
        std::move(pool),
        blocksize_bytes
    );
}
} // namespace svs::c_runtime
