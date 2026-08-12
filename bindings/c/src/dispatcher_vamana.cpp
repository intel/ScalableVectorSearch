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
#include "index.hpp"
#include "storage.hpp"
#include "threadpool.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/vamana.h>

#include <filesystem>
#include <memory>
#include <variant>

namespace svs::c_runtime {

template <typename DataBuilder, typename Distance>
svs::Vamana build_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> src_data,
    DataBuilder builder,
    Distance distance,
    svs::threads::ThreadPoolHandle pool
) {
    auto data = builder.build(std::move(src_data), pool);
    return svs::Vamana::build<float>(
        build_params, std::move(data), distance, std::move(pool)
    );
}

template <typename DataLoader, typename Distance>
svs::Vamana load_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& SVS_UNUSED(build_params),
    const std::filesystem::path& directory,
    DataLoader loader,
    Distance distance,
    svs::threads::ThreadPoolHandle pool
) {
    auto data = loader.load(directory / "data");
    return svs::Vamana::assemble<float>(
        directory / "config",
        svs::GraphLoader{directory / "graph"},
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
    svs::threads::ThreadPoolHandle>;

const BuildIndexDispatcher& build_vamana_index_dispatcher() {
    static BuildIndexDispatcher dispatcher = [] {
        BuildIndexDispatcher d{};
        register_vamana_index_specializations(d);
        return d;
    }();
    return dispatcher;
}

svs::Vamana dispatch_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> data,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    return build_vamana_index_dispatcher().invoke(
        build_params, VamanaSource{std::move(data)}, storage, distance_type, std::move(pool)
    );
}

svs::Vamana dispatch_vamana_index_load(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const std::filesystem::path& directory,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    return build_vamana_index_dispatcher().invoke(
        build_params, VamanaSource{directory}, storage, distance_type, std::move(pool)
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
