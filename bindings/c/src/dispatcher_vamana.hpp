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

#include "svs/c_api/svs_c.h"

#include "algorithm.hpp"
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

using VamanaSource =
    std::variant<svs::data::ConstSimpleDataView<float>, std::filesystem::path>;

template <typename DataBuilder>
svs::Vamana build_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> src_data,
    DataBuilder builder,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    auto data = builder.build(std::move(src_data), pool);
    return svs::Vamana::build<float>(
        build_params, std::move(data), distance_type, std::move(pool)
    );
}

template <typename DataLoader>
svs::Vamana load_vamana_index(
    const svs::index::vamana::VamanaBuildParameters& SVS_UNUSED(build_params),
    const std::filesystem::path& directory,
    DataLoader loader,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    auto data = loader.load(directory / "data");
    return svs::Vamana::assemble<float>(
        directory / "config",
        svs::GraphLoader{directory / "graph"},
        std::move(data),
        distance_type,
        std::move(pool)
    );
}

template <typename Dispatcher>
void register_build_vamana_index_methods(Dispatcher& dispatcher) {
    dispatcher.register_target(&build_vamana_index<SimpleDataBuilder<float>>);
    dispatcher.register_target(&build_vamana_index<SimpleDataBuilder<svs::Float16>>);

    dispatcher.register_target(&build_vamana_index<LeanVecDataBuilder<4, 4>>);
    dispatcher.register_target(&build_vamana_index<LeanVecDataBuilder<4, 8>>);
    dispatcher.register_target(&build_vamana_index<LeanVecDataBuilder<8, 8>>);

    dispatcher.register_target(&build_vamana_index<LVQDataBuilder<4, 0>>);
    dispatcher.register_target(&build_vamana_index<LVQDataBuilder<8, 0>>);
    dispatcher.register_target(&build_vamana_index<LVQDataBuilder<4, 4>>);
    dispatcher.register_target(&build_vamana_index<LVQDataBuilder<4, 8>>);

    dispatcher.register_target(&build_vamana_index<SQDataBuilder<uint8_t>>);
    dispatcher.register_target(&build_vamana_index<SQDataBuilder<int8_t>>);
}

template <typename Dispatcher>
void register_load_vamana_index_methods(Dispatcher& dispatcher) {
    dispatcher.register_target(&load_vamana_index<SimpleDataBuilder<float>>);
    dispatcher.register_target(&load_vamana_index<SimpleDataBuilder<svs::Float16>>);

    dispatcher.register_target(&load_vamana_index<LeanVecDataBuilder<4, 4>>);
    dispatcher.register_target(&load_vamana_index<LeanVecDataBuilder<4, 8>>);
    dispatcher.register_target(&load_vamana_index<LeanVecDataBuilder<8, 8>>);

    dispatcher.register_target(&load_vamana_index<LVQDataBuilder<4, 0>>);
    dispatcher.register_target(&load_vamana_index<LVQDataBuilder<8, 0>>);
    dispatcher.register_target(&load_vamana_index<LVQDataBuilder<4, 4>>);
    dispatcher.register_target(&load_vamana_index<LVQDataBuilder<4, 8>>);

    dispatcher.register_target(&load_vamana_index<SQDataBuilder<uint8_t>>);
    dispatcher.register_target(&load_vamana_index<SQDataBuilder<int8_t>>);
}

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
        register_build_vamana_index_methods(d);
        register_load_vamana_index_methods(d);
        return d;
    }();
    return dispatcher;
}

svs::Vamana dispatch_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    VamanaSource src_data,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    return build_vamana_index_dispatcher().invoke(
        build_params, std::move(src_data), storage, distance_type, std::move(pool)
    );
}
} // namespace svs::c_runtime
