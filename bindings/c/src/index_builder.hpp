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

namespace svs::c_runtime {

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
    svs::data::ConstSimpleDataView<float>,
    const Storage*,
    svs::DistanceType,
    svs::threads::ThreadPoolHandle>;

BuildIndexDispatcher build_vamana_index_dispatcher() {
    auto dispatcher = BuildIndexDispatcher{};
    register_build_vamana_index_methods(dispatcher);
    return dispatcher;
}

svs::Vamana dispatch_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> src_data,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    return build_vamana_index_dispatcher().invoke(
        build_params, std::move(src_data), storage, distance_type, std::move(pool)
    );
}

using LoadIndexDispatcher = svs::lib::Dispatcher<
    svs::Vamana,
    const std::filesystem::path&,
    const Storage*,
    svs::DistanceType,
    svs::threads::ThreadPoolHandle>;

LoadIndexDispatcher load_vamana_index_dispatcher() {
    auto dispatcher = LoadIndexDispatcher{};
    register_load_vamana_index_methods(dispatcher);
    return dispatcher;
}

svs::Vamana dispatch_vamana_index_load(
    const std::filesystem::path& directory,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    return load_vamana_index_dispatcher().invoke(
        directory, storage, distance_type, std::move(pool)
    );
}

struct IndexBuilder {
    svs_distance_metric_t distance_metric;
    size_t dimension;
    std::shared_ptr<Algorithm> algorithm;
    std::shared_ptr<Storage> storage;
    ThreadPoolBuilder pool_builder;

    IndexBuilder(
        svs_distance_metric_t distance_metric,
        size_t dimension,
        std::shared_ptr<Algorithm> algorithm
    )
        : distance_metric(distance_metric)
        , dimension(dimension)
        , algorithm(std::move(algorithm))
        , storage(std::make_shared<StorageSimple>(SVS_DATA_TYPE_FLOAT32))
        , pool_builder{} {}

    ~IndexBuilder() {}

    void set_storage(std::shared_ptr<Storage> storage) {
        this->storage = std::move(storage);
    }

    void set_threadpool_builder(ThreadPoolBuilder threadpool_builder) {
        std::swap(this->pool_builder, threadpool_builder);
    }

    std::shared_ptr<Index> build(const svs::data::ConstSimpleDataView<float>& data) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index = std::make_shared<IndexVamana>(dispatch_vamana_index_build(
                vamana_algorithm->build_parameters(),
                data,
                storage.get(),
                to_distance_type(distance_metric),
                pool_builder.build()
            ));

            return index;
        }
        return nullptr;
    }

    std::shared_ptr<Index> load(const std::filesystem::path& directory) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto index = std::make_shared<IndexVamana>(dispatch_vamana_index_load(
                directory,
                storage.get(),
                to_distance_type(distance_metric),
                pool_builder.build()
            ));

            return index;
        }
        return nullptr;
    }
};
} // namespace svs::c_runtime
