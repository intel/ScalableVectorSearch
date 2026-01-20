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
#include "thread_pool.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/vamana.h>

namespace svs::c_runtime {

template <typename T>
svs::Vamana build_vamana_index_uncompressed(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> src_data,
    SimpleDataBuilder<T> builder,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    auto data = builder.build(std::move(src_data), pool);
    return svs::Vamana::build<float>(
        build_params, std::move(data), distance_type, std::move(pool)
    );
}

template <size_t I1, size_t I2>
svs::Vamana build_vamana_index_leanvec(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> src_data,
    LeanVecDataBuilder<I1, I2> builder,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    auto data = builder.build(std::move(src_data), pool);
    return svs::Vamana::build<float>(
        build_params, std::move(data), distance_type, std::move(pool)
    );
}

template <size_t PrimaryBits, size_t ResidualBits>
svs::Vamana build_vamana_index_lvq(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> src_data,
    LVQDataBuilder<PrimaryBits, ResidualBits> builder,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    auto data = builder.build(std::move(src_data), pool);
    return svs::Vamana::build<float>(
        build_params, std::move(data), distance_type, std::move(pool)
    );
}

template <typename T>
svs::Vamana build_vamana_index_sq(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> src_data,
    SQDataBuilder<T> builder,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
) {
    auto data = builder.build(std::move(src_data), pool);
    return svs::Vamana::build<float>(
        build_params, std::move(data), distance_type, std::move(pool)
    );
}

template <typename Dispatcher>
void register_build_vamana_index_methods(Dispatcher& dispatcher) {
    dispatcher.register_target(&build_vamana_index_uncompressed<float>);
    dispatcher.register_target(&build_vamana_index_uncompressed<svs::Float16>);

    dispatcher.register_target(&build_vamana_index_leanvec<4, 4>);
    dispatcher.register_target(&build_vamana_index_leanvec<4, 8>);
    dispatcher.register_target(&build_vamana_index_leanvec<8, 8>);

    dispatcher.register_target(&build_vamana_index_lvq<4, 0>);
    dispatcher.register_target(&build_vamana_index_lvq<8, 0>);
    dispatcher.register_target(&build_vamana_index_lvq<4, 4>);
    dispatcher.register_target(&build_vamana_index_lvq<4, 8>);

    dispatcher.register_target(&build_vamana_index_sq<uint8_t>);
    dispatcher.register_target(&build_vamana_index_sq<int8_t>);
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

svs::Vamana build_vamana_index(
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

    void set_thread_pool(ThreadPoolBuilder thread_pool_builder) {
        std::swap(this->pool_builder, thread_pool_builder);
    }

    std::shared_ptr<Index> build(const svs::data::ConstSimpleDataView<float>& data) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            svs::index::vamana::VamanaBuildParameters build_params =
                vamana_algorithm->get_build_parameters();

            auto index = std::make_shared<IndexVamana>(build_vamana_index(
                vamana_algorithm->get_build_parameters(),
                data,
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
