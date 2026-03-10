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
#include "dispatcher_dynamic_vamana.hpp"
#include "dispatcher_vamana.hpp"
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
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index = std::make_shared<IndexVamana>(dispatch_vamana_index_load(
                vamana_algorithm->build_parameters(),
                directory,
                storage.get(),
                to_distance_type(distance_metric),
                pool_builder.build()
            ));

            return index;
        }
        return nullptr;
    }

    std::shared_ptr<DynamicIndex> build_dynamic(
        const svs::data::ConstSimpleDataView<float>& data,
        std::span<const size_t> ids,
        size_t blocksize_bytes
    ) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index =
                std::make_shared<DynamicIndexVamana>(dispatch_dynamic_vamana_index_build(
                    vamana_algorithm->build_parameters(),
                    data,
                    ids,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build(),
                    blocksize_bytes
                ));

            return index;
        }
        return nullptr;
    }

    std::shared_ptr<DynamicIndex>
    load_dynamic(const std::filesystem::path& directory, size_t blocksize_bytes) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            auto index =
                std::make_shared<DynamicIndexVamana>(dispatch_dynamic_vamana_index_load(
                    vamana_algorithm->build_parameters(),
                    directory,
                    storage.get(),
                    to_distance_type(distance_metric),
                    pool_builder.build(),
                    blocksize_bytes
                ));

            return index;
        }
        return nullptr;
    }
};
} // namespace svs::c_runtime
