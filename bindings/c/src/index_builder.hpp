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

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/orchestrators/vamana.h>

namespace svs::c_runtime {

struct IndexBuilder {
    svs_distance_metric_t distance_metric;
    size_t dimension;
    std::shared_ptr<Algorithm> algorithm;
    std::shared_ptr<Storage> storage;
    IndexBuilder(
        svs_distance_metric_t distance_metric,
        size_t dimension,
        std::shared_ptr<Algorithm> algorithm
    )
        : distance_metric(distance_metric)
        , dimension(dimension)
        , algorithm(std::move(algorithm))
        , storage(std::make_shared<StorageSimple>(SVS_DATA_TYPE_FLOAT32)) {}

    ~IndexBuilder() {}

    void set_storage(std::shared_ptr<Storage> storage) {
        this->storage = std::move(storage);
    }

    svs::DistanceType get_distance_type() const {
        switch (distance_metric) {
            case SVS_DISTANCE_METRIC_EUCLIDEAN:
                return svs::DistanceType::L2;
            case SVS_DISTANCE_METRIC_DOT_PRODUCT:
                return svs::DistanceType::MIP;
            case SVS_DISTANCE_METRIC_COSINE:
                return svs::DistanceType::Cosine;
            default:
                return svs::DistanceType::L2; // Default fallback
        }
    }

    std::shared_ptr<Index> build(const svs::data::ConstSimpleDataView<float>& data) {
        if (algorithm->type == SVS_ALGORITHM_TYPE_VAMANA &&
            storage->kind == SVS_STORAGE_KIND_SIMPLE) {
            auto vamana_algorithm = std::static_pointer_cast<AlgorithmVamana>(algorithm);

            svs::index::vamana::VamanaBuildParameters build_params =
                vamana_algorithm->get_build_parameters();

            auto storage = svs::data::SimpleData<float>(data.size(), data.dimensions());

            svs::data::copy(data, storage);

            auto index = std::make_shared<IndexVamana>(svs::Vamana::build<float>(
                vamana_algorithm->get_build_parameters(),
                std::move(storage),
                get_distance_type()
            ));

            return index;
        }
        return nullptr;
    }
};
} // namespace svs::c_runtime
