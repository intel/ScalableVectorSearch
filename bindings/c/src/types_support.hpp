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

#include <svs/core/distance.h>
#include <svs/lib/datatype.h>

namespace svs {
namespace c_runtime {

inline svs::DistanceType to_distance_type(svs_distance_metric_t distance_metric) {
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

inline svs::DataType to_data_type(svs_data_type_t data_type) {
    switch (data_type) {
        case SVS_DATA_TYPE_FLOAT32:
            return svs::DataType::float32;
        case SVS_DATA_TYPE_FLOAT16:
            return svs::DataType::float16;
        case SVS_DATA_TYPE_INT8:
            return svs::DataType::int8;
        case SVS_DATA_TYPE_UINT8:
            return svs::DataType::uint8;
        case SVS_DATA_TYPE_INT4:
            return svs::DataType::int8; // No direct mapping, using int8 as placeholder
        case SVS_DATA_TYPE_UINT4:
            return svs::DataType::uint8; // No direct mapping, using uint8 as placeholder
        default:
            return svs::DataType::undef;
    }
}

} // namespace c_runtime
} // namespace svs
