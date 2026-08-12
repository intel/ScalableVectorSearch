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

#include "svs/c/svs_c.h"

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

struct IDFilterInterface {
    virtual ~IDFilterInterface() = default;
    virtual bool is_member(size_t id) const = 0;
    // filter_rate() returns the estimated selectivity of the filter, i.e., the fraction of
    // IDs that are expected to pass the filter. A value of 0.01 indicates that 1% of IDs
    // are expected to pass, while a value of 1.0 indicates that all IDs are expected to
    // pass. If the filter does not provide an estimate, it should return 0.0.
    virtual float filter_rate() const = 0;
    bool operator()(size_t id) const { return is_member(id); }
};

struct IDFilterAdapter : public IDFilterInterface {
    const svs_id_filter_i c_filter;
    float filter_rate_value; // Caching the filter rate value

    IDFilterAdapter(const svs_id_filter_i filter)
        : c_filter(filter)
        , filter_rate_value(0.0f) {
        if (c_filter != nullptr) {
            if (c_filter->ops == nullptr) {
                throw std::invalid_argument("Custom ID filter is not initialized.");
            }
            if (c_filter->ops->is_member == nullptr) {
                throw std::invalid_argument(
                    "Custom ID filter is missing the is_member function."
                );
            }
            if (c_filter->ops->filter_rate != nullptr) {
                filter_rate_value = c_filter->ops->filter_rate(c_filter->self);
                if (filter_rate_value < 0.0f || filter_rate_value > 1.0f) {
                    throw std::invalid_argument(
                        "Filter rate must be between 0.0 and 1.0, inclusive."
                    );
                }
            }
        }
    }

    bool is_member(size_t id) const override {
        // If no filter is provided, consider all IDs as valid
        return c_filter != nullptr ? c_filter->ops->is_member(c_filter->self, id) : true;
    }

    float filter_rate() const override { return filter_rate_value; }
};

template <typename Alloc>
size_t
adjust_blocked_size(size_t num_vectors, size_t element_size, const Alloc& allocator) {
    if constexpr (svs::data::is_blocked_v<Alloc>) {
        // If using blocked allocator, account for block size overhead
        // following the same logic as in SimpleData .ctor for Blocked allocators
        assert(element_size > 0);
        const auto blocksize =
            lib::prevpow2(allocator.parameters().blocksize_bytes.value() / element_size);
        size_t elements_per_block = blocksize.value();
        size_t num_blocks = lib::div_round_up(num_vectors, elements_per_block);
        return num_blocks * blocksize.value() * element_size;
    } else {
        return num_vectors * element_size;
    }
}

} // namespace c_runtime
} // namespace svs
