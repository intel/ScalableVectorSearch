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

#include "svs/core/data/simple.h"
#include "svs/lib/saveload.h"

#include <string>
#include <string_view>

namespace svs::index::ivf {

/// @brief Data type configuration for IVF save/load
///
/// This struct holds information about the data type stored in an IVF index,
/// allowing automatic reconstruction of the appropriate loader during load.
///
/// The base implementation supports uncompressed data (fp32, fp16, bf16).
/// Extensions (e.g., LVQ, LeanVec) can be added by including additional
/// specialization headers that specialize DataTypeTraits for their types.
struct DataTypeConfig {
    // Schema identifier (e.g., "uncompressed_data", "one_level_lvq_dataset",
    // "leanvec_dataset")
    std::string schema;

    // For uncompressed data: element type
    DataType element_type = DataType::undef;

    // Centroid type (bfloat16 or float16) - saved separately to match centroid storage
    DataType centroid_type = DataType::bfloat16;

    // For LVQ: compression parameters
    size_t primary_bits = 0;
    size_t residual_bits = 0;
    std::string strategy; // "sequential" or "turbo"

    // For LeanVec: dimensionality and encoding kinds
    std::string primary_kind;   // "float32", "float16", "lvq4", "lvq8"
    std::string secondary_kind; // "float32", "float16", "lvq4", "lvq8"
    size_t leanvec_dims = 0;

    // Serialization
    static constexpr std::string_view serialization_schema = "ivf_data_type_config";
    static constexpr lib::Version save_version{0, 0, 0};

    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"schema", lib::save(schema)},
             {"element_type", lib::save(element_type)},
             {"centroid_type", lib::save(centroid_type)},
             {"primary_bits", lib::save(primary_bits)},
             {"residual_bits", lib::save(residual_bits)},
             {"strategy", lib::save(strategy)},
             {"primary_kind", lib::save(primary_kind)},
             {"secondary_kind", lib::save(secondary_kind)},
             {"leanvec_dims", lib::save(leanvec_dims)}}
        );
    }

    static DataTypeConfig load(const lib::ContextFreeLoadTable& table) {
        DataTypeConfig config;
        config.schema = lib::load_at<std::string>(table, "schema");
        config.element_type = lib::load_at<DataType>(table, "element_type");
        // centroid_type may not exist in older configs - default to bfloat16
        auto centroid_node = table.try_at("centroid_type");
        if (centroid_node.has_value()) {
            config.centroid_type = lib::load<DataType>(*centroid_node);
        } else {
            config.centroid_type = DataType::bfloat16;
        }
        config.primary_bits = lib::load_at<size_t>(table, "primary_bits");
        config.residual_bits = lib::load_at<size_t>(table, "residual_bits");
        config.strategy = lib::load_at<std::string>(table, "strategy");
        config.primary_kind = lib::load_at<std::string>(table, "primary_kind");
        config.secondary_kind = lib::load_at<std::string>(table, "secondary_kind");
        config.leanvec_dims = lib::load_at<size_t>(table, "leanvec_dims");
        return config;
    }
};

/// @brief Trait to extract DataTypeConfig from a data type
///
/// Default implementation for uncompressed SimpleData.
/// Specializations for LVQ/LeanVec are provided in svs/extensions/ivf/lvq.h
/// and svs/extensions/ivf/leanvec.h respectively.
template <typename Data> struct DataTypeTraits {
    static DataTypeConfig get_config() {
        DataTypeConfig config;
        config.schema = "uncompressed_data";
        config.element_type = datatype_v<typename Data::element_type>;
        return config;
    }
};

} // namespace svs::index::ivf
