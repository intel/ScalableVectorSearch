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

// svs
#include "svs/core/distance.h"
#include "svs/index/ivf/data_traits.h"
#include "svs/lib/datatype.h"
#include "svs/lib/exception.h"
#include "svs/lib/float16.h"
#include "svs/lib/saveload.h"

// toml
#include <toml++/toml.h>

// stl
#include <filesystem>
#include <string>

namespace svs::python::ivf_loader {

///
/// @brief Generic IVF index loader with type dispatch based on saved configuration.
///
/// This template-based loader eliminates code duplication between static IVF and
/// DynamicIVF by dispatching to the correct typed loader based on the data_type_config
/// saved in the index configuration file.
///
/// @tparam IndexType The type of index to return (svs::IVF or svs::DynamicIVF)
/// @tparam LoaderF32BF16 Callable type for loading float32 data with bfloat16 centroids
/// @tparam LoaderF32F16 Callable type for loading float32 data with float16 centroids
/// @tparam LoaderF16BF16 Callable type for loading float16 data with bfloat16 centroids
/// @tparam LoaderF16F16 Callable type for loading float16 data with float16 centroids
///
template <
    typename IndexType,
    typename LoaderF32BF16,
    typename LoaderF32F16,
    typename LoaderF16BF16,
    typename LoaderF16F16>
IndexType load_index_with_dispatch(
    const std::string& config_path,
    const std::string& data_path,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads,
    LoaderF32BF16&& loader_f32_bf16,
    LoaderF32F16&& loader_f32_f16,
    LoaderF16BF16&& loader_f16_bf16,
    LoaderF16F16&& loader_f16_f16
) {
    // Read the config file to get data_type_config
    auto config_file = std::filesystem::path(config_path) / svs::lib::config_file_name;
    auto table = toml::parse_file(config_file.string());

    // The data_type_config is nested inside "object" section
    auto object_node = table["object"];
    if (!object_node) {
        throw ANNEXCEPTION("Config file missing 'object' section.");
    }
    auto* object_table = object_node.as_table();
    if (!object_table) {
        throw ANNEXCEPTION("'object' section is not a table.");
    }

    // Get the data_type_config section from object
    auto data_type_node = (*object_table)["data_type_config"];
    if (!data_type_node) {
        // Backward compatibility: no data_type_config means old format, default to
        // float32/bfloat16
        return loader_f32_bf16(
            config_path, data_path, distance_type, num_threads, intra_query_threads
        );
    }

    // Convert to table and create ContextFreeLoadTable
    auto* data_type_table = data_type_node.as_table();
    if (!data_type_table) {
        throw ANNEXCEPTION("data_type_config is not a table");
    }
    auto ctx_free = svs::lib::ContextFreeLoadTable(*data_type_table);
    auto data_config = svs::index::ivf::DataTypeConfig::load(ctx_free);

    // Dispatch based on schema
    if (data_config.schema == "uncompressed_data") {
        // Dispatch based on element type and centroid type
        bool is_f16_centroids = (data_config.centroid_type == svs::DataType::float16);
        bool is_f16_data = (data_config.element_type == svs::DataType::float16);

        if (is_f16_data) {
            if (is_f16_centroids) {
                return loader_f16_f16(
                    config_path, data_path, distance_type, num_threads, intra_query_threads
                );
            } else {
                return loader_f16_bf16(
                    config_path, data_path, distance_type, num_threads, intra_query_threads
                );
            }
        } else {
            if (is_f16_centroids) {
                return loader_f32_f16(
                    config_path, data_path, distance_type, num_threads, intra_query_threads
                );
            } else {
                return loader_f32_bf16(
                    config_path, data_path, distance_type, num_threads, intra_query_threads
                );
            }
        }
    }

    throw ANNEXCEPTION(
        "Unknown or unsupported data type schema: ",
        data_config.schema,
        ". Only uncompressed data is supported in the public repository. "
        "For LVQ/LeanVec support, use the private repository."
    );
}

///
/// @brief Simplified loader when all typed loaders follow the same signature pattern.
///
/// This overload accepts a single template loader that will be invoked with appropriate
/// data type and centroid type template arguments based on the saved configuration.
///
/// Usage example:
/// @code
/// auto index = load_index_auto<svs::IVF>(
///     config_path, data_path, distance_type, num_threads, intra_query_threads,
///     []<typename DataType, typename CentroidType>(
///         const std::string& config_path,
///         const std::string& data_path,
///         svs::DistanceType distance_type,
///         size_t num_threads,
///         size_t intra_query_threads
///     ) {
///         return svs::IVF::assemble<float, CentroidType, DataType>(
///             config_path, data_path, distance_type, num_threads, intra_query_threads
///         );
///     }
/// );
/// @endcode
///
template <typename IndexType, typename GenericLoader>
IndexType load_index_auto(
    const std::string& config_path,
    const std::string& data_path,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads,
    GenericLoader&& loader
) {
    return load_index_with_dispatch<IndexType>(
        config_path,
        data_path,
        distance_type,
        num_threads,
        intra_query_threads,
        // float32 data, bfloat16 centroids
        [&loader](
            const std::string& cfg,
            const std::string& data,
            svs::DistanceType dist,
            size_t threads,
            size_t intra_threads
        ) {
            return loader.template operator(
            )<float, svs::BFloat16>(cfg, data, dist, threads, intra_threads);
        },
        // float32 data, float16 centroids
        [&loader](
            const std::string& cfg,
            const std::string& data,
            svs::DistanceType dist,
            size_t threads,
            size_t intra_threads
        ) {
            return loader.template operator(
            )<float, svs::Float16>(cfg, data, dist, threads, intra_threads);
        },
        // float16 data, bfloat16 centroids
        [&loader](
            const std::string& cfg,
            const std::string& data,
            svs::DistanceType dist,
            size_t threads,
            size_t intra_threads
        ) {
            return loader.template operator(
            )<svs::Float16, svs::BFloat16>(cfg, data, dist, threads, intra_threads);
        },
        // float16 data, float16 centroids
        [&loader](
            const std::string& cfg,
            const std::string& data,
            svs::DistanceType dist,
            size_t threads,
            size_t intra_threads
        ) {
            return loader.template operator(
            )<svs::Float16, svs::Float16>(cfg, data, dist, threads, intra_threads);
        }
    );
}

} // namespace svs::python::ivf_loader
