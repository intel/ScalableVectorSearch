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
/// @brief Rebind an allocator to a different type.
///
template <typename Allocator, typename T>
using RebindAlloc = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;

///
/// @brief Read and parse the data type configuration from a saved IVF index.
///
/// @param config_path Path to the configuration directory
/// @return The parsed DataTypeConfig
///
inline svs::index::ivf::DataTypeConfig
read_data_type_config(const std::string& config_path) {
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
        throw ANNEXCEPTION("Config file missing 'data_type_config' section.");
    }

    // Convert to table and create ContextFreeLoadTable
    auto* data_type_table = data_type_node.as_table();
    if (!data_type_table) {
        throw ANNEXCEPTION("data_type_config is not a table");
    }
    auto ctx_free = svs::lib::ContextFreeLoadTable(*data_type_table);
    return svs::index::ivf::DataTypeConfig::load(ctx_free);
}

///
/// @brief Generic loader function template for IVF index assembly.
///
/// This template reduces boilerplate by providing a generic loader that can be
/// instantiated with different centroid and data types.
///
/// @tparam IndexType The IVF index type (svs::IVF or svs::DynamicIVF)
/// @tparam CentroidType The centroid type (Float16 or BFloat16)
/// @tparam DataType The data type for the clusters
///
template <typename IndexType, typename CentroidType, typename DataType>
IndexType load_typed(
    const std::string& config_path,
    const std::string& data_path,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads
) {
    return IndexType::template assemble<float, CentroidType, DataType>(
        config_path, data_path, distance_type, num_threads, intra_query_threads
    );
}

///
/// @brief Loader for uncompressed IVF data with type dispatch.
///
/// Dispatches to the appropriate loader based on element type and centroid type.
///
/// @tparam IndexType The type of index to return
/// @tparam DataContainer The data container template (SimpleData or BlockedData)
/// @tparam Allocator The allocator type for the data (will be rebound to element type)
///
template <
    typename IndexType,
    template <typename, size_t, typename>
    class DataContainer,
    typename Allocator>
IndexType load_uncompressed_with_dispatch(
    const std::string& config_path,
    const std::string& data_path,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads,
    const svs::index::ivf::DataTypeConfig& data_config
) {
    bool is_f16_centroids = (data_config.centroid_type == svs::DataType::float16);
    bool is_f16_data = (data_config.element_type == svs::DataType::float16);

    // Dispatch based on data type and centroid type combinations
    // Rebind the allocator to the appropriate element type
    if (is_f16_data) {
        using ReboundAlloc = RebindAlloc<Allocator, svs::Float16>;
        using DataType = DataContainer<svs::Float16, svs::Dynamic, ReboundAlloc>;
        if (is_f16_centroids) {
            return load_typed<IndexType, svs::Float16, DataType>(
                config_path, data_path, distance_type, num_threads, intra_query_threads
            );
        } else {
            return load_typed<IndexType, svs::BFloat16, DataType>(
                config_path, data_path, distance_type, num_threads, intra_query_threads
            );
        }
    } else {
        using ReboundAlloc = RebindAlloc<Allocator, float>;
        using DataType = DataContainer<float, svs::Dynamic, ReboundAlloc>;
        if (is_f16_centroids) {
            return load_typed<IndexType, svs::Float16, DataType>(
                config_path, data_path, distance_type, num_threads, intra_query_threads
            );
        } else {
            return load_typed<IndexType, svs::BFloat16, DataType>(
                config_path, data_path, distance_type, num_threads, intra_query_threads
            );
        }
    }
}

///
/// @brief Generic IVF index loader with type dispatch based on saved configuration.
///
/// @tparam IndexType The type of index to return (svs::IVF or svs::DynamicIVF)
/// @tparam DataContainer The data container template (SimpleData or BlockedData)
/// @tparam Allocator The allocator type for uncompressed data
///
template <
    typename IndexType,
    template <typename, size_t, typename>
    class DataContainer,
    typename Allocator>
IndexType load_index_auto(
    const std::string& config_path,
    const std::string& data_path,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads
) {
    auto data_config = read_data_type_config(config_path);

    // Dispatch based on schema - only uncompressed supported in public repo
    if (data_config.schema == "uncompressed_data") {
        return load_uncompressed_with_dispatch<IndexType, DataContainer, Allocator>(
            config_path,
            data_path,
            distance_type,
            num_threads,
            intra_query_threads,
            data_config
        );
    }

    throw ANNEXCEPTION(
        "Unknown or unsupported data type schema: ",
        data_config.schema,
        ". Only uncompressed data is supported in the public repository. "
    );
}

} // namespace svs::python::ivf_loader
