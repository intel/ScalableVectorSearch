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

#include "storage.hpp"

#include <svs/core/data/simple.h>
#include <svs/core/distance.h>
#include <svs/core/logging.h>
#include <svs/index/vamana/build_params.h>
#include <svs/lib/threads/threadpool.h>
#include <svs/orchestrators/dynamic_vamana.h>

#include <filesystem>
#include <istream>
#include <memory>
#include <span>
#include <utility>
#include <variant>

namespace svs::c_runtime {

svs::DynamicVamana dispatch_dynamic_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> data,
    std::span<const size_t> ids,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params,
    svs::logging::logger_ptr logger
);

svs::DynamicVamana dispatch_dynamic_vamana_index_load(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const std::filesystem::path& directory,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params,
    svs::logging::logger_ptr logger
);

svs::DynamicVamana dispatch_dynamic_vamana_index_load_stream(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    std::unique_ptr<std::istream> stream,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params
);

svs::DynamicVamana dispatch_dynamic_vamana_index_copy(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const svs::DynamicVamana& src_index,
    const Storage* src_storage,
    const Storage* dst_storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool,
    const AllocatorBuilder& allocator_builder,
    const svs::data::BlockingParameters& block_params,
    svs::logging::logger_ptr logger
);

svs::index::vamana::MemoryBreakdown dispatch_dynamic_vamana_memory_estimate(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    size_t num_vectors,
    size_t dimension,
    const Storage* storage,
    svs::DistanceType distance_type,
    const svs::data::BlockingParameters& block_params
);

} // namespace svs::c_runtime
