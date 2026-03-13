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
#include "threadpool.hpp"

#include <svs/concepts/data.h>
#include <svs/core/data/simple.h>
#include <svs/core/distance.h>
#include <svs/index/vamana/build_params.h>
#include <svs/lib/threads/threadpool.h>
#include <svs/orchestrators/vamana.h>

#include <filesystem>

namespace svs::c_runtime {
svs::Vamana dispatch_vamana_index_build(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    svs::data::ConstSimpleDataView<float> data,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
);

svs::Vamana dispatch_vamana_index_load(
    const svs::index::vamana::VamanaBuildParameters& build_params,
    const std::filesystem::path& directory,
    const Storage* storage,
    svs::DistanceType distance_type,
    svs::threads::ThreadPoolHandle pool
);

} // namespace svs::c_runtime
