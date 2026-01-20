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

#include "types_support.hpp"

#include <svs/lib/threads.h>

#include <algorithm>
#include <thread>

namespace svs::c_runtime {

struct ThreadPoolBuilder {
    svs_thread_pool_kind kind;
    size_t num_threads;
    ThreadPoolBuilder(svs_thread_pool_kind kind, size_t num_threads)
        : kind(kind)
        , num_threads(num_threads) {}

    ThreadPoolBuilder()
        : ThreadPoolBuilder(SVS_THREAD_POOL_KIND_NATIVE, default_threads_num()) {}

    static size_t default_threads_num() {
        return std::max(size_t{1}, size_t{std::thread::hardware_concurrency()});
    }

    svs::threads::ThreadPoolHandle build() const {
        using namespace svs::threads;
        switch (kind) {
            case SVS_THREAD_POOL_KIND_NATIVE:
                return ThreadPoolHandle(NativeThreadPool(num_threads));
            case SVS_THREAD_POOL_KIND_OMP:
                return ThreadPoolHandle(OMPThreadPool(num_threads));
            case SVS_THREAD_POOL_KIND_SINGLE_THREAD:
                return ThreadPoolHandle(SequentialThreadPool());
            case SVS_THREAD_POOL_KIND_MANUAL:
                throw std::invalid_argument(
                    "SVS_THREAD_POOL_KIND_MANUAL cannot be built automatically."
                );
            default:
                throw std::invalid_argument("Unknown svs_thread_pool_kind value.");
        }
    }
};
} // namespace svs::c_runtime
