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

class ThreadPoolBuilder {
    struct CustomThreadPool {
        static svs_threadpool_i validate(svs_threadpool_i impl) {
            if (impl == nullptr) {
                throw std::invalid_argument("Custom threadpool pointer cannot be null.");
            }
            if (impl->ops.size == nullptr || impl->ops.parallel_for == nullptr) {
                throw std::invalid_argument(
                    "Custom threadpool interface has null function pointers."
                );
            }
            return impl;
        }

        CustomThreadPool(svs_threadpool_i impl)
            : impl{validate(impl)} {}

        size_t size() const {
            assert(impl != nullptr);
            return impl->ops.size(impl->self);
        }

        void parallel_for(std::function<void(size_t)> f, size_t n) const {
            assert(impl != nullptr);
            impl->ops.parallel_for(
                impl->self,
                [](void* svs_param, size_t i) {
                    auto& func = *static_cast<std::function<void(size_t)>*>(svs_param);
                    func(i);
                },
                &f,
                n
            );
        }

        svs_threadpool_i impl;
    };

    svs_threadpool_kind kind;
    size_t num_threads;
    svs_threadpool_i user_threadpool;

  public:
    ThreadPoolBuilder()
        : ThreadPoolBuilder(SVS_THREADPOOL_KIND_NATIVE, default_threads_num()) {}

    ThreadPoolBuilder(svs_threadpool_kind kind, size_t num_threads)
        : kind(kind)
        , num_threads(num_threads) {
        if (kind == SVS_THREADPOOL_KIND_CUSTOM) {
            throw std::invalid_argument(
                "SVS_THREADPOOL_KIND_CUSTOM cannot be built automatically."
            );
        }
    }

    ThreadPoolBuilder(svs_threadpool_i pool)
        : kind(SVS_THREADPOOL_KIND_CUSTOM)
        , num_threads(0)
        , user_threadpool(CustomThreadPool::validate(pool)) {}

    static size_t default_threads_num() {
        return std::max(size_t{1}, size_t{std::thread::hardware_concurrency()});
    }

    svs::threads::ThreadPoolHandle build() const {
        using namespace svs::threads;
        switch (kind) {
            case SVS_THREADPOOL_KIND_NATIVE:
                return ThreadPoolHandle(NativeThreadPool(num_threads));
            case SVS_THREADPOOL_KIND_OMP:
                return ThreadPoolHandle(OMPThreadPool(num_threads));
            case SVS_THREADPOOL_KIND_SINGLE_THREAD:
                return ThreadPoolHandle(SequentialThreadPool());
            case SVS_THREADPOOL_KIND_CUSTOM:
                assert(user_threadpool != nullptr);
                return ThreadPoolHandle(CustomThreadPool{this->user_threadpool});
            default:
                throw std::invalid_argument("Unknown svs_threadpool_kind value.");
        }
    }
};
} // namespace svs::c_runtime
