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

#include "allocator.hpp"
#include "storage.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/data/simple.h>
#include <svs/core/distance.h>
#include <svs/lib/datatype.h>
#include <svs/lib/dispatcher.h>
#include <svs/lib/float16.h>
#include <svs/lib/memory.h>
#include <svs/lib/threads/threadpool.h>
#include <svs/lib/type_traits.h>

#include <filesystem>
#include <stdexcept>

namespace svs {

template <Arithmetic T, typename Allocator = svs::lib::Allocator<T>>
class SimpleDataBuilder {
  public:
    SimpleDataBuilder() {}

    using data_type = svs::data::SimpleData<T, svs::Dynamic, Allocator>;
    using allocator_type = Allocator;

    template <Arithmetic U>
    data_type build(
        svs::data::ConstSimpleDataView<U> view,
        svs::threads::ThreadPoolHandle& SVS_UNUSED(pool),
        const allocator_type& allocator = {}
    ) {
        auto data = data_type(view.size(), view.dimensions(), allocator);
        svs::data::copy(view, data);
        return data;
    }

    data_type
    load(const std::filesystem::path& path, const allocator_type& allocator = {}) {
        return svs::lib::load_from_disk<data_type>(path, allocator);
    }
};

template <Arithmetic T, typename Allocator>
struct lib::DispatchConverter<const c_runtime::Storage*, SimpleDataBuilder<T, Allocator>> {
    using From = const svs::c_runtime::Storage*;
    using To = SimpleDataBuilder<T, Allocator>;

    static int64_t match(From from) {
        if constexpr (svs::is_arithmetic_v<T>) {
            if (from->kind == SVS_STORAGE_KIND_SIMPLE) {
                auto simple = static_cast<const c_runtime::StorageSimple*>(from);
                if (simple->data_type == svs::datatype_v<T>) {
                    return svs::lib::perfect_match;
                }
            }
        }
        return svs::lib::invalid_match;
    }

    static To convert(From SVS_UNUSED(from)) { return To{}; }
};

template <bool UseBlocked, typename F> void for_simple_specializations(F&& f) {
    using float_alloc = svs::c_runtime::MaybeBlockedAlloc<float, UseBlocked>;
    using float16_alloc = svs::c_runtime::MaybeBlockedAlloc<svs::Float16, UseBlocked>;
#define X(T, A, D) f.template operator()<SimpleDataBuilder<T, A>, D>();
#define XX(T, A) X(T, A, DistanceL2) X(T, A, DistanceIP) X(T, A, DistanceCosineSimilarity)
    XX(float, float_alloc)
    XX(svs::Float16, float16_alloc)
#undef XX
#undef X
}

} // namespace svs
