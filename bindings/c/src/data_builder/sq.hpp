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
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/datatype.h>
#include <svs/lib/dispatcher.h>
#include <svs/lib/memory.h>
#include <svs/lib/threads/threadpool.h>
#include <svs/lib/type_traits.h>
#include <svs/quantization/scalar/scalar.h>

#include <filesystem>
#include <stdexcept>

namespace svs {

template <Arithmetic T, typename Allocator = svs::lib::Allocator<T>> class SQDataBuilder {
  public:
    SQDataBuilder() {}

    using data_type = svs::quantization::scalar::SQDataset<T, svs::Dynamic, Allocator>;
    using allocator_type = Allocator;

    template <Arithmetic U>
    data_type build(
        svs::data::ConstSimpleDataView<U> view,
        svs::threads::ThreadPoolHandle& pool,
        const allocator_type& allocator = {}
    ) {
        return data_type::compress(view, pool, allocator);
    }

    data_type
    load(const std::filesystem::path& path, const allocator_type& allocator = {}) {
        return svs::lib::load_from_disk<data_type>(path, allocator);
    }
};

template <Arithmetic T, typename Allocator>
struct lib::DispatchConverter<const c_runtime::Storage*, SQDataBuilder<T, Allocator>> {
    using From = const svs::c_runtime::Storage*;
    using To = SQDataBuilder<T, Allocator>;

    static int64_t match(From from) {
        if (from->kind == SVS_STORAGE_KIND_SQ) {
            auto sq = static_cast<const c_runtime::StorageSQ*>(from);
            if (sq->data_type == svs::datatype_v<T>) {
                return svs::lib::perfect_match;
            }
        }
        return svs::lib::invalid_match;
    }

    static To convert(From SVS_UNUSED(from)) { return To{}; }
};

template <bool UseBlocked, typename F> void for_sq_specializations(F&& f) {
    using int8_alloc = svs::c_runtime::MaybeBlockedAlloc<int8_t, UseBlocked>;
    using uint8_alloc = svs::c_runtime::MaybeBlockedAlloc<uint8_t, UseBlocked>;
#define X(T, A, D) f.template operator()<SQDataBuilder<T, A>, D>();
#define XX(T, A) X(T, A, DistanceL2) X(T, A, DistanceIP) X(T, A, DistanceCosineSimilarity)
    XX(uint8_t, uint8_alloc)
    XX(int8_t, int8_alloc)
#undef XX
#undef X
}

} // namespace svs
