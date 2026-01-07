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

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC

#include "svs/c_api/svs_c.h"

#include "storage.hpp"
#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/data/simple.h>
#include <svs/lib/datatype.h>
#include <svs/lib/dispatcher.h>
#include <svs/lib/memory.h>
#include <svs/lib/threads/threadpool.h>
#include <svs/lib/type_traits.h>

#ifdef SVS_LEANVEC_HEADER
#include SVS_LEANVEC_HEADER
#else // SVS_LEANVEC_HEADER not defined
#ifdef SVS_RUNTIME_ENABLE_IVF
#include <svs/extensions/ivf/leanvec.h>
#endif
#include <svs/extensions/vamana/leanvec.h>
#endif // SVS_LEANVEC_HEADER

#include <filesystem>
#include <stdexcept>

namespace svs {

template <size_t I1, size_t I2, typename Allocator = svs::lib::Allocator<std::byte>>
class LeanVecDataBuilder {
    size_t leanvec_dims_;

  public:
    LeanVecDataBuilder(size_t leanvec_dims)
        : leanvec_dims_(leanvec_dims) {}

    using data_type = svs::leanvec::LeanDataset<
        svs::leanvec::UsingLVQ<I1>,
        svs::leanvec::UsingLVQ<I2>,
        svs::Dynamic,
        svs::Dynamic,
        Allocator>;
    using allocator_type = Allocator;

    template <Arithmetic T>
    data_type build(
        svs::data::ConstSimpleDataView<T> view,
        svs::threads::ThreadPoolHandle& pool,
        const allocator_type& allocator = {}
    ) {
        return data_type::reduce(
            view, std::nullopt, pool, 0, svs::lib::MaybeStatic{leanvec_dims_}, allocator
        );
    }

    data_type
    load(const std::filesystem::path& path, const allocator_type& allocator = {}) {
        return svs::lib::load_from_disk<data_type>(path, allocator);
    }
};

template <size_t I1, size_t I2, typename Allocator>
struct lib::
    DispatchConverter<const c_runtime::Storage*, LeanVecDataBuilder<I1, I2, Allocator>> {
    using From = const svs::c_runtime::Storage*;
    using To = LeanVecDataBuilder<I1, I2, Allocator>;

    static int64_t match(From from) {
        if (from->kind == SVS_STORAGE_KIND_LEANVEC) {
            auto leanvec = static_cast<const c_runtime::StorageLeanVec*>(from);
            if (leanvec->primary_bits == I1 && leanvec->secondary_bits == I2) {
                return svs::lib::perfect_match;
            }
        }
        return svs::lib::invalid_match;
    }

    static To convert(From from) {
        auto leanvec = static_cast<const c_runtime::StorageLeanVec*>(from);
        return To{leanvec->lenavec_dims};
    }
};

template <bool UseBlocked, typename F> void for_leanvec_specializations(F&& f) {
    using byte_alloc = svs::c_runtime::MaybeBlockedAlloc<std::byte, UseBlocked>;

#define X(P, S, D) f.template operator()<LeanVecDataBuilder<P, S, byte_alloc>, D>();
#define XX(P, S) X(P, S, DistanceL2) X(P, S, DistanceIP) X(P, S, DistanceCosineSimilarity)
    // Pattern:
    // PrimaryBits, SecondaryBits, Distance
    XX(4, 4)
    XX(4, 8)
    XX(8, 8)
#undef XX
#undef X
}

} // namespace svs

#else // SVS_RUNTIME_ENABLE_LVQ_LEANVEC not enabled
namespace svs {
// Define empty stubs for LeanVec-related functions when LVQ/LeanVec support is disabled
template <bool UseBlocked, typename F> void for_leanvec_specializations(F&&) {}
} // namespace svs

#endif // SVS_RUNTIME_ENABLE_LVQ_LEANVEC
