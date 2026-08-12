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

#ifdef SVS_LVQ_HEADER
#include SVS_LVQ_HEADER
#else // SVS_LVQ_HEADER not defined
#ifdef SVS_RUNTIME_ENABLE_IVF
#include <svs/extensions/ivf/lvq.h>
#endif
#include <svs/extensions/vamana/lvq.h>
#endif // SVS_LVQ_HEADER

#include <filesystem>
#include <stdexcept>

namespace svs {

template <
    size_t PrimaryBits,
    size_t ResidualBits,
    typename Allocator = svs::lib::Allocator<std::byte>>
class LVQDataBuilder {
  public:
    LVQDataBuilder() {}

    // Follow the logic of svs::leanvec::detail::PickContainer which looks like:
    // "Use Turbo-encoding for 4-bit LVQ."
    using Sequential = svs::quantization::lvq::Sequential;
    using Turbo16x8 = svs::quantization::lvq::Turbo<16, 8>;
    template <size_t Primary, size_t Residual>
    using AutoStrategy = std::conditional_t<(Primary == 4), Turbo16x8, Sequential>;

    using data_type = svs::quantization::lvq::LVQDataset<
        PrimaryBits,
        ResidualBits,
        svs::Dynamic,
        AutoStrategy<PrimaryBits, ResidualBits>,
        Allocator>;
    using allocator_type = Allocator;

    template <Arithmetic T>
    data_type build(
        svs::data::ConstSimpleDataView<T> view,
        svs::threads::ThreadPoolHandle& pool,
        const allocator_type& allocator = {}
    ) {
        return data_type::compress(view, pool, 0, allocator);
    }

    data_type
    load(const std::filesystem::path& path, const allocator_type& allocator = {}) {
        return svs::lib::load_from_disk<data_type>(path, allocator);
    }

    static constexpr size_t primary_element_size(size_t dimension, size_t alignment = 0) {
        using primary_type = typename data_type::primary_type;
        using layout_type = typename primary_type::helper_type;
        using layout_dims_type = svs::lib::MaybeStatic<data_type::extent>;
        const auto layout_dims = layout_dims_type{dimension};
        return primary_type::compute_data_dimensions(layout_type{layout_dims}, alignment);
    }

    static constexpr size_t residual_element_size(size_t dims) {
        if constexpr (ResidualBits == 0) {
            return 0;
        } else {
            using residual_type = typename data_type::residual_type;
            using dims_type = svs::lib::MaybeStatic<data_type::extent>;
            auto residual_dims = dims_type{dims};
            return residual_type::total_bytes(residual_dims);
        }
    }

    size_t estimate_size(
        size_t num_vectors, size_t dimension, const allocator_type& allocator = {}
    ) const {
        const size_t alignment = 0; // Assuming no specific alignment for estimation

        const auto primary_element_sz = primary_element_size(dimension, alignment);
        const auto primary_size =
            svs::c_runtime::adjust_blocked_size(num_vectors, primary_element_sz, allocator);

        const auto residual_element_sz = residual_element_size(dimension);
        const auto residual_size = svs::c_runtime::adjust_blocked_size(
            num_vectors, residual_element_sz, allocator
        );

        // Assuming a single centroid for estimation purposes
        const size_t num_centroids = 1; // Assuming 1 centroid for estimation
        // TODO: Fix the actual memory breakdown reported by index by implementing
        // dataset_allocated_bytes() specialization for LVQDataset.
        const size_t centroid_size = 0; // Skipping centroids for estimation
        // const auto centroid_size =
        //     sizeof(typename data_type::centroid_type::element_type) * dimension;

        const auto total_size =
            primary_size + residual_size + num_centroids * centroid_size;
        return total_size;
    }
};

template <size_t PrimaryBits, size_t ResidualBits, typename Alloc>
struct lib::DispatchConverter<
    const c_runtime::Storage*,
    LVQDataBuilder<PrimaryBits, ResidualBits, Alloc>> {
    using From = const svs::c_runtime::Storage*;
    using To = LVQDataBuilder<PrimaryBits, ResidualBits, Alloc>;

    static int64_t match(From from) {
        if (from->kind == SVS_STORAGE_KIND_LVQ) {
            auto lvq = static_cast<const c_runtime::StorageLVQ*>(from);
            if (lvq->primary_bits == PrimaryBits && lvq->residual_bits == ResidualBits) {
                return svs::lib::perfect_match;
            }
        }
        return svs::lib::invalid_match;
    }

    static To convert(From SVS_UNUSED(from)) { return To{}; }
};

template <bool UseBlocked, typename F> void for_lvq_specializations(F&& f) {
    using byte_alloc = svs::c_runtime::MaybeBlockedAlloc<std::byte, UseBlocked>;
#define X(P, S, D) f.template operator()<LVQDataBuilder<P, S, byte_alloc>, D>();
#define XX(P, S) X(P, S, DistanceL2) X(P, S, DistanceIP) X(P, S, DistanceCosineSimilarity)
    // Pattern:
    // PrimaryBits, SecondaryBits, Distance
    XX(4, 0)
    XX(8, 0)
    XX(4, 4)
    XX(4, 8)
#undef XX
#undef X
}

} // namespace svs

#else // SVS_RUNTIME_ENABLE_LVQ_LEANVEC not enabled
namespace svs {
// Define empty stubs for LVQ-related functions when LVQ/LeanVec support is disabled
template <bool UseBlocked, typename F> void for_lvq_specializations(F&&) {}
} // namespace svs

#endif // SVS_RUNTIME_ENABLE_LVQ_LEANVEC
