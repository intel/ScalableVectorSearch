/*
 * Copyright 2025 Intel Corporation
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

#include "svs/fallback/fallback_mode.h"
#include "svs/leanvec/leanvec_common.h"
#include "svs/quantization/lvq/lvq_fallback.h"

// #include leanvec_common.h

namespace fallback = svs::fallback;

namespace svs {
namespace leanvec {

template <size_t Extent> struct LeanVecMatrices {
  public:
    using leanvec_matrix_type = data::SimpleData<float, Extent>;

    LeanVecMatrices() = default;
    LeanVecMatrices(leanvec_matrix_type data_matrix, leanvec_matrix_type query_matrix)
        : data_matrix_{std::move(data_matrix)}
        , query_matrix_{std::move(query_matrix)} {
        // Check that the size and dimensionality of both the matrices should be same
        if (data_matrix_.size() != query_matrix_.size()) {
            throw ANNEXCEPTION("Mismatched data and query matrix sizes!");
        }
        if (data_matrix_.dimensions() != query_matrix_.dimensions()) {
            throw ANNEXCEPTION("Mismatched data and query matrix dimensions!");
        }
    }

  private:
    leanvec_matrix_type data_matrix_;
    leanvec_matrix_type query_matrix_;
};

// is this necessary or duplicate of LVQ?
namespace detail {
template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;

template <typename T, typename A, bool = is_blocked<A>> struct select_rebind_allocator {
    using type = lib::rebind_allocator_t<T, A>;
};
template <typename T, typename A> struct select_rebind_allocator<T, A, true> {
    using base_allocator = typename A::allocator_type;
    using rebind_base_allocator = lib::rebind_allocator_t<T, base_allocator>;
    using type = data::Blocked<rebind_base_allocator>;
};
template <typename T, typename A>
using select_rebind_allocator_t = typename select_rebind_allocator<T, A>::type;
} // namespace detail

template <
    typename T1,
    typename T2,
    size_t LeanVecDims,
    size_t Extent,
    typename Alloc = lib::Allocator<std::byte>>
class LeanDataset {
  public:
    using allocator_type = detail::select_rebind_allocator_t<float, Alloc>;

  private:
    data::SimpleData<float, Extent, allocator_type> primary_;

  public:
    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using leanvec_matrices_type = LeanVecMatrices<LeanVecDims>;
    using const_value_type =
        typename data::SimpleData<float, Extent, allocator_type>::const_value_type;
    using element_type = float;
    using value_type = const_value_type;
    using primary_type = data::SimpleData<float, Extent, allocator_type>;

    LeanDataset(primary_type primary)
        : primary_{std::move(primary)} {
        if (fallback::get_mode() == fallback::FallbackMode::Error) {
            throw fallback::UnsupportedHardwareError();
        } else if (fallback::get_mode() == fallback::FallbackMode::Warning) {
            fmt::print(fallback::fallback_warning);
        }
    }

    size_t size() const { return primary_.size(); }
    size_t dimensions() const { return primary_.dimensions(); }
    const_value_type get_datum(size_t i) const { return primary_.get_datum(i); }
    void prefetch(size_t i) const { primary_.prefetch(i); }
    template <typename U, size_t N> void set_datum(size_t i, std::span<U, N> datum) {
        primary_.set_datum(i, datum);
    }

    void resize(size_t new_size)
        requires is_resizeable
    {
        primary_.resize(new_size);
    }
    template <std::integral I, threads::ThreadPool Pool>
        requires is_resizeable
    void
    compact(std::span<const I> new_to_old, Pool& threadpool, size_t batchsize = 1'000'000) {
        primary_.compact(new_to_old, threadpool, batchsize);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LeanDataset reduce(
        const Dataset& data,
        size_t num_threads = 1,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const Alloc& allocator = {}
    ) {
        return reduce(data, std::nullopt, num_threads, alignment, leanvec_dims, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LeanDataset reduce(
        const Dataset& data,
        std::optional<leanvec_matrices_type> matrices,
        size_t num_threads = 1,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const Alloc& allocator = {}
    ) {
        auto pool = threads::NativeThreadPool{num_threads};
        return reduce(data, std::move(matrices), pool, alignment, leanvec_dims, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LeanDataset reduce(
        const Dataset& data,
        std::optional<leanvec_matrices_type> SVS_UNUSED(matrices),
        Pool& SVS_UNUSED(threadpool),
        size_t SVS_UNUSED(alignment) = 0,
        lib::MaybeStatic<LeanVecDims> SVS_UNUSED(leanvec_dims) = {},
        const Alloc& allocator = {}
    ) {
        primary_type primary =
            primary_type{data.size(), data.dimensions(), allocator_type{allocator}};
        svs::data::copy(data, primary);
        return LeanDataset{primary};
    }

    static constexpr lib::Version save_version = fallback_save_version;
    static constexpr std::string_view serialization_schema = fallback_schema;
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            serialization_schema, save_version, {SVS_LIST_SAVE_(primary, ctx)}
        );
    }

    static LeanDataset load(
        const lib::LoadTable& table,
        size_t SVS_UNUSED(alignment) = 0,
        const Alloc& allocator = {}
    ) {
        return LeanDataset{SVS_LOAD_MEMBER_AT_(table, primary, allocator)};
    }
};

} // namespace leanvec
} // namespace svs
