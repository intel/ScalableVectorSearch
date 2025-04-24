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

#include "svs/core/data/simple.h"
#include "svs/fallback/fallback_mode.h"
#include "svs/lib/saveload/save.h"
#include "svs/lib/threads.h"
#include "svs/quantization/lvq/lvq_common.h"

namespace fallback = svs::fallback;

namespace svs {
namespace quantization {
namespace lvq {

struct Sequential {
    static constexpr std::string_view name() { return "sequential"; }
};

template <size_t Lanes, size_t ElementsPerLane> struct Turbo {
    static constexpr std::string name() {
        return fmt::format("turbo<{}x{}>", Lanes, ElementsPerLane);
    }
};

namespace detail {

// Trait to identify and dispatch based on the Turbo class itself.
template <typename T> inline constexpr bool is_turbo_like_v = false;
template <typename T> inline constexpr bool is_lvq_packing_strategy_v = false;

template <size_t Lanes, size_t ElementsPerLane>
inline constexpr bool is_turbo_like_v<lvq::Turbo<Lanes, ElementsPerLane>> = true;

template <> inline constexpr bool is_lvq_packing_strategy_v<lvq::Sequential> = true;
template <size_t Lanes, size_t ElementsPerLane>

inline constexpr bool is_lvq_packing_strategy_v<lvq::Turbo<Lanes, ElementsPerLane>> = true;

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

template <typename T>
concept LVQPackingStrategy = detail::is_lvq_packing_strategy_v<T>;

template <typename T>
concept TurboLike = detail::is_turbo_like_v<T>;

// LVQDataset
template <
    size_t Primary,
    size_t Residual = 0,
    size_t Extent = Dynamic,
    LVQPackingStrategy Strategy = Sequential,
    typename Alloc = lib::Allocator<std::byte>>
class LVQDataset {
  public:
    using allocator_type = detail::select_rebind_allocator_t<float, Alloc>;

  private:
    data::SimpleData<float, Extent, allocator_type> primary_;

  public:
    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using const_value_type =
        typename data::SimpleData<float, Extent, allocator_type>::const_value_type;
    using element_type = float;
    using value_type = const_value_type;
    using primary_type = data::SimpleData<float, Extent, allocator_type>;
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
    LVQDataset(Dataset primary)
        : primary_{primary} {
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

    template <typename QueryType, size_t N>
    void set_datum(
        size_t i, std::span<QueryType, N> datum, size_t SVS_UNUSED(centroid_selector) = 0
    ) {
        primary_.set_datum(i, datum);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(const Dataset& data, const Alloc& allocator = {}) {
        return compress(data, 1, 0, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(
        const Dataset& data,
        size_t num_threads,
        size_t alignment,
        const Alloc& allocator = {}
    ) {
        auto pool = threads::NativeThreadPool{num_threads};
        return compress(data, pool, alignment, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LVQDataset compress(
        const Dataset& data,
        Pool& SVS_UNUSED(threadpool),
        size_t SVS_UNUSED(alignment),
        const Alloc& allocator = {}
    ) {
        primary_type primary =
            primary_type{data.size(), data.dimensions(), allocator_type{allocator}};
        svs::data::copy(data, primary);
        return LVQDataset{primary};
    }

    static constexpr lib::Version save_version = fallback_save_version;
    static constexpr std::string_view serialization_schema = fallback_serialization_schema;
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            serialization_schema, save_version, {SVS_LIST_SAVE_(primary, ctx)}
        );
    }

    static LVQDataset load(
        const lib::LoadTable& table,
        size_t SVS_UNUSED(alignment) = 0,
        const Alloc& allocator = {}
    ) {
        return LVQDataset{SVS_LOAD_MEMBER_AT_(table, primary, allocator)};
    }
};

// No constraints on fallback for primary, residual, strategy
template <size_t Primary, size_t Residual>
inline bool check_primary_residual(size_t SVS_UNUSED(p), size_t SVS_UNUSED(r)) {
    return false;
}

inline bool check_strategy_match(int64_t SVS_UNUSED(strategy_match)) { return false; }

namespace detail {

template <LVQPackingStrategy Strategy>
constexpr bool is_compatible(LVQStrategyDispatch SVS_UNUSED(strategy)) {
    return true;
}

} // namespace detail

} // namespace lvq
} // namespace quantization
} // namespace svs
