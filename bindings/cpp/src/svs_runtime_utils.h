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

#include <svs/runtime/api_defs.h>

// TODO remove unused includes
#include <algorithm>
#include <concepts>
#include <functional>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include <omp.h>

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/exception.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
#ifndef SVS_LVQ_HEADER
#define SVS_LVQ_HEADER "svs/quantization/lvq/lvq.h"
#endif

#ifndef SVS_LEANVEC_HEADER
#define SVS_LEANVEC_HEADER "svs/leanvec/leanvec.h"
#endif

#include SVS_LVQ_HEADER
#include SVS_LEANVEC_HEADER
#include <svs/cpuid.h>
#endif

namespace svs::runtime {

inline svs::DistanceType to_svs_distance(MetricType metric) {
    switch (metric) {
        case MetricType::L2:
            return svs::DistanceType::L2;
        case MetricType::INNER_PRODUCT:
            return svs::DistanceType::MIP;
    }
    throw ANNEXCEPTION("unreachable reached"); // Make GCC happy
}

class StatusException : public svs::lib::ANNException {
  public:
    StatusException(const svs::runtime::ErrorCode& code, const std::string& message)
        : svs::lib::ANNException(message)
        , errcode_{code} {}

    svs::runtime::ErrorCode code() const { return errcode_; }

  private:
    svs::runtime::ErrorCode errcode_;
};

template <typename Callable>
inline auto runtime_error_wrapper(Callable&& func) noexcept -> Status {
    try {
        func();
        return Status_Ok;
    } catch (const svs::runtime::StatusException& ex) {
        return Status(ex.code(), ex.what());
    } catch (const std::invalid_argument& ex) {
        return Status(ErrorCode::INVALID_ARGUMENT, ex.what());
    } catch (const std::runtime_error& ex) {
        return Status(ErrorCode::RUNTIME_ERROR, ex.what());
    } catch (const std::exception& ex) {
        return Status(ErrorCode::UNKNOWN_ERROR, ex.what());
    } catch (...) {
        return Status(ErrorCode::UNKNOWN_ERROR, "An unknown error has occurred.");
    }
}

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
using LeanVecMatricesType = svs::leanvec::LeanVecMatrices<svs::Dynamic>;
#endif

namespace storage {

// Simplified trait checking
template <typename T> inline constexpr bool is_simple_dataset = false;
template <typename Elem, size_t Extent, typename Allocator>
inline constexpr bool is_simple_dataset<svs::data::SimpleData<Elem, Extent, Allocator>> =
    true;

template <typename T>
concept IsSimpleDataset = is_simple_dataset<T>;

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
// Consolidated storage kind checks using constexpr functions
inline constexpr bool is_lvq_storage(StorageKind kind) {
    return kind == StorageKind::LVQ4x0 || kind == StorageKind::LVQ4x4 ||
           kind == StorageKind::LVQ4x8;
}

inline constexpr bool is_leanvec_storage(StorageKind kind) {
    return kind == StorageKind::LeanVec4x4 || kind == StorageKind::LeanVec4x8 ||
           kind == StorageKind::LeanVec8x8;
}
#endif

// Storage kind processing
// Most kinds map to std::byte storage, but some have specific element types.
// Storage kind tag types for function argument deduction
template <StorageKind K> struct StorageKindTag {
    static constexpr StorageKind value = K;
};

#define SVS_DEFINE_STORAGE_KIND_TAG(Kind) \
    using Kind##Tag = StorageKindTag<StorageKind::Kind>

SVS_DEFINE_STORAGE_KIND_TAG(FP32);
SVS_DEFINE_STORAGE_KIND_TAG(FP16);
SVS_DEFINE_STORAGE_KIND_TAG(SQI8);
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
SVS_DEFINE_STORAGE_KIND_TAG(LVQ4x0);
SVS_DEFINE_STORAGE_KIND_TAG(LVQ4x4);
SVS_DEFINE_STORAGE_KIND_TAG(LVQ4x8);
SVS_DEFINE_STORAGE_KIND_TAG(LeanVec4x4);
SVS_DEFINE_STORAGE_KIND_TAG(LeanVec4x8);
SVS_DEFINE_STORAGE_KIND_TAG(LeanVec8x8);
#endif

#undef SVS_DEFINE_STORAGE_KIND_TAG

template <typename T> inline constexpr bool is_storage_tag = false;
template <StorageKind K> inline constexpr bool is_storage_tag<StorageKindTag<K>> = true;

template <typename T>
concept StorageTag = is_storage_tag<T>;

// Storage types
template <typename T>
using SimpleDatasetType =
    svs::data::SimpleData<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

template <typename T>
using SQDatasetType = svs::quantization::scalar::
    SQDataset<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
template <size_t Primary, size_t Residual>
using LVQDatasetType = svs::quantization::lvq::LVQDataset<
    Primary,
    Residual,
    svs::Dynamic,
    svs::quantization::lvq::Turbo<16, 8>,
    svs::data::Blocked<svs::lib::Allocator<std::byte>>>;

template <size_t I1, size_t I2>
using LeanDatasetType = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<I1>,
    svs::leanvec::UsingLVQ<I2>,
    svs::Dynamic,
    svs::Dynamic,
    svs::data::Blocked<svs::lib::Allocator<std::byte>>>;
#endif

// Storage type mapping - use macro to reduce repetition
template <StorageTag Tag> struct StorageType;
template <StorageTag Tag> using StorageType_t = typename StorageType<Tag>::type;

#define DEFINE_STORAGE_TYPE(Kind, ...)          \
    template <> struct StorageType<Kind##Tag> { \
        using type = __VA_ARGS__;               \
    }

DEFINE_STORAGE_TYPE(FP32, SimpleDatasetType<float>);
DEFINE_STORAGE_TYPE(FP16, SimpleDatasetType<svs::Float16>);
DEFINE_STORAGE_TYPE(SQI8, SQDatasetType<std::int8_t>);
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
DEFINE_STORAGE_TYPE(LVQ4x0, LVQDatasetType<4, 0>);
DEFINE_STORAGE_TYPE(LVQ4x4, LVQDatasetType<4, 4>);
DEFINE_STORAGE_TYPE(LVQ4x8, LVQDatasetType<4, 8>);
DEFINE_STORAGE_TYPE(LeanVec4x4, LeanDatasetType<4, 4>);
DEFINE_STORAGE_TYPE(LeanVec4x8, LeanDatasetType<4, 8>);
DEFINE_STORAGE_TYPE(LeanVec8x8, LeanDatasetType<8, 8>);
#endif

#undef DEFINE_STORAGE_TYPE

// Storage factory functions
template <IsSimpleDataset StorageType, svs::threads::ThreadPool Pool>
StorageType make_storage(const svs::data::ConstSimpleDataView<float>& data, Pool& pool) {
    StorageType result(data.size(), data.dimensions());
    svs::threads::parallel_for(
        pool,
        svs::threads::StaticPartition(result.size()),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for (auto i : is) {
                result.set_datum(i, data.get_datum(i));
            }
        }
    );
    return result;
}

template <svs::quantization::scalar::IsSQData SQStorageType, svs::threads::ThreadPool Pool>
SQStorageType make_storage(const svs::data::ConstSimpleDataView<float>& data, Pool& pool) {
    return SQStorageType::compress(data, pool);
}

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
template <
    svs::quantization::lvq::IsLVQDataset LVQStorageType,
    svs::threads::ThreadPool Pool>
LVQStorageType make_storage(const svs::data::ConstSimpleDataView<float>& data, Pool& pool) {
    return LVQStorageType::compress(data, pool, 0);
}

template <svs::leanvec::IsLeanDataset LeanVecStorageType, svs::threads::ThreadPool Pool>
LeanVecStorageType make_storage(
    const svs::data::ConstSimpleDataView<float>& data,
    Pool& pool,
    size_t leanvec_d = 0,
    std::optional<LeanVecMatricesType> matrices = std::nullopt
) {
    if (leanvec_d == 0) {
        leanvec_d = (data.dimensions() + 1) / 2;
    }
    return LeanVecStorageType::reduce(
        data, std::move(matrices), pool, 0, svs::lib::MaybeStatic{leanvec_d}
    );
}
#endif

template <StorageTag Tag, typename... Args>
auto make_storage(Tag&& SVS_UNUSED(tag), Args&&... args) {
    return make_storage<StorageType_t<Tag>>(std::forward<Args>(args)...);
}

inline bool is_supported_storage_kind(StorageKind kind) {
    if (is_lvq_storage(kind) || is_leanvec_storage(kind)) {
        return svs::detail::lvq_leanvec_enabled();
    }
    return true;
}

template <typename F, typename... Args>
auto dispatch_storage_kind(StorageKind kind, F&& f, Args&&... args) {
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
    if (!is_supported_storage_kind(kind)) {
        throw StatusException(
            ErrorCode::NOT_IMPLEMENTED, "Requested storage kind is not supported by CPU"
        );
    }
#endif
    switch (kind) {
        case StorageKind::FP32:
            return f(FP32Tag{}, std::forward<Args>(args)...);
        case StorageKind::FP16:
            return f(FP16Tag{}, std::forward<Args>(args)...);
        case StorageKind::SQI8:
            return f(SQI8Tag{}, std::forward<Args>(args)...);
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
        case StorageKind::LVQ4x0:
            return f(LVQ4x0Tag{}, std::forward<Args>(args)...);
        case StorageKind::LVQ4x4:
            return f(LVQ4x4Tag{}, std::forward<Args>(args)...);
        case StorageKind::LVQ4x8:
            return f(LVQ4x8Tag{}, std::forward<Args>(args)...);
        case StorageKind::LeanVec4x4:
            return f(LeanVec4x4Tag{}, std::forward<Args>(args)...);
        case StorageKind::LeanVec4x8:
            return f(LeanVec4x8Tag{}, std::forward<Args>(args)...);
        case StorageKind::LeanVec8x8:
            return f(LeanVec8x8Tag{}, std::forward<Args>(args)...);
#endif
        default:
            throw ANNEXCEPTION("not supported SVS storage kind");
    }
}
} // namespace storage

inline svs::threads::ThreadPoolHandle default_threadpool() {
    return svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads())
    );
}
} // namespace svs::runtime
