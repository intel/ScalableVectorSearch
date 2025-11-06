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

// TODO emplace content of IndexSVSImplUtils.h here
#include "IndexSVSImplUtils.h"

// TODO remove unused includes
#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/cpuid.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/exception.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

#ifndef SVS_LVQ_HEADER
#define SVS_LVQ_HEADER "svs/quantization/lvq/lvq.h"
#endif

#ifndef SVS_LEANVEC_HEADER
#define SVS_LEANVEC_HEADER "svs/leanvec/leanvec.h"
#endif

#include SVS_LVQ_HEADER
#include SVS_LEANVEC_HEADER

namespace svs::runtime {

class StatusException : public svs::lib::ANNException {
  public:
    StatusException(const svs::runtime::ErrorCode& code, const std::string& message)
        : svs::lib::ANNException(message)
        , errcode_{code} {}

    svs::runtime::ErrorCode code() const { return errcode_; }

  private:
    svs::runtime::ErrorCode errcode_;
};

#define SVS_RUNTIME_TRY_BEGIN try {
#define SVS_RUNTIME_TRY_END                                                                \
    }                                                                                      \
    catch (const svs::runtime::StatusException& ex) {                                      \
        return svs::runtime::Status(ex.code(), ex.what());                                 \
    }                                                                                      \
    catch (const std::invalid_argument& ex) {                                              \
        return svs::runtime::Status(svs::runtime::ErrorCode::INVALID_ARGUMENT, ex.what()); \
    }                                                                                      \
    catch (const std::runtime_error& ex) {                                                 \
        return svs::runtime::Status(svs::runtime::ErrorCode::RUNTIME_ERROR, ex.what());    \
    }                                                                                      \
    catch (const std::exception& ex) {                                                     \
        return svs::runtime::Status(svs::runtime::ErrorCode::UNKNOWN_ERROR, ex.what());    \
    }                                                                                      \
    catch (...) {                                                                          \
        return svs::runtime::Status(                                                       \
            svs::runtime::ErrorCode::UNKNOWN_ERROR, "An unknown error has occurred."       \
        );                                                                                 \
    }

using LeanVecMatricesType = svs::leanvec::LeanVecMatrices<svs::Dynamic>;

namespace storage {

template <typename T> inline constexpr bool is_simple_dataset = false;
template <typename Elem, size_t Extent, typename Allocator>
inline constexpr bool is_simple_dataset<svs::data::SimpleData<Elem, Extent, Allocator>> =
    true;

template <typename T>
concept IsSimpleDataset = is_simple_dataset<T>;

inline constexpr bool is_lvq_storage(StorageKind kind) {
    return kind == StorageKind::LVQ4x0 || kind == StorageKind::LVQ4x4 ||
           kind == StorageKind::LVQ4x8;
}

template <StorageKind K>
concept IsLVQStorageKind = is_lvq_storage(K);

inline constexpr bool is_leanvec_storage(StorageKind kind) {
    return kind == StorageKind::LeanVec4x4 || kind == StorageKind::LeanVec4x8 ||
           kind == StorageKind::LeanVec8x8;
}

template <StorageKind K>
concept IsLeanVecStorageKind = is_leanvec_storage(K);

// Storage kind processing
// Most kinds map to std::byte storage, but some have specific element types.
// Storage kind tag types for function argument deduction
template <StorageKind K> struct StorageKindTag {
    static constexpr StorageKind value = K;
};

using FP32Tag = StorageKindTag<StorageKind::FP32>;
using FP16Tag = StorageKindTag<StorageKind::FP16>;
using SQI8Tag = StorageKindTag<StorageKind::SQI8>;
using LVQ4x0Tag = StorageKindTag<StorageKind::LVQ4x0>;
using LVQ4x4Tag = StorageKindTag<StorageKind::LVQ4x4>;
using LVQ4x8Tag = StorageKindTag<StorageKind::LVQ4x8>;
using LeanVec4x4Tag = StorageKindTag<StorageKind::LeanVec4x4>;
using LeanVec4x8Tag = StorageKindTag<StorageKind::LeanVec4x8>;
using LeanVec8x8Tag = StorageKindTag<StorageKind::LeanVec8x8>;

// Storage types
template <typename T>
using SimpleDatasetType =
    svs::data::SimpleData<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

template <typename T>
using SQDatasetType = svs::quantization::scalar::
    SQDataset<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

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

template <StorageKind K> struct StorageType;
template <StorageKind K> using StorageType_t = typename StorageType<K>::type;

template <> struct StorageType<StorageKind::FP32> {
    using type = SimpleDatasetType<float>;
};

template <> struct StorageType<StorageKind::FP16> {
    using type = SimpleDatasetType<svs::Float16>;
};

template <> struct StorageType<StorageKind::SQI8> {
    using type = SQDatasetType<std::int8_t>;
};

template <> struct StorageType<StorageKind::LVQ4x0> {
    using type = LVQDatasetType<4, 0>;
};

template <> struct StorageType<StorageKind::LVQ4x4> {
    using type = LVQDatasetType<4, 4>;
};

template <> struct StorageType<StorageKind::LVQ4x8> {
    using type = LVQDatasetType<4, 8>;
};

template <> struct StorageType<StorageKind::LeanVec4x4> {
    using type = LeanDatasetType<4, 4>;
};

template <> struct StorageType<StorageKind::LeanVec4x8> {
    using type = LeanDatasetType<4, 8>;
};

template <> struct StorageType<StorageKind::LeanVec8x8> {
    using type = LeanDatasetType<8, 8>;
};

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
        data, std::move(matrices), pool, 0, svs::lib::MaybeStatic<svs::Dynamic>{leanvec_d}
    );
}

template <typename StorageTag, typename... Args>
auto make_storage(StorageTag&& SVS_UNUSED(tag), Args&&... args) {
    return make_storage<StorageType_t<StorageTag::value>>(std::forward<Args>(args)...);
}

template <typename F, typename... Args>
static auto dispatch_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    using SK = StorageKind;
    switch (kind) {
        case SK::FP32:
            return f(FP32Tag{}, std::forward<Args>(args)...);
        case SK::FP16:
            return f(FP16Tag{}, std::forward<Args>(args)...);
        case SK::SQI8:
            return f(SQI8Tag{}, std::forward<Args>(args)...);
        case SK::LVQ4x0:
            return f(LVQ4x0Tag{}, std::forward<Args>(args)...);
        case SK::LVQ4x4:
            return f(LVQ4x4Tag{}, std::forward<Args>(args)...);
        case SK::LVQ4x8:
            return f(LVQ4x8Tag{}, std::forward<Args>(args)...);
        case SK::LeanVec4x4:
            return f(LeanVec4x4Tag{}, std::forward<Args>(args)...);
        case SK::LeanVec4x8:
            return f(LeanVec4x8Tag{}, std::forward<Args>(args)...);
        case SK::LeanVec8x8:
            return f(LeanVec8x8Tag{}, std::forward<Args>(args)...);
        default:
            throw ANNEXCEPTION("not supported SVS storage kind");
    }
}

} // namespace storage
} // namespace svs::runtime
