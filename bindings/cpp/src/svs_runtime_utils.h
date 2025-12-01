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

#ifdef SVS_LVQ_HEADER
#include SVS_LVQ_HEADER
#endif

#ifdef SVS_LEANVEC_HEADER
#include SVS_LEANVEC_HEADER
#endif

#if defined(SVS_LEANVEC_HEADER) || defined(SVS_LVQ_HEADER)
#include <svs/cpuid.h>
#else
namespace svs::detail {
inline bool lvq_leanvec_enabled() { return false; }
} // namespace svs::detail
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

inline void set_if_specified(bool& target, const OptionalBool& value) {
    if (is_specified(value)) {
        target = value.is_enabled();
    }
}

template <typename T> void set_if_specified(T& target, const T& value) {
    if (is_specified(value)) {
        target = value;
    }
}

template <typename T> void require_specified(const T& value, const char* name) {
    if (!is_specified(value)) {
        throw StatusException{
            ErrorCode::INVALID_ARGUMENT,
            std::string("The parameter '") + name + "' must be specified."};
    }
}

namespace storage {

// Consolidated storage kind checks using constexpr functions
inline constexpr bool is_lvq_storage(StorageKind kind) {
    return kind == StorageKind::LVQ4x0 || kind == StorageKind::LVQ4x4 ||
           kind == StorageKind::LVQ4x8;
}

inline constexpr bool is_leanvec_storage(StorageKind kind) {
    return kind == StorageKind::LeanVec4x4 || kind == StorageKind::LeanVec4x8 ||
           kind == StorageKind::LeanVec8x8;
}

inline bool is_supported_storage_kind(StorageKind kind) {
    if (is_lvq_storage(kind) || is_leanvec_storage(kind)) {
        return svs::detail::lvq_leanvec_enabled();
    }
    return true;
}

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
SVS_DEFINE_STORAGE_KIND_TAG(LVQ4x0);
SVS_DEFINE_STORAGE_KIND_TAG(LVQ4x4);
SVS_DEFINE_STORAGE_KIND_TAG(LVQ4x8);
SVS_DEFINE_STORAGE_KIND_TAG(LeanVec4x4);
SVS_DEFINE_STORAGE_KIND_TAG(LeanVec4x8);
SVS_DEFINE_STORAGE_KIND_TAG(LeanVec8x8);

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

// Storage type mapping

// Unsupported storage type defined as unique type to cause runtime error if used
struct UnsupportedStorageType {};

// clang-format off
template <StorageTag Tag> struct StorageType { using type = UnsupportedStorageType; };
template <StorageTag Tag> using StorageType_t = typename StorageType<Tag>::type;

template <> struct StorageType<FP32Tag> { using type = SimpleDatasetType<float>; };
template <> struct StorageType<FP16Tag> { using type = SimpleDatasetType<svs::Float16>; };
template <> struct StorageType<SQI8Tag> { using type = SQDatasetType<std::int8_t>; };
// clang-format on

// Storage factory
template <typename T> struct StorageFactory;

// Unsupported storage type factory returning runtime error when attempted to be used.
// Return type defined to simple to allow substitution in templates.
template <> struct StorageFactory<UnsupportedStorageType> {
    using StorageType = SimpleDatasetType<float>;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& SVS_UNUSED(data),
        Pool& SVS_UNUSED(pool),
        svs::lib::PowerOfTwo SVS_UNUSED(blocksize_bytes)
    ) {
        throw StatusException(
            ErrorCode::NOT_IMPLEMENTED, "Requested storage kind is not supported"
        );
    }

    template <typename... Args>
    static StorageType
    load(const std::filesystem::path& SVS_UNUSED(path), Args&&... SVS_UNUSED(args)) {
        throw StatusException(
            ErrorCode::NOT_IMPLEMENTED, "Requested storage kind is not supported"
        );
    }
};

template <typename ElementType> struct StorageFactory<SimpleDatasetType<ElementType>> {
    using StorageType = SimpleDatasetType<ElementType>;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        svs::lib::PowerOfTwo blocksize_bytes =
            svs::data::BlockingParameters::default_blocksize_bytes
    ) {
        auto parameters = svs::data::BlockingParameters{.blocksize_bytes = blocksize_bytes};
        typename StorageType::allocator_type alloc(parameters);
        StorageType result(data.size(), data.dimensions(), alloc);
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

    template <typename... Args>
    static StorageType load(const std::filesystem::path& path, Args&&... args) {
        return svs::lib::load_from_disk<StorageType>(path, SVS_FWD(args)...);
    }
};

// SQ Storage support
template <svs::quantization::scalar::IsSQData SQStorageType>
struct StorageFactory<SQStorageType> {
    using StorageType = SQStorageType;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        svs::lib::PowerOfTwo SVS_UNUSED(blocksize_bytes)
    ) {
        return SQStorageType::compress(data, pool);
    }

    template <typename... Args>
    static StorageType load(const std::filesystem::path& path, Args&&... args) {
        return svs::lib::load_from_disk<StorageType>(path, SVS_FWD(args)...);
    }
};

// LVQ Storage support
#ifdef SVS_LVQ_HEADER
template <size_t Primary, size_t Residual>
using LVQDatasetType = svs::quantization::lvq::LVQDataset<
    Primary,
    Residual,
    svs::Dynamic,
    svs::quantization::lvq::Turbo<16, 8>,
    svs::data::Blocked<svs::lib::Allocator<std::byte>>>;

// clang-format off
template <> struct StorageType<LVQ4x0Tag> { using type = LVQDatasetType<4, 0>; };
template <> struct StorageType<LVQ4x4Tag> { using type = LVQDatasetType<4, 4>; };
template <> struct StorageType<LVQ4x8Tag> { using type = LVQDatasetType<4, 8>; };
// clang-format on

template <svs::quantization::lvq::IsLVQDataset LVQStorageType>
struct StorageFactory<LVQStorageType> {
    using StorageType = LVQStorageType;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        svs::lib::PowerOfTwo SVS_UNUSED(blocksize_bytes)
    ) {
        return LVQStorageType::compress(data, pool, 0);
    }

    template <typename... Args>
    static StorageType load(const std::filesystem::path& path, Args&&... args) {
        return svs::lib::load_from_disk<StorageType>(path, SVS_FWD(args)...);
    }
};
#endif // SVS_LVQ_HEADER

#ifdef SVS_LEANVEC_HEADER
template <size_t I1, size_t I2>
using LeanDatasetType = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<I1>,
    svs::leanvec::UsingLVQ<I2>,
    svs::Dynamic,
    svs::Dynamic,
    svs::data::Blocked<svs::lib::Allocator<std::byte>>>;

// clang-format off
template <> struct StorageType<LeanVec4x4Tag> { using type = LeanDatasetType<4, 4>; };
template <> struct StorageType<LeanVec4x8Tag> { using type = LeanDatasetType<4, 8>; };
template <> struct StorageType<LeanVec8x8Tag> { using type = LeanDatasetType<8, 8>; };
// clang-format on

template <svs::leanvec::IsLeanDataset LeanVecStorageType>
struct StorageFactory<LeanVecStorageType> {
    using StorageType = LeanVecStorageType;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        svs::lib::PowerOfTwo SVS_UNUSED(blocksize_bytes),
        size_t leanvec_d = 0,
        std::optional<svs::leanvec::LeanVecMatrices<svs::Dynamic>> matrices = std::nullopt
    ) {
        if (leanvec_d == 0) {
            leanvec_d = (data.dimensions() + 1) / 2;
        }
        return LeanVecStorageType::reduce(
            data, std::move(matrices), pool, 0, svs::lib::MaybeStatic{leanvec_d}
        );
    }

    template <typename... Args>
    static StorageType load(const std::filesystem::path& path, Args&&... args) {
        return svs::lib::load_from_disk<StorageType>(path, SVS_FWD(args)...);
    }
};
#endif // SVS_LEANVEC_HEADER

template <StorageTag Tag, typename... Args>
auto make_storage(Tag&& SVS_UNUSED(tag), Args&&... args) {
    return StorageFactory<StorageType_t<Tag>>::init(std::forward<Args>(args)...);
}

template <StorageTag Tag, typename... Args>
auto load_storage(Tag&& SVS_UNUSED(tag), Args&&... args) {
    return StorageFactory<StorageType_t<Tag>>::load(std::forward<Args>(args)...);
}

template <typename F, typename... Args>
auto dispatch_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    if (!is_supported_storage_kind(kind)) {
        throw StatusException(
            ErrorCode::NOT_IMPLEMENTED, "Requested storage kind is not supported by CPU"
        );
    }
    switch (kind) {
        case StorageKind::FP32:
            return f(FP32Tag{}, std::forward<Args>(args)...);
        case StorageKind::FP16:
            return f(FP16Tag{}, std::forward<Args>(args)...);
        case StorageKind::SQI8:
            return f(SQI8Tag{}, std::forward<Args>(args)...);
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
