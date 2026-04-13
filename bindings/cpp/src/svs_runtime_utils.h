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

#include "svs/runtime/api_defs.h"

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/exception.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
#include <svs/cpuid.h>
#ifdef SVS_LVQ_HEADER
#include SVS_LVQ_HEADER
#else
#ifdef SVS_RUNTIME_ENABLE_IVF
#include <svs/extensions/ivf/lvq.h>
#endif
#include <svs/extensions/vamana/lvq.h>
#endif
#ifdef SVS_LEANVEC_HEADER
#include SVS_LEANVEC_HEADER
#else
#ifdef SVS_RUNTIME_ENABLE_IVF
#include <svs/extensions/ivf/leanvec.h>
#endif
#include <svs/extensions/vamana/leanvec.h>
#endif
#else
namespace svs::detail {
inline bool lvq_leanvec_enabled() { return false; }
} // namespace svs::detail
#endif

#include <algorithm>
#include <concepts>
#include <functional>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include <omp.h>

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

namespace storage {

// Consolidated storage kind checks using constexpr functions
inline constexpr bool is_lvq_storage(StorageKind kind) {
    return kind == StorageKind::LVQ4x0 || kind == StorageKind::LVQ8x0 ||
           kind == StorageKind::LVQ4x4 || kind == StorageKind::LVQ4x8;
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

template <typename A> struct AllocatorTypeExtractor {
    using type = A;
};

template <typename A>
concept AllocatorAwareType = requires { typename A::allocator_type; };

template <AllocatorAwareType A> struct AllocatorTypeExtractor<A> {
    using type = typename A::allocator_type;
};

template <typename A> using extract_allocator_t = typename AllocatorTypeExtractor<A>::type;

template <typename T, typename Alloc> struct ExtractedAllocatorRebinder {
    using type = lib::rebind_allocator_t<T, extract_allocator_t<Alloc>>;
};

template <typename T, typename Alloc>
struct ExtractedAllocatorRebinder<T, svs::data::Blocked<Alloc>> {
    using type = svs::data::Blocked<lib::rebind_allocator_t<T, extract_allocator_t<Alloc>>>;
};

template <typename T, typename Alloc>
using rebind_extracted_allocator_t = typename ExtractedAllocatorRebinder<T, Alloc>::type;

template <typename Alloc>
Alloc make_allocator()
    requires(!svs::data::is_blocked_v<Alloc>)
{
    return Alloc{};
}

template <typename Alloc>
Alloc make_allocator(svs::lib::PowerOfTwo blocksize_bytes)
    requires(svs::data::is_blocked_v<Alloc>)
{
    if (blocksize_bytes.raw() == 0) {
        throw StatusException(
            ErrorCode::INVALID_ARGUMENT,
            "Blocked storage types require a non-zero blocksize"
        );
    }
    auto parameters = svs::data::BlockingParameters{.blocksize_bytes = blocksize_bytes};
    return Alloc(parameters);
}

// Storage types
template <typename T, typename Alloc>
using SimpleDatasetType = svs::data::SimpleData<T, svs::Dynamic, Alloc>;

template <typename T, typename Alloc>
using SQDatasetType = svs::quantization::scalar::SQDataset<T, svs::Dynamic, Alloc>;

// Storage type mapping

// Unsupported storage type defined as unique type to cause runtime error if used
template <typename Alloc> struct UnsupportedStorageType {
    using allocator_type = Alloc; // Dummy allocator type to satisfy template requirements
};

template <StorageKind Kind, typename Alloc> struct StorageType {
    using allocator_type = Alloc;
    using type = UnsupportedStorageType<Alloc>;
};
template <StorageKind Kind, typename Alloc>
using StorageType_t = typename StorageType<Kind, Alloc>::type;

template <typename Alloc> struct StorageType<StorageKind::FP32, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<float, Alloc>;
    using type = SimpleDatasetType<float, allocator_type>;
};
template <typename Alloc> struct StorageType<StorageKind::FP16, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<svs::Float16, Alloc>;
    using type = SimpleDatasetType<svs::Float16, allocator_type>;
};
template <typename Alloc> struct StorageType<StorageKind::SQI8, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<std::int8_t, Alloc>;
    using type = SQDatasetType<std::int8_t, allocator_type>;
};

// Storage factory
template <typename T> struct StorageFactory;

// Unsupported storage type factory returning runtime error when attempted to be used.
// Return type defined to simple to allow substitution in templates.
template <typename Alloc> struct StorageFactory<UnsupportedStorageType<Alloc>> {
    using StorageType =
        SimpleDatasetType<float, rebind_extracted_allocator_t<float, Alloc>>;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& SVS_UNUSED(data),
        Pool& SVS_UNUSED(pool),
        const typename StorageType::allocator_type& SVS_UNUSED(alloc) = {}
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

template <typename T, size_t Extent, typename Alloc>
struct StorageFactory<svs::data::SimpleData<T, Extent, Alloc>> {
    using StorageType = svs::data::SimpleData<T, Extent, Alloc>;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        const Alloc& alloc = {}
    ) {
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
template <typename T, size_t Extent, typename Alloc>
struct StorageFactory<svs::quantization::scalar::SQDataset<T, Extent, Alloc>> {
    using StorageType = svs::quantization::scalar::SQDataset<T, Extent, Alloc>;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        const Alloc& alloc = {}
    ) {
        return StorageType::compress(data, pool, alloc);
    }

    template <typename... Args>
    static StorageType load(const std::filesystem::path& path, Args&&... args) {
        return svs::lib::load_from_disk<StorageType>(path, SVS_FWD(args)...);
    }
};

// LVQ Storage support
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
using Sequential = svs::quantization::lvq::Sequential;
using Turbo16x8 = svs::quantization::lvq::Turbo<16, 8>;
template <size_t Primary, size_t Residual>
using AutoStrategy = std::conditional_t<(Primary == 4), Turbo16x8, Sequential>;

template <
    size_t Primary,
    size_t Residual,
    typename Alloc,
    size_t Extent = svs::Dynamic,
    svs::quantization::lvq::LVQPackingStrategy Strategy = AutoStrategy<Primary, Residual>>
using LVQDatasetType =
    svs::quantization::lvq::LVQDataset<Primary, Residual, Extent, Strategy, Alloc>;

template <typename Alloc> struct StorageType<StorageKind::LVQ4x0, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<std::byte, Alloc>;
    using type = LVQDatasetType<4, 0, allocator_type>;
};
template <typename Alloc> struct StorageType<StorageKind::LVQ8x0, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<std::byte, Alloc>;
    using type = LVQDatasetType<8, 0, allocator_type>;
};
template <typename Alloc> struct StorageType<StorageKind::LVQ4x4, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<std::byte, Alloc>;
    using type = LVQDatasetType<4, 4, allocator_type>;
};
template <typename Alloc> struct StorageType<StorageKind::LVQ4x8, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<std::byte, Alloc>;
    using type = LVQDatasetType<4, 8, allocator_type>;
};

template <svs::quantization::lvq::IsLVQDataset LVQStorageType>
struct StorageFactory<LVQStorageType> {
    using StorageType = LVQStorageType;
    using Alloc = typename StorageType::allocator_type;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        const Alloc& alloc = {}
    ) {
        return StorageType::compress(data, pool, 0, alloc);
    }

    template <typename... Args>
    static StorageType load(const std::filesystem::path& path, Args&&... args) {
        return svs::lib::load_from_disk<StorageType>(path, SVS_FWD(args)...);
    }
};

// LeanVec Storage support
template <
    size_t I1,
    size_t I2,
    typename Alloc,
    size_t LeanVecDims = svs::Dynamic,
    size_t Extent = svs::Dynamic>
using LeanDatasetType = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<I1>,
    svs::leanvec::UsingLVQ<I2>,
    LeanVecDims,
    Extent,
    Alloc>;

template <typename Alloc> struct StorageType<StorageKind::LeanVec4x4, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<std::byte, Alloc>;
    using type = LeanDatasetType<4, 4, allocator_type>;
};
template <typename Alloc> struct StorageType<StorageKind::LeanVec4x8, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<std::byte, Alloc>;
    using type = LeanDatasetType<4, 8, allocator_type>;
};
template <typename Alloc> struct StorageType<StorageKind::LeanVec8x8, Alloc> {
    using allocator_type = rebind_extracted_allocator_t<std::byte, Alloc>;
    using type = LeanDatasetType<8, 8, allocator_type>;
};

template <svs::leanvec::IsLeanDataset LeanVecStorageType>
struct StorageFactory<LeanVecStorageType> {
    using StorageType = LeanVecStorageType;
    using Alloc = typename StorageType::allocator_type;

    template <svs::threads::ThreadPool Pool>
    static StorageType init(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        const Alloc& alloc = {},
        size_t leanvec_d = 0,
        std::optional<svs::leanvec::LeanVecMatrices<svs::Dynamic>> matrices = std::nullopt
    ) {
        if (leanvec_d == 0) {
            leanvec_d = (data.dimensions() + 1) / 2;
        }
        return LeanVecStorageType::reduce(
            data, std::move(matrices), pool, 0, svs::lib::MaybeStatic{leanvec_d}, alloc
        );
    }

    template <typename... Args>
    static StorageType load(const std::filesystem::path& path, Args&&... args) {
        return svs::lib::load_from_disk<StorageType>(path, SVS_FWD(args)...);
    }
};
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

template <StorageKind Kind, typename Alloc, typename... Args>
auto make_storage(StorageType<Kind, Alloc> SVS_UNUSED(tag), Args&&... args) {
    return StorageFactory<StorageType_t<Kind, Alloc>>::init(std::forward<Args>(args)...);
}

template <StorageKind Kind, typename Alloc, typename... Args>
auto load_storage(StorageType<Kind, Alloc> SVS_UNUSED(tag), Args&&... args) {
    return StorageFactory<StorageType_t<Kind, Alloc>>::load(std::forward<Args>(args)...);
}

template <typename Alloc, typename F, typename... Args>
auto dispatch_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    if (!is_supported_storage_kind(kind)) {
        throw StatusException(
            ErrorCode::NOT_IMPLEMENTED, "Requested storage kind is not supported by CPU"
        );
    }
#define SVS_DISPATCH_STORAGE_KIND(Kind) \
    case StorageKind::Kind:             \
        return f(StorageType<StorageKind::Kind, Alloc>{}, std::forward<Args>(args)...)

    switch (kind) {
        SVS_DISPATCH_STORAGE_KIND(FP32);
        SVS_DISPATCH_STORAGE_KIND(FP16);
        SVS_DISPATCH_STORAGE_KIND(SQI8);
        SVS_DISPATCH_STORAGE_KIND(LVQ4x0);
        SVS_DISPATCH_STORAGE_KIND(LVQ8x0);
        SVS_DISPATCH_STORAGE_KIND(LVQ4x4);
        SVS_DISPATCH_STORAGE_KIND(LVQ4x8);
        SVS_DISPATCH_STORAGE_KIND(LeanVec4x4);
        SVS_DISPATCH_STORAGE_KIND(LeanVec4x8);
        SVS_DISPATCH_STORAGE_KIND(LeanVec8x8);
        default:
            throw StatusException(
                ErrorCode::INVALID_ARGUMENT, "Unknown or unsupported SVS storage kind"
            );
    }

#undef SVS_DISPATCH_STORAGE_KIND
}
} // namespace storage

// Predict how many more items need to be processed to reach the goal,
// based on the observed hit rate so far.
// If no hits yet, returns `hint` unchanged.
// Result is capped at `max_batch_size` (e.g., number of vectors in the index).
inline size_t predict_further_processing(
    size_t processed, size_t hits, size_t goal, size_t hint, size_t max_batch_size
) {
    if (hits == 0 || hits >= goal) {
        return std::min(hint, max_batch_size);
    }
    float batch_size = static_cast<float>(goal - hits) * processed / hits;
    return std::min(std::max(static_cast<size_t>(batch_size), size_t{1}), max_batch_size);
}

// Check if the filtered search should stop early based on the observed hit rate.
// Returns true if the hit rate is below the threshold, meaning the caller should
// give up and let the caller fall back to exact search.
inline bool
should_stop_filtered_search(size_t total_checked, size_t found, float filter_stop) {
    if (filter_stop <= 0 || total_checked == 0 || found == 0) {
        return false;
    }
    float hit_rate = static_cast<float>(found) / total_checked;
    return hit_rate < filter_stop;
}

inline svs::threads::ThreadPoolHandle default_threadpool() {
    return svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads())
    );
}
} // namespace svs::runtime
