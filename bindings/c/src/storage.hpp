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

#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/data/simple.h>
#include <svs/extensions/vamana/leanvec.h>
#include <svs/extensions/vamana/lvq.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/leanvec/impl/leanvec_impl.h>
#include <svs/lib/datatype.h>
#include <svs/lib/dispatcher.h>
#include <svs/lib/float16.h>
#include <svs/lib/type_traits.h>
#include <svs/quantization/lvq/impl/lvq_impl.h>
#include <svs/quantization/scalar/scalar.h>

namespace svs {
namespace c_runtime {

struct Storage {
    svs_storage_kind kind;
    Storage(svs_storage_kind kind)
        : kind(kind) {}
    virtual ~Storage() = default;
};

struct StorageSimple : public Storage {
    svs::DataType data_type;

    StorageSimple(svs_data_type_t dt)
        : Storage{SVS_STORAGE_KIND_SIMPLE}
        , data_type(to_data_type(dt)) {
        if (dt != SVS_DATA_TYPE_FLOAT32 && dt != SVS_DATA_TYPE_FLOAT16) {
            throw std::invalid_argument(
                "Simple storage only supports float32 and float16 data types"
            );
        }
    }
};

struct StorageLeanVec : public Storage {
    size_t lenavec_dims;
    size_t primary_bits;
    size_t secondary_bits;

    StorageLeanVec(size_t lenavec_dims, svs_data_type_t primary, svs_data_type_t secondary)
        : Storage{SVS_STORAGE_KIND_LEANVEC}
        , lenavec_dims(lenavec_dims)
        , primary_bits(to_bits_number(primary))
        , secondary_bits(to_bits_number(secondary)) {}

    static size_t to_bits_number(svs_data_type_t data_type) {
        switch (data_type) {
            case SVS_DATA_TYPE_INT4:
            case SVS_DATA_TYPE_UINT4:
                return 4;
            case SVS_DATA_TYPE_INT8:
            case SVS_DATA_TYPE_UINT8:
                return 8;
            case SVS_DATA_TYPE_VOID:
                return 0;
            default:
                throw std::invalid_argument("Unsupported data type for LeanVec storage");
        }
    }
};

struct StorageLVQ : public Storage {
    size_t primary_bits;
    size_t residual_bits;

    StorageLVQ(svs_data_type_t primary, svs_data_type_t residual)
        : Storage{SVS_STORAGE_KIND_LVQ}
        , primary_bits(to_bits_number(primary))
        , residual_bits(to_bits_number(residual)) {}

    static size_t to_bits_number(svs_data_type_t data_type) {
        switch (data_type) {
            case SVS_DATA_TYPE_INT4:
            case SVS_DATA_TYPE_UINT4:
                return 4;
            case SVS_DATA_TYPE_INT8:
            case SVS_DATA_TYPE_UINT8:
                return 8;
            case SVS_DATA_TYPE_VOID:
                return 0;
            default:
                throw std::invalid_argument("Unsupported data type for LVQ storage");
        }
    }
};

struct StorageSQ : public Storage {
    svs::DataType data_type;

    StorageSQ(svs_data_type_t dt)
        : Storage{SVS_STORAGE_KIND_SQ}
        , data_type(to_data_type(dt)) {
        if (dt != SVS_DATA_TYPE_UINT8 && dt != SVS_DATA_TYPE_INT8) {
            throw std::invalid_argument("Scalar quantization only supports 8-bit data types"
            );
        }
    }
};

} // namespace c_runtime

template <Arithmetic T> class SimpleDataBuilder {
  public:
    SimpleDataBuilder() {}

    using SimpleDataType =
        svs::data::SimpleData<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

    template <Arithmetic U>
    SimpleDataType build(
        svs::data::ConstSimpleDataView<U> view,
        svs::threads::ThreadPoolHandle& SVS_UNUSED(pool)
    ) {
        auto data = SimpleDataType(view.size(), view.dimensions());
        svs::data::copy(view, data);
        return data;
    }
};

template <Arithmetic T>
struct lib::DispatchConverter<const c_runtime::Storage*, SimpleDataBuilder<T>> {
    using From = const svs::c_runtime::Storage*;
    using To = SimpleDataBuilder<T>;

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

    static To convert(From from) { return To{}; }
};

template <size_t I1, size_t I2> class LeanVecDataBuilder {
    size_t leanvec_dims_;

  public:
    LeanVecDataBuilder(size_t leanvec_dims)
        : leanvec_dims_(leanvec_dims) {}

    using LeanDatasetType = svs::leanvec::LeanDataset<
        svs::leanvec::UsingLVQ<I1>,
        svs::leanvec::UsingLVQ<I2>,
        svs::Dynamic,
        svs::Dynamic,
        svs::data::Blocked<svs::lib::Allocator<std::byte>>>;

    template <Arithmetic T>
    LeanDatasetType
    build(svs::data::ConstSimpleDataView<T> view, svs::threads::ThreadPoolHandle& pool) {
        return LeanDatasetType::reduce(
            view, std::nullopt, pool, 0, svs::lib::MaybeStatic{leanvec_dims_}
        );
    }
};

template <size_t I1, size_t I2>
struct lib::DispatchConverter<const c_runtime::Storage*, LeanVecDataBuilder<I1, I2>> {
    using From = const svs::c_runtime::Storage*;
    using To = LeanVecDataBuilder<I1, I2>;

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
        return LeanVecDataBuilder<I1, I2>(leanvec->lenavec_dims);
    }
};

template <size_t PrimaryBits, size_t ResidualBits> class LVQDataBuilder {
  public:
    LVQDataBuilder() {}

    using LVQDatasetType = svs::quantization::lvq::LVQDataset<
        PrimaryBits,
        ResidualBits,
        svs::Dynamic,
        svs::quantization::lvq::Sequential,
        svs::data::Blocked<svs::lib::Allocator<std::byte>>>;

    template <Arithmetic T>
    LVQDatasetType
    build(svs::data::ConstSimpleDataView<T> view, svs::threads::ThreadPoolHandle& pool) {
        return LVQDatasetType::compress(view, pool, 0);
    }
};

template <size_t PrimaryBits, size_t ResidualBits>
struct lib::DispatchConverter<
    const c_runtime::Storage*,
    LVQDataBuilder<PrimaryBits, ResidualBits>> {
    using From = const svs::c_runtime::Storage*;
    using To = LVQDataBuilder<PrimaryBits, ResidualBits>;

    static int64_t match(From from) {
        if (from->kind == SVS_STORAGE_KIND_LVQ) {
            auto lvq = static_cast<const c_runtime::StorageLVQ*>(from);
            if (lvq->primary_bits == PrimaryBits && lvq->residual_bits == ResidualBits) {
                return svs::lib::perfect_match;
            }
        }
        return svs::lib::invalid_match;
    }

    static To convert(From from) { return To{}; }
};

template <Arithmetic T> class SQDataBuilder {
  public:
    SQDataBuilder() {}

    using SQDatasetType = svs::quantization::scalar::
        SQDataset<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

    template <Arithmetic U>
    SQDatasetType
    build(svs::data::ConstSimpleDataView<U> view, svs::threads::ThreadPoolHandle& pool) {
        return SQDatasetType::compress(view, pool);
    }
};

template <Arithmetic T>
struct lib::DispatchConverter<const c_runtime::Storage*, SQDataBuilder<T>> {
    using From = const svs::c_runtime::Storage*;
    using To = SQDataBuilder<T>;

    static int64_t match(From from) {
        if (from->kind == SVS_STORAGE_KIND_SQ) {
            auto sq = static_cast<const c_runtime::StorageSQ*>(from);
            if (sq->data_type == svs::datatype_v<T>) {
                return svs::lib::perfect_match;
            }
        }
        return svs::lib::invalid_match;
    }

    static To convert(From from) { return To{}; }
};

} // namespace svs
