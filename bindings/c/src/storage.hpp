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
#include "error.hpp"
#include "types_support.hpp"

#include <svs/lib/datatype.h>
#include <svs/lib/type_traits.h>

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
#include <svs/cpuid.h>
#endif

#include <filesystem>
#include <stdexcept>

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
        , secondary_bits(to_bits_number(secondary)) {
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
        if (!enabled()) {
            throw svs::c_runtime::unsupported_hw(
                "LeanVec storage is not supported on this hardware"
            );
        }
#else
        throw svs::c_runtime::not_implemented(
            "LeanVec storage is not implemented in this build"
        );
#endif
    }

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

    static bool enabled() {
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
        return svs::detail::intel_enabled();
#else
        return false;
#endif
    }
};

struct StorageLVQ : public Storage {
    size_t primary_bits;
    size_t residual_bits;

    StorageLVQ(svs_data_type_t primary, svs_data_type_t residual)
        : Storage{SVS_STORAGE_KIND_LVQ}
        , primary_bits(to_bits_number(primary))
        , residual_bits(to_bits_number(residual)) {
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
        if (!enabled()) {
            throw svs::c_runtime::unsupported_hw(
                "LVQ storage is not supported on this hardware"
            );
        }
#else
        throw svs::c_runtime::not_implemented("LVQ storage is not implemented in this build"
        );
#endif
    }

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

    static bool enabled() {
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
        return svs::detail::intel_enabled();
#else
        return false;
#endif
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
} // namespace svs
