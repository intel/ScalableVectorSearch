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

#include <svs/concepts/data.h>

namespace svs::c_runtime {

struct Storage {
    svs_storage_kind kind;
    Storage(svs_storage_kind kind)
        : kind(kind) {}
    virtual ~Storage() = default;
};

struct StorageSimple : public Storage {
    svs_data_type_t data_type;

    StorageSimple(svs_data_type_t data_type)
        : Storage{SVS_STORAGE_KIND_SIMPLE}
        , data_type(data_type) {}
};

struct StorageLeanVec : public Storage {
    size_t lenavec_dims;
    svs_data_type_t primary_type;
    svs_data_type_t secondary_type;

    StorageLeanVec(size_t lenavec_dims, svs_data_type_t primary, svs_data_type_t secondary)
        : Storage{SVS_STORAGE_KIND_LEANVEC}
        , lenavec_dims(lenavec_dims)
        , primary_type(primary)
        , secondary_type(secondary) {}
};

} // namespace svs::c_runtime
