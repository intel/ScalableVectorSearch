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

#include <cstddef>

#include <svs/lib/exception.h>
#include <svs/lib/misc.h>

namespace svs::runtime::v0 {

class IndexBlockSize {
    constexpr static size_t kMaxBlockSizeExp = 30; // 1GB
    constexpr static size_t kMinBlockSizeExp = 12; // 4KB

    svs::lib::PowerOfTwo blocksize_bytes_;

  public:
    explicit IndexBlockSize(size_t blocksize_exp) {
        if (blocksize_exp > kMaxBlockSizeExp) {
            throw ANNEXCEPTION("Blocksize is too large!");
        } else if (blocksize_exp < kMinBlockSizeExp) {
            throw ANNEXCEPTION("Blocksize is too small!");
        }

        blocksize_bytes_ = svs::lib::PowerOfTwo(blocksize_exp);
    }

    svs::lib::PowerOfTwo BlockSizeBytes() const { return blocksize_bytes_; }
};

} // namespace svs::runtime::v0