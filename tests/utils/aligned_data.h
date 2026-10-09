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

#include "svs/core/data/simple.h"
#include "svs/lib/saveload.h"

#include <istream>

namespace svs_test {

// SimpleData with explicit alignment. Not used in in open-source SVS, but explicitly
// supported by (dynamic) Vamana index
template <typename Allocator>
class AlignedData : public svs::data::SimpleData<float, svs::Dynamic, Allocator> {
  public:
    using base_type = svs::data::SimpleData<float, svs::Dynamic, Allocator>;
    using allocator_type = Allocator;

    // Set by the most recently invoked loader below; the orchestrator type-erases the
    // dataset, so tests reset this to 0 and read it back here rather than from the index.
    inline static size_t last_alignment = 0;

    explicit AlignedData(base_type&& base)
        : base_type(std::move(base)) {}

    static AlignedData load(
        const svs::lib::ContextFreeLoadTable& table,
        std::istream& is,
        size_t alignment,
        const allocator_type& allocator
    ) {
        last_alignment = alignment;
        return AlignedData(base_type::load(table, is, allocator));
    }

    static AlignedData load(
        const svs::lib::LoadTable& table, size_t alignment, const allocator_type& allocator
    ) {
        last_alignment = alignment;
        return AlignedData(base_type::load(table, allocator));
    }
};

} // namespace svs_test
