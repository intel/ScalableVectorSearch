/*
 * Copyright 2023 Intel Corporation
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

#include "svs/index/vamana/dynamic_search_buffer.h"

namespace svs::index::vamana::concurrent {

///
/// A search buffer that allows entries to be predicated out.
/// The search buffer will still navigate through those entries, but it won't include
/// them in its final result.
///
template <typename Idx, typename Cmp = std::less<>>
class MutableBuffer : public vamana::MutableBuffer<Idx, Cmp> {
    using base_type = vamana::MutableBuffer<Idx, Cmp>;

  public:
    using base_type::base_type;

    MutableBuffer() = default;

    /// Copy the portions of the MutableBuffer that matter for the purposes of scratch
    /// space.
    ///
    /// Preserves the sizes of various containers but not necessarily the values.
    MutableBuffer shallow_copy() const {
        // We don't care about the contents of the buffer - just its size.
        // Therefore, we can construct a new buffer from scratch.
        return MutableBuffer{
            base_type::config(), this->compare_, base_type::visited_set_enabled()};
    }
};

} // namespace svs::index::vamana::concurrent
