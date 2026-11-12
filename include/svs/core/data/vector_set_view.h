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

// svs
#include "svs/concepts/vector_set.h"
#include "svs/lib/exception.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"
#include "svs/lib/prefetch.h"

// stl
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace svs {
namespace data {

///
/// @brief Non-owning view of one vector set: a row-major, contiguous run of token
/// vectors sharing a dimensionality.
///
template <typename T, size_t Extent = Dynamic> class ConstVectorSet {
  public:
    using element_type = T;

    ConstVectorSet(std::span<const T> data, size_t cardinality, size_t dimensions)
        : data_{data}
        , cardinality_{cardinality}
        , dimensions_{dimensions} {}

    const T* data() const { return data_.data(); }
    size_t cardinality() const { return cardinality_; }
    constexpr size_t dimensions() const {
        if constexpr (Extent != Dynamic) {
            return Extent;
        } else {
            return dimensions_;
        }
    }
    bool empty() const { return cardinality_ == 0; }

    std::span<const T, Extent> token(size_t i) const {
        assert(i < cardinality_);
        const T* base = data_.data() + i * dimensions();
        if constexpr (Extent != Dynamic) {
            return std::span<const T, Extent>(base, Extent);
        } else {
            return std::span<const T, Extent>(base, dimensions_);
        }
    }

  private:
    std::span<const T> data_;
    size_t cardinality_;
    size_t dimensions_;
};

static_assert(VectorSet<ConstVectorSet<float>>);

///
/// @brief Non-owning CSR-style view of a dataset whose elements are vector sets.
///
/// ``offsets`` counts token vectors (not elements), has length ``size() + 1``, is
/// non-decreasing, and ``offsets[0] == 0``; ``offsets[i + 1] - offsets[i]`` is the
/// cardinality of set ``i``.
///
template <typename T, size_t Extent = Dynamic> class ConstVectorSetView {
  public:
    using element_type = T;
    using value_type = ConstVectorSet<T, Extent>;
    using const_value_type = ConstVectorSet<T, Extent>;

    ConstVectorSetView(
        std::span<const T> tokens, std::span<const uint64_t> offsets, size_t dimensions
    )
        : tokens_{tokens}
        , offsets_{offsets}
        , dimensions_{dimensions} {
        if (offsets_.empty()) {
            throw ANNEXCEPTION("Vector set offsets must be non-empty, got an empty span");
        }
        if (offsets_.front() != 0) {
            throw ANNEXCEPTION(
                "Vector set offsets must start at 0, got {}", offsets_.front()
            );
        }
        for (size_t i = 1; i < offsets_.size(); ++i) {
            if (offsets_[i] < offsets_[i - 1]) {
                throw ANNEXCEPTION(
                    "Vector set offsets must be non-decreasing, got {} after {}",
                    offsets_[i],
                    offsets_[i - 1]
                );
            }
        }
        size_t total_tokens = lib::narrow<size_t>(offsets_.back());
        if (dimensions_ != 0 &&
            total_tokens > std::numeric_limits<size_t>::max() / dimensions_) {
            throw ANNEXCEPTION(
                "Token count {} times dimensions {} overflows size_t!",
                total_tokens,
                dimensions_
            );
        }
        size_t expected_tokens = total_tokens * dimensions_;
        if (expected_tokens != tokens_.size()) {
            throw ANNEXCEPTION(
                "Vector set offsets imply {} token elements but got {} token elements",
                expected_tokens,
                tokens_.size()
            );
        }
        if constexpr (Extent != Dynamic) {
            if (dimensions_ != Extent) {
                throw ANNEXCEPTION(
                    "Vector set view has static extent {} but got dimension {}",
                    Extent,
                    dimensions_
                );
            }
        }
    }

    size_t size() const { return offsets_.size() - 1; }
    constexpr size_t dimensions() const {
        if constexpr (Extent != Dynamic) {
            return Extent;
        } else {
            return dimensions_;
        }
    }
    size_t cardinality(size_t i) const {
        assert(i < size());
        return lib::narrow<size_t>(offsets_[i + 1] - offsets_[i]);
    }
    size_t total_tokens() const { return lib::narrow<size_t>(offsets_.back()); }

    const_value_type get_datum(size_t i) const {
        size_t set_cardinality = cardinality(i);
        size_t begin = lib::narrow<size_t>(offsets_[i]) * dimensions();
        size_t count = set_cardinality * dimensions();
        return ConstVectorSet<T, Extent>(
            tokens_.subspan(begin, count), set_cardinality, dimensions()
        );
    }

    // Only the first token vector of the set is close enough to be worth prefetching;
    // a large set still has to stream the rest from memory.
    void prefetch(size_t i) const {
        if (cardinality(i) == 0) {
            return;
        }
        size_t begin = lib::narrow<size_t>(offsets_[i]) * dimensions();
        lib::prefetch(tokens_.subspan(begin, dimensions()));
    }

  private:
    std::span<const T> tokens_;
    std::span<const uint64_t> offsets_;
    size_t dimensions_;
};

static_assert(ImmutableVectorSetDataset<ConstVectorSetView<float>>);

} // namespace data
} // namespace svs
