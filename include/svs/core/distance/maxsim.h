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
#include "svs/core/distance/inner_product_matrix.h"
#include "svs/lib/exception.h"
#include "svs/lib/float16.h"

// stl
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>

namespace svs::distance {

/// Sum over query tokens of the max inner product over document tokens. Asymmetric;
/// larger is better; inputs are not normalised here.
///
/// Per-query state, following the library's fix_argument protocol (like
/// QChamferExactQuery in gem.h): `fix_argument(query)` packs the query's tokens once into
/// a `PackedOperand` and sizes a member maxima buffer to the query's cardinality; every
/// `compute(document)` call that follows scores one document against that packed query
/// through `InnerProductMatrix::column_max`, writing into the member buffer rather than a
/// thread_local one, so compute performs no thread_local lookup per document. That is
/// safe precisely because `implicit_broadcast` is false: the packed query and the maxima
/// buffer are both ordinary (non-shared) state, so each search thread's shallow_copy of
/// this functor owns its own copy of both rather than sharing either one.
/// `must_fix_argument` is set because `compute` is meaningless without a prior
/// `fix_argument`.
///
/// The query (fix_argument's argument) is a `svs::data::ContiguousVectorSet` of float32
/// tokens. The document (compute's second argument) is a `svs::data::ContiguousVectorSet`
/// of float32 or svs::Float16 tokens with the same dimensionality as the query; a
/// mismatch throws (checked once per document, which is negligible next to the kernel
/// it guards). An empty document scores -infinity, strictly worse than any real score.
class MaxSim {
  public:
    using compare = std::greater<>;

    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;
    static constexpr std::string_view name = "maxsim";

    /// Packs `query`'s tokens (float32, contiguous) into this object's `PackedOperand`,
    /// reusing its storage across calls. `query.cardinality() == 0` is a caller error,
    /// asserted only: a MaxSim score over zero query tokens is undefined, not merely
    /// small, so there is nothing for `compute` to usefully return.
    template <svs::data::ContiguousVectorSet S>
        requires std::same_as<typename S::element_type, float>
    void fix_argument(const S& query) {
        assert(query.cardinality() > 0);
        packed_.pack(query.data(), query.cardinality(), query.dimensions());
        maxima_.resize(packed_.n());
    }

    /// `query` is unused: `fix_argument` already consumed it into `packed_`, and every
    /// `compute` call in a search shares that one query's packing, matching the two-arg
    /// overload `distance::compute` dispatches to for a `must_fix_argument` functor
    /// (see QChamferExactQuery::compute in gem.h for the same shape). Non-const: writes
    /// into the member maxima buffer `fix_argument` sized.
    template <typename Query, svs::data::ContiguousVectorSet D>
        requires std::same_as<typename D::element_type, float> ||
                 std::same_as<typename D::element_type, svs::Float16>
    float compute(const Query& /*query*/, const D& document) {
        if (document.dimensions() != packed_.d()) {
            throw ANNEXCEPTION(
                "MaxSim::compute: document dimension {} does not match the fixed "
                "query's dimension {}",
                document.dimensions(),
                packed_.d()
            );
        }
        size_t n_doc = document.cardinality();
        if (n_doc == 0) {
            return -std::numeric_limits<float>::infinity();
        }

        InnerProductMatrix::column_max(document.data(), n_doc, packed_, maxima_.data());

        double total = 0.0;
        for (float v : maxima_) {
            total += static_cast<double>(v);
        }
        return static_cast<float>(total);
    }

  private:
    PackedOperand packed_;
    std::vector<float> maxima_;
};

inline constexpr bool operator==(const MaxSim&, const MaxSim&) { return true; }

} // namespace svs::distance
