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
#include "svs/lib/avx_detection.h"
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
/// QChamferExactQuery in gem.h): `fix_argument(query)` packs the query's tokens once
/// into a `PackedOperand`; every `compute(document)` call that follows scores one
/// document against that packed query through `InnerProductMatrixImpl::column_max`, with
/// no allocation beyond the per-query packing and a per-document maxima scratch that
/// only grows. `must_fix_argument` is set because `compute` is meaningless without a
/// prior `fix_argument`; `implicit_broadcast` is false because the packed query is
/// ordinary (non-shared) state, so each search thread's shallow_copy of this functor
/// must own its own `PackedOperand` rather than sharing one.
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
    }

    /// `query` is unused: `fix_argument` already consumed it into `packed_`, and every
    /// `compute` call in a search shares that one query's packing, matching the two-arg
    /// overload `distance::compute` dispatches to for a `must_fix_argument` functor
    /// (see QChamferExactQuery::compute in gem.h for the same shape).
    template <typename Query, svs::data::ContiguousVectorSet D>
        requires std::same_as<typename D::element_type, float> ||
                 std::same_as<typename D::element_type, svs::Float16>
    float compute(const Query& /*query*/, const D& document) const {
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

        thread_local std::vector<float> maxima;
        maxima.resize(packed_.n());
        column_max_dispatch(document.data(), n_doc, maxima.data());

        double total = 0.0;
        for (float v : maxima) {
            total += static_cast<double>(v);
        }
        return static_cast<float>(total);
    }

  private:
    // Dispatched once per document, mirroring InnerProductMatrix::select_isa(): the
    // query-packing and document-scoring ISA choices are independent calls, so this is
    // not shared with InnerProductMatrix itself (which has no document-shaped operand).
    template <typename T>
    void column_max_dispatch(const T* a, size_t n_doc, float* out) const {
        switch (InnerProductMatrix::select_isa()) {
            case AVX_AVAILABILITY::AVX512:
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
                    a, n_doc, packed_, out
                );
                return;
            case AVX_AVAILABILITY::AVX2:
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::column_max(
                    a, n_doc, packed_, out
                );
                return;
            case AVX_AVAILABILITY::NONE:
                InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::column_max(
                    a, n_doc, packed_, out
                );
                return;
        }
        // Exhaustive switch over AVX_AVAILABILITY's enumerators, no default: see
        // InnerProductMatrix::compute for why this stays a compile error, not a
        // silent fallback, if a 4th enumerator is ever added.
        __builtin_unreachable();
    }

    PackedOperand packed_;
};

inline constexpr bool operator==(const MaxSim&, const MaxSim&) { return true; }

} // namespace svs::distance
