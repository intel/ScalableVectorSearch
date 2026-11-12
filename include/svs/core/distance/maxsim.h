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
#include "svs/core/distance/inner_product.h"

// stl
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <string_view>

namespace svs::distance {

/// Sum over query tokens of the max inner product over document tokens. Asymmetric;
/// larger is better; inputs are not normalised here.
struct MaxSim {
  public:
    using compare = std::greater<>;

    static constexpr bool implicit_broadcast = true;
    static constexpr std::string_view name = "maxsim";
};

inline constexpr bool operator==(const MaxSim&, const MaxSim&) { return true; }

/// Empty document -> -infinity, strictly worse than any real score. Empty query and
/// dimension mismatch are caller errors, asserted only: a mismatch reads out of bounds.
template <svs::data::VectorSet Q, svs::data::VectorSet D>
float compute(MaxSim /*unused*/, const Q& query, const D& document) {
    size_t n_query = query.cardinality();
    assert(n_query > 0);
    size_t n_doc = document.cardinality();
    if (n_doc == 0) {
        return -std::numeric_limits<float>::infinity();
    }
    assert(query.dimensions() == document.dimensions());

    double total = 0.0;
    for (size_t i = 0; i < n_query; ++i) {
        auto q_token = query.token(i);
        float best = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < n_doc; ++j) {
            auto d_token = document.token(j);
            float ip = IP::compute(q_token.data(), d_token.data(), q_token.size());
            best = std::max(best, ip);
        }
        total += best;
    }
    return static_cast<float>(total);
}

/// `get_datum(id)` must return a type modeling `svs::data::VectorSet`; `scores` must have
/// the same length as `ids`.
template <svs::data::VectorSet Q, typename GetDatum, std::integral Id>
void maxsim_batch(
    const Q& query, GetDatum&& get_datum, std::span<const Id> ids, std::span<float> scores
) {
    assert(ids.size() == scores.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        scores[i] = compute(MaxSim{}, query, get_datum(ids[i]));
    }
}

} // namespace svs::distance
