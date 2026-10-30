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

#if defined(__x86_64__)
#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/gem.h"
#include "svs/core/distance/inner_product.h"

namespace svs::distance {

// TODO: connect with dim_supported_list
DISTANCE_L2_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512);

DISTANCE_IP_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512);

DISTANCE_CS_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512);

template struct QEmdImpl<AVX_AVAILABILITY::AVX512>;
template struct QChamferImpl<AVX_AVAILABILITY::AVX512>;
template struct QChamferExactImpl<AVX_AVAILABILITY::AVX512>;

// build_table vectorizes over query tokens, not over `dim`: per output lane i, the
// d=0..dim-1 FMA order matches the scalar version exactly, so only the single-rounding
// FMA (vs. separate mul+add) can move the result off the scalar value, well under the
// 1e-6 relative tolerance this table is checked against. Transposing query_tokens into
// [dim][n_query] turns each centroid row into contiguous per-token accumulators with no
// horizontal reduction, at the cost of a one-time O(dim * n_query) transpose (query is a
// handful of KiB; centroids are tens to hundreds of MiB, so the transpose is noise
// against the table build it replaces the scalar dot loop in).
void QChamferExactImpl<AVX_AVAILABILITY::AVX512>::build_table(
    const float* centroids,
    size_t fine_centers,
    size_t dim,
    const float* query_tokens,
    size_t n_query,
    float* table
) {
    constexpr size_t vl = 16;
    thread_local std::vector<float> qt;
    qt.resize(dim * n_query);
    for (size_t i = 0; i < n_query; ++i) {
        const float* t = query_tokens + i * dim;
        for (size_t d = 0; d < dim; ++d) {
            qt[d * n_query + i] = t[d];
        }
    }

    size_t vec_end = n_query - n_query % vl;
    for (size_t f = 0; f < fine_centers; ++f) {
        const float* c = centroids + f * dim;
        float* row = table + f * n_query;
        size_t ci = 0;
        for (; ci < vec_end; ci += vl) {
            __m512 acc = _mm512_setzero_ps();
            for (size_t d = 0; d < dim; ++d) {
                __m512 cval = _mm512_set1_ps(c[d]);
                __m512 qv = _mm512_loadu_ps(qt.data() + d * n_query + ci);
                acc = _mm512_fmadd_ps(cval, qv, acc);
            }
            _mm512_storeu_ps(row + ci, acc);
        }
        for (; ci < n_query; ++ci) {
            const float* t = query_tokens + ci * dim;
            float dot = 0.0F;
            for (size_t d = 0; d < dim; ++d) {
                dot += c[d] * t[d];
            }
            row[ci] = dot;
        }
    }
}

float QChamferExactImpl<AVX_AVAILABILITY::AVX512>::compute_decoded(
    const float* table, size_t n_query, const uint32_t* doc_ids, size_t n_doc
) {
    return gem_detail::qchamfer_exact_reduce(table, n_query, doc_ids, n_doc);
}

} // namespace svs::distance

#endif
