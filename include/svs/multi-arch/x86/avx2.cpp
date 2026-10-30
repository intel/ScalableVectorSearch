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
DISTANCE_L2_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

DISTANCE_IP_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

DISTANCE_CS_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

template struct QEmdImpl<AVX_AVAILABILITY::AVX2>;
template struct QChamferImpl<AVX_AVAILABILITY::AVX2>;
template struct QChamferExactImpl<AVX_AVAILABILITY::AVX2>;

// See avx512.cpp's QChamferExactImpl<AVX512>::build_table for the accumulation-order and
// rounding rationale; this is the same scheme at AVX2's 8-wide registers.
void QChamferExactImpl<AVX_AVAILABILITY::AVX2>::build_table(
    const float* centroids,
    size_t fine_centers,
    size_t dim,
    const float* query_tokens,
    size_t n_query,
    float* table
) {
    constexpr size_t vl = 8;
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
            __m256 acc = _mm256_setzero_ps();
            for (size_t d = 0; d < dim; ++d) {
                __m256 cval = _mm256_set1_ps(c[d]);
                __m256 qv = _mm256_loadu_ps(qt.data() + d * n_query + ci);
                acc = _mm256_fmadd_ps(cval, qv, acc);
            }
            _mm256_storeu_ps(row + ci, acc);
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

float QChamferExactImpl<AVX_AVAILABILITY::AVX2>::compute_decoded(
    const float* table, size_t n_query, const uint32_t* doc_ids, size_t n_doc
) {
    return gem_detail::qchamfer_exact_reduce(table, n_query, doc_ids, n_doc);
}

} // namespace svs::distance

#endif
