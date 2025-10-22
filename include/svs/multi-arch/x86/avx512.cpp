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
#include "svs/core/distance/inner_product.h"

// Force compilation of SIMD ops by explicitly instantiating distance implementations
// These are compiled with -march=cascadelake, generating optimized AVX512 code
namespace svs::distance {

// Explicitly instantiate for a representative set of type combinations
// This ensures the AVX512 SIMD ops are actually compiled into the library
template struct IPImpl<64, float, float, AVX_AVAILABILITY::AVX512>;
template struct IPImpl<128, float, float, AVX_AVAILABILITY::AVX512>;
template struct IPImpl<64, int8_t, int8_t, AVX_AVAILABILITY::AVX512>;
template struct IPImpl<128, uint8_t, uint8_t, AVX_AVAILABILITY::AVX512>;

template struct L2Impl<64, float, float, AVX_AVAILABILITY::AVX512>;
template struct L2Impl<128, float, float, AVX_AVAILABILITY::AVX512>;
template struct L2Impl<64, int8_t, int8_t, AVX_AVAILABILITY::AVX512>;
template struct L2Impl<128, uint8_t, uint8_t, AVX_AVAILABILITY::AVX512>;

template struct CosineSimilarityImpl<64, float, float, AVX_AVAILABILITY::AVX512>;
template struct CosineSimilarityImpl<128, float, float, AVX_AVAILABILITY::AVX512>;

} // namespace svs::distance

#endif
