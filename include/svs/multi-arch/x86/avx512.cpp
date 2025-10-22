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

/**
 * @file avx512.cpp
 * @brief AVX-512 specific SIMD operation instantiations
 *
 * This compilation unit is built with corresponding compiler flags that enable
 * AVX-512 instruction generation. It contains explicit instantiations of SIMD operation
 * structs to force the compiler to generate optimized code using AVX-512 intrinsics.
 *
 * The SIMD ops contain all AVX-specific code and are defined in the distance headers
 * within #if SVS_AVX512_F guards. By instantiating them here with proper compiler flags,
 * we ensure optimized machine code is generated and linked into the library.
 */

#if defined(__x86_64__)

// Include distance headers to get SIMD op definitions
#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"

namespace svs::distance {

/////
///// Inner Product SIMD Ops
/////

// Instantiate the primary floating-point SIMD op for AVX-512
template struct IPFloatOp<16, AVX_AVAILABILITY::AVX512>;

// Instantiate VNNI integer operation for AVX-512
template struct IPVNNIOp<int16_t, 32, AVX_AVAILABILITY::AVX512>;

/////
///// L2 (Euclidean) SIMD Ops
/////

// Instantiate the primary floating-point SIMD op for AVX-512
template struct L2FloatOp<16, AVX_AVAILABILITY::AVX512>;

// Instantiate VNNI integer operation for AVX-512
template struct L2VNNIOp<int16_t, 32, AVX_AVAILABILITY::AVX512>;

/////
///// Cosine Similarity SIMD Ops
/////

// Instantiate the floating-point SIMD op for AVX-512
template struct CosineFloatOp<16, AVX_AVAILABILITY::AVX512>;

} // namespace svs::distance

#endif // defined(__x86_64__)
