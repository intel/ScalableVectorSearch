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
 * @file avx2.cpp
 * @brief AVX2 specific SIMD operation instantiations
 *
 * This compilation unit is built with `-march=haswell` compiler flags to enable
 * AVX2 instruction generation. It contains explicit instantiations of SIMD operation
 * structs to force the compiler to generate optimized code using AVX2 intrinsics.
 *
 * IMPORTANT: We instantiate SIMD ops (IPFloatOp, L2FloatOp, CosineFloatOp), NOT
 * distance implementations (*Impl). This eliminates the need to instantiate for all
 * combinations of dimensionality N, which would create 9× more instantiations.
 *
 * The SIMD ops contain all AVX-specific code and are defined in the distance headers
 * within #if !SVS_AVX512_F && SVS_AVX2 guards. By instantiating them here with proper
 * compiler flags, we ensure optimized machine code is generated and linked into the library.
 */

#if defined(__x86_64__)

// Include distance headers to get SIMD op definitions
#include "svs/core/distance/inner_product.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/cosine.h"

namespace svs::distance {

/////
///// Inner Product SIMD Ops
/////

// Instantiate the primary floating-point SIMD op for AVX2
template struct IPFloatOp<8, AVX_AVAILABILITY::AVX2>;

/////
///// L2 (Euclidean) SIMD Ops
/////

// Instantiate the primary floating-point SIMD op for AVX2
template struct L2FloatOp<8, AVX_AVAILABILITY::AVX2>;

/////
///// Cosine Similarity SIMD Ops
/////

// Instantiate the floating-point SIMD op for AVX2
template struct CosineFloatOp<8, AVX_AVAILABILITY::AVX2>;

} // namespace svs::distance

#endif // defined(__x86_64__)
