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

#pragma once

// New approach: Instantiate SIMD Ops instead of distance implementations
// This eliminates the N parameter from instantiations since SIMD ops work with fixed widths

// Helper to get SIMD width from AVX availability
// AVX512 uses width 16, AVX2 uses width 8
#define SIMD_WIDTH_FOR_AVX512 16
#define SIMD_WIDTH_FOR_AVX2 8

// Macros for L2 SIMD ops
#define SIMD_L2_OPS_HELPER(SPEC, WIDTH, AVX) \
    SPEC struct L2FloatOp<WIDTH, AVX>;

#define SIMD_L2_OPS_INSTANTIATE_AVX512(SPEC) \
    SIMD_L2_OPS_HELPER(SPEC, SIMD_WIDTH_FOR_AVX512, AVX_AVAILABILITY::AVX512)

#define SIMD_L2_OPS_INSTANTIATE_AVX2(SPEC) \
    SIMD_L2_OPS_HELPER(SPEC, SIMD_WIDTH_FOR_AVX2, AVX_AVAILABILITY::AVX2)

// Macros for Inner Product SIMD ops
#define SIMD_IP_OPS_HELPER(SPEC, WIDTH, AVX) \
    SPEC struct IPFloatOp<WIDTH, AVX>;

#define SIMD_IP_OPS_INSTANTIATE_AVX512(SPEC) \
    SIMD_IP_OPS_HELPER(SPEC, SIMD_WIDTH_FOR_AVX512, AVX_AVAILABILITY::AVX512)

#define SIMD_IP_OPS_INSTANTIATE_AVX2(SPEC) \
    SIMD_IP_OPS_HELPER(SPEC, SIMD_WIDTH_FOR_AVX2, AVX_AVAILABILITY::AVX2)

// Macros for Cosine Similarity SIMD ops
#define SIMD_COSINE_OPS_HELPER(SPEC, WIDTH, AVX) \
    SPEC struct CosineFloatOp<WIDTH, AVX>;

#define SIMD_COSINE_OPS_INSTANTIATE_AVX512(SPEC) \
    SIMD_COSINE_OPS_HELPER(SPEC, SIMD_WIDTH_FOR_AVX512, AVX_AVAILABILITY::AVX512)

#define SIMD_COSINE_OPS_INSTANTIATE_AVX2(SPEC) \
    SIMD_COSINE_OPS_HELPER(SPEC, SIMD_WIDTH_FOR_AVX2, AVX_AVAILABILITY::AVX2)
