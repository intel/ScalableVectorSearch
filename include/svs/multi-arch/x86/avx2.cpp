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

///
/// @file avx2.cpp
/// @brief Explicit instantiations of distance implementations for AVX2
///
/// This file contains explicit template instantiations for distance computations
/// optimized for Intel(R) AVX2 instructions. It is compiled with compiler flags
/// targeting the Haswell microarchitecture (`-march=haswell`), which includes
/// AVX2, FMA, and related extensions.
///
/// ## Purpose
///
/// By explicitly instantiating templates in this compilation unit with AVX2
/// compiler flags, we enable runtime ISA dispatch: the library can detect at runtime
/// whether the CPU supports AVX2 (but not AVX-512) and call these optimized
/// implementations, while falling back to generic implementations if neither AVX-512
/// nor AVX2 are available.
///
/// ## Architecture
///
/// Each distance implementation (L2Impl, IPImpl, CosineSimilarityImpl) is a thin
/// wrapper around `generic_simd_op` which in turn uses SIMD operation structs like:
/// - `L2FloatOp<8>` - L2 distance using AVX2 floating-point operations (SIMD width 8)
/// - `IPFloatOp<8>` - Inner product using AVX2 floating-point operations
/// - `CosineFloatOp<8>` - Cosine similarity using AVX2 floating-point operations
///
/// Note: AVX2 uses SIMD width 8 (256-bit vectors) vs AVX-512's width 16 (512-bit vectors).
///
/// These SIMD ops are defined in the distance headers and contain the actual AVX2
/// intrinsics. The instantiations here ensure this AVX2 code is generated.
///
/// ## Dimensions Instantiated
///
/// We instantiate for the following dimensionalities:
/// - Fixed: 64, 96, 100, 128, 160, 200, 512, 768
/// - Dynamic: For runtime-determined dimensions
///
/// For each dimension, 16 type combinations are instantiated (4 element types × 4):
/// float, int8_t, uint8_t, Float16
///

#if defined(__x86_64__)
#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"

namespace svs::distance {

// TODO: connect with dim_supported_list

// ============================================================================
// L2 (Euclidean) Distance Instantiations
// ============================================================================
DISTANCE_L2_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

// ============================================================================
// Inner Product Instantiations
// ============================================================================
DISTANCE_IP_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

// ============================================================================
// Cosine Similarity Instantiations
// ============================================================================
DISTANCE_CS_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

} // namespace svs::distance

#endif
