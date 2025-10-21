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

///
/// @file preprocessor.h
/// @brief Macros for explicit instantiation of distance implementations
///
/// This file contains macros to systematically generate explicit template instantiations
/// for distance implementations (L2Impl, IPImpl, CosineSimilarityImpl).
///
/// ## Why Explicit Instantiation?
///
/// The library supports runtime ISA dispatch - detecting AVX512/AVX2 support at runtime
/// and calling the appropriate optimized implementation. This requires:
/// 1. Separate compilation with architecture-specific compiler flags
/// 2. Explicit instantiation of templates in those compilation units
///
/// Without explicit instantiation, the templates would be instantiated inline wherever
/// used, which would prevent proper ISA-specific optimization.
///
/// ## Architecture
///
/// Distance implementations (e.g., `L2Impl`) are thin wrappers that call `generic_simd_op`
/// with a SIMD operation struct (e.g., `L2FloatOp<16>`). The SIMD ops contain the actual
/// AVX intrinsics. By explicitly instantiating the distance implementations in files
/// compiled with `-march=cascadelake` or `-march=haswell`, we ensure the AVX code is
/// generated with appropriate optimizations.
///
/// ## Type Combinations
///
/// Each macro instantiates 16 type combinations (4 element types × 4 element types):
/// - float, int8_t, uint8_t, Float16
///
/// This covers all supported mixed-type distance computations.
///

/// Helper macro for L2 distance explicit instantiation
/// @param SPEC Either `template` (for definitions) or `extern template` (for declarations)
/// @param N Dimensionality (e.g., 64, 128, Dynamic)
/// @param AVX AVX availability level (AVX_AVAILABILITY::AVX512 or AVX_AVAILABILITY::AVX2)
#define DISTANCE_L2_TEMPLATE_HELPER(SPEC, N, AVX)               \
    SPEC struct L2Impl<N, float, float, AVX>;                   \
    SPEC struct L2Impl<N, float, int8_t, AVX>;                  \
    SPEC struct L2Impl<N, float, uint8_t, AVX>;                 \
    SPEC struct L2Impl<N, float, svs::float16::Float16, AVX>;   \
    SPEC struct L2Impl<N, int8_t, float, AVX>;                  \
    SPEC struct L2Impl<N, int8_t, int8_t, AVX>;                 \
    SPEC struct L2Impl<N, int8_t, uint8_t, AVX>;                \
    SPEC struct L2Impl<N, int8_t, svs::float16::Float16, AVX>;  \
    SPEC struct L2Impl<N, uint8_t, float, AVX>;                 \
    SPEC struct L2Impl<N, uint8_t, int8_t, AVX>;                \
    SPEC struct L2Impl<N, uint8_t, uint8_t, AVX>;               \
    SPEC struct L2Impl<N, uint8_t, svs::float16::Float16, AVX>; \
    SPEC struct L2Impl<N, svs::float16::Float16, float, AVX>;   \
    SPEC struct L2Impl<N, svs::float16::Float16, int8_t, AVX>;  \
    SPEC struct L2Impl<N, svs::float16::Float16, uint8_t, AVX>; \
    SPEC struct L2Impl<N, svs::float16::Float16, svs::float16::Float16, AVX>;

/// Instantiate L2 distance implementations (use in .cpp files)
#define DISTANCE_L2_INSTANTIATE_TEMPLATE(N, AVX) \
    DISTANCE_L2_TEMPLATE_HELPER(template, N, AVX);

/// Declare external L2 distance implementations (use in .h files)
#define DISTANCE_L2_EXTERN_TEMPLATE(N, AVX) \
    DISTANCE_L2_TEMPLATE_HELPER(extern template, N, AVX);

/// Helper macro for Inner Product explicit instantiation
/// @param SPEC Either `template` (for definitions) or `extern template` (for declarations)
/// @param N Dimensionality (e.g., 64, 128, Dynamic)
/// @param AVX AVX availability level (AVX_AVAILABILITY::AVX512 or AVX_AVAILABILITY::AVX2)
#define DISTANCE_IP_TEMPLATE_HELPER(SPEC, N, AVX)               \
    SPEC struct IPImpl<N, float, float, AVX>;                   \
    SPEC struct IPImpl<N, float, int8_t, AVX>;                  \
    SPEC struct IPImpl<N, float, uint8_t, AVX>;                 \
    SPEC struct IPImpl<N, float, svs::float16::Float16, AVX>;   \
    SPEC struct IPImpl<N, int8_t, float, AVX>;                  \
    SPEC struct IPImpl<N, int8_t, int8_t, AVX>;                 \
    SPEC struct IPImpl<N, int8_t, uint8_t, AVX>;                \
    SPEC struct IPImpl<N, int8_t, svs::float16::Float16, AVX>;  \
    SPEC struct IPImpl<N, uint8_t, float, AVX>;                 \
    SPEC struct IPImpl<N, uint8_t, int8_t, AVX>;                \
    SPEC struct IPImpl<N, uint8_t, uint8_t, AVX>;               \
    SPEC struct IPImpl<N, uint8_t, svs::float16::Float16, AVX>; \
    SPEC struct IPImpl<N, svs::float16::Float16, float, AVX>;   \
    SPEC struct IPImpl<N, svs::float16::Float16, int8_t, AVX>;  \
    SPEC struct IPImpl<N, svs::float16::Float16, uint8_t, AVX>; \
    SPEC struct IPImpl<N, svs::float16::Float16, svs::float16::Float16, AVX>;

/// Instantiate Inner Product implementations (use in .cpp files)
#define DISTANCE_IP_INSTANTIATE_TEMPLATE(N, AVX) \
    DISTANCE_IP_TEMPLATE_HELPER(template, N, AVX);

/// Declare external Inner Product implementations (use in .h files)
#define DISTANCE_IP_EXTERN_TEMPLATE(N, AVX) \
    DISTANCE_IP_TEMPLATE_HELPER(extern template, N, AVX);

/// Helper macro for Cosine Similarity explicit instantiation
/// @param SPEC Either `template` (for definitions) or `extern template` (for declarations)
/// @param N Dimensionality (e.g., 64, 128, Dynamic)
/// @param AVX AVX availability level (AVX_AVAILABILITY::AVX512 or AVX_AVAILABILITY::AVX2)
#define DISTANCE_CS_TEMPLATE_HELPER(SPEC, N, AVX)                             \
    SPEC struct CosineSimilarityImpl<N, float, float, AVX>;                   \
    SPEC struct CosineSimilarityImpl<N, float, int8_t, AVX>;                  \
    SPEC struct CosineSimilarityImpl<N, float, uint8_t, AVX>;                 \
    SPEC struct CosineSimilarityImpl<N, float, svs::float16::Float16, AVX>;   \
    SPEC struct CosineSimilarityImpl<N, int8_t, float, AVX>;                  \
    SPEC struct CosineSimilarityImpl<N, int8_t, int8_t, AVX>;                 \
    SPEC struct CosineSimilarityImpl<N, int8_t, uint8_t, AVX>;                \
    SPEC struct CosineSimilarityImpl<N, int8_t, svs::float16::Float16, AVX>;  \
    SPEC struct CosineSimilarityImpl<N, uint8_t, float, AVX>;                 \
    SPEC struct CosineSimilarityImpl<N, uint8_t, int8_t, AVX>;                \
    SPEC struct CosineSimilarityImpl<N, uint8_t, uint8_t, AVX>;               \
    SPEC struct CosineSimilarityImpl<N, uint8_t, svs::float16::Float16, AVX>; \
    SPEC struct CosineSimilarityImpl<N, svs::float16::Float16, float, AVX>;   \
    SPEC struct CosineSimilarityImpl<N, svs::float16::Float16, int8_t, AVX>;  \
    SPEC struct CosineSimilarityImpl<N, svs::float16::Float16, uint8_t, AVX>; \
    SPEC struct CosineSimilarityImpl<N, svs::float16::Float16, svs::float16::Float16, AVX>;

/// Instantiate Cosine Similarity implementations (use in .cpp files)
#define DISTANCE_CS_INSTANTIATE_TEMPLATE(N, AVX) \
    DISTANCE_CS_TEMPLATE_HELPER(template, N, AVX);

/// Declare external Cosine Similarity implementations (use in .h files)
#define DISTANCE_CS_EXTERN_TEMPLATE(N, AVX) \
    DISTANCE_CS_TEMPLATE_HELPER(extern template, N, AVX);
