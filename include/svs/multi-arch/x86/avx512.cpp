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
///// Inner Product Runtime Dispatch Wrappers
/////

template<typename Ea, typename Eb, size_t N>
float ip_float_avx512(const Ea* a, const Eb* b, lib::MaybeStatic<N> length) {
    return simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
}

// Explicit instantiations for common type combinations
template float ip_float_avx512<float, float, Dynamic>(const float*, const float*, lib::MaybeStatic<Dynamic>);
template float ip_float_avx512<float, uint8_t, Dynamic>(const float*, const uint8_t*, lib::MaybeStatic<Dynamic>);
template float ip_float_avx512<float, int8_t, Dynamic>(const float*, const int8_t*, lib::MaybeStatic<Dynamic>);
template float ip_float_avx512<Float16, Float16, Dynamic>(const Float16*, const Float16*, lib::MaybeStatic<Dynamic>);

/////
///// L2 (Euclidean) Runtime Dispatch Wrappers
/////

template<typename Ea, typename Eb, size_t N>
float l2_float_avx512(const Ea* a, const Eb* b, lib::MaybeStatic<N> length) {
    return simd::generic_simd_op(L2FloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
}

// Explicit instantiations for common type combinations
template float l2_float_avx512<float, float, Dynamic>(const float*, const float*, lib::MaybeStatic<Dynamic>);
template float l2_float_avx512<float, uint8_t, Dynamic>(const float*, const uint8_t*, lib::MaybeStatic<Dynamic>);
template float l2_float_avx512<float, int8_t, Dynamic>(const float*, const int8_t*, lib::MaybeStatic<Dynamic>);
template float l2_float_avx512<Float16, Float16, Dynamic>(const Float16*, const Float16*, lib::MaybeStatic<Dynamic>);

/////
///// Cosine Similarity Runtime Dispatch Wrappers
/////

template<typename Ea, typename Eb, size_t N>
std::pair<float, float> cosine_float_avx512(const Ea* a, const Eb* b, lib::MaybeStatic<N> length) {
    return simd::generic_simd_op(CosineFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
}

// Explicit instantiations for common type combinations
template std::pair<float, float> cosine_float_avx512<float, float, Dynamic>(const float*, const float*, lib::MaybeStatic<Dynamic>);
template std::pair<float, float> cosine_float_avx512<Float16, Float16, Dynamic>(const Float16*, const Float16*, lib::MaybeStatic<Dynamic>);

} // namespace svs::distance

#endif // defined(__x86_64__)
