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

#include "svs/core/distance/dispatch_surface.h"

/////
///// Element-type pairs.
/////
///// Hand-written, unlike the generated extent list: a pair is here because a
///// kernel exists for it, and generating it would invite unimplemented pairs.
/////

// Invokes M(query_type, dataset_type, ...) once per type pair.
#define SVS_FOR_EACH_TYPE_PAIR(M, ...)             \
    M(float, float, __VA_ARGS__)                   \
    M(float, int8_t, __VA_ARGS__)                  \
    M(float, uint8_t, __VA_ARGS__)                 \
    M(float, svs::float16::Float16, __VA_ARGS__)   \
    M(int8_t, float, __VA_ARGS__)                  \
    M(int8_t, int8_t, __VA_ARGS__)                 \
    M(int8_t, uint8_t, __VA_ARGS__)                \
    M(int8_t, svs::float16::Float16, __VA_ARGS__)  \
    M(uint8_t, float, __VA_ARGS__)                 \
    M(uint8_t, int8_t, __VA_ARGS__)                \
    M(uint8_t, uint8_t, __VA_ARGS__)               \
    M(uint8_t, svs::float16::Float16, __VA_ARGS__) \
    M(svs::float16::Float16, float, __VA_ARGS__)   \
    M(svs::float16::Float16, int8_t, __VA_ARGS__)  \
    M(svs::float16::Float16, uint8_t, __VA_ARGS__) \
    M(svs::float16::Float16, svs::float16::Float16, __VA_ARGS__)

/////
///// Which type pairs each ISA level has kernels for.
/////
///// One line per level in SVS_ISA_LEVELS. A level with no entry here is a
///// compile error rather than a silently empty instantiation list.
/////

#define SVS_TYPE_PAIRS_NONE SVS_FOR_EACH_TYPE_PAIR
#define SVS_TYPE_PAIRS_AVX2 SVS_FOR_EACH_TYPE_PAIR
#define SVS_TYPE_PAIRS_AVX512 SVS_FOR_EACH_TYPE_PAIR

/////
///// Instantiation.
/////

// Resolve LEVEL to its type-pair list. The indirection is required so that
// LEVEL is expanded before being pasted.
#define SVS_TYPE_PAIRS_FOR_(LEVEL) SVS_TYPE_PAIRS_##LEVEL
#define SVS_TYPE_PAIRS_FOR(LEVEL, M, ...) SVS_TYPE_PAIRS_FOR_(LEVEL)(M, __VA_ARGS__)

#define SVS_DECLARE_ONE_L2(Ea, Eb, SPEC, N, LEVEL) \
    SPEC struct L2Impl<N, Ea, Eb, AVX_AVAILABILITY::LEVEL>;
#define SVS_DECLARE_ONE_IP(Ea, Eb, SPEC, N, LEVEL) \
    SPEC struct IPImpl<N, Ea, Eb, AVX_AVAILABILITY::LEVEL>;
#define SVS_DECLARE_ONE_CS(Ea, Eb, SPEC, N, LEVEL) \
    SPEC struct CosineSimilarityImpl<N, Ea, Eb, AVX_AVAILABILITY::LEVEL>;

// SPEC is `template` for a definition or `extern template` for a declaration.
#define SVS_INSTANTIATE_L2(SPEC, N, LEVEL) \
    SVS_TYPE_PAIRS_FOR(LEVEL, SVS_DECLARE_ONE_L2, SPEC, N, LEVEL)
#define SVS_INSTANTIATE_IP(SPEC, N, LEVEL) \
    SVS_TYPE_PAIRS_FOR(LEVEL, SVS_DECLARE_ONE_IP, SPEC, N, LEVEL)
#define SVS_INSTANTIATE_CS(SPEC, N, LEVEL) \
    SVS_TYPE_PAIRS_FOR(LEVEL, SVS_DECLARE_ONE_CS, SPEC, N, LEVEL)

// All three distances at once. Only usable where all three are declared, i.e.
// in a generated translation unit.
#define SVS_INSTANTIATE_DISTANCES(SPEC, N, LEVEL) \
    SVS_INSTANTIATE_L2(SPEC, N, LEVEL)            \
    SVS_INSTANTIATE_IP(SPEC, N, LEVEL)            \
    SVS_INSTANTIATE_CS(SPEC, N, LEVEL)
