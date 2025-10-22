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

#define DISTANCE_L2_INSTANTIATE_TEMPLATE(N, AVX) \
    DISTANCE_L2_TEMPLATE_HELPER(template, N, AVX);

#define DISTANCE_L2_EXTERN_TEMPLATE(N, AVX) \
    DISTANCE_L2_TEMPLATE_HELPER(extern template, N, AVX);

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

#define DISTANCE_IP_INSTANTIATE_TEMPLATE(N, AVX) \
    DISTANCE_IP_TEMPLATE_HELPER(template, N, AVX);

#define DISTANCE_IP_EXTERN_TEMPLATE(N, AVX) \
    DISTANCE_IP_TEMPLATE_HELPER(extern template, N, AVX);

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

#define DISTANCE_CS_INSTANTIATE_TEMPLATE(N, AVX) \
    DISTANCE_CS_TEMPLATE_HELPER(template, N, AVX);

#define DISTANCE_CS_EXTERN_TEMPLATE(N, AVX) \
    DISTANCE_CS_TEMPLATE_HELPER(extern template, N, AVX);
