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

// clang-format off

#define SVS_PACK_ARGS(...) __VA_ARGS__

#define SVS_DISTANCE_DYNAMIC_TEMPLATES_BY_MICROARCH(dist, spec, uarch)                  \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, int8_t, int8_t)                 \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, int8_t, uint8_t)                \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, int8_t, float)                  \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, int8_t, svs::float16::Float16)  \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, uint8_t, int8_t)                \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, uint8_t, uint8_t)               \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, uint8_t, float)                 \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, uint8_t, svs::float16::Float16) \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, float, int8_t)                  \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, float, uint8_t)                 \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, float, float)                   \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, float, svs::float16::Float16)   \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, svs::float16::Float16, int8_t)  \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, svs::float16::Float16, uint8_t) \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, svs::float16::Float16, float)   \
    SVS_##dist##_DISTANCE_DYNAMIC_TEMPLATE(spec, uarch, svs::float16::Float16, svs::float16::Float16)

#define SVS_DISTANCE_FIXED_N_TEMPLATES_BY_MICROARCH(dist, spec, uarch, length)                  \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, int8_t, int8_t, length)                 \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, int8_t, uint8_t, length)                \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, int8_t, float, length)                  \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, int8_t, svs::float16::Float16, length)  \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, uint8_t, int8_t, length)                \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, uint8_t, uint8_t, length)               \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, uint8_t, float, length)                 \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, uint8_t, svs::float16::Float16, length) \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, float, int8_t, length)                  \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, float, uint8_t, length)                 \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, float, float, length)                   \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, float, svs::float16::Float16, length)   \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, svs::float16::Float16, int8_t, length)  \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, svs::float16::Float16, uint8_t, length) \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, svs::float16::Float16, float, length)   \
    SVS_##dist##_DISTANCE_FIXED_N_TEMPLATE(spec, uarch, svs::float16::Float16, svs::float16::Float16, length)

#define SVS_DISTANCE_TEMPLATES_BY_MICROARCH(dist, spec, uarch) \
    SVS_DISTANCE_DYNAMIC_TEMPLATES_BY_MICROARCH(dist, spec, uarch)

// clang-format on

#define SVS_FOR_EACH_MICROARCH

#define SVS_FOR_EACH_KNOWN_MICROARCH
