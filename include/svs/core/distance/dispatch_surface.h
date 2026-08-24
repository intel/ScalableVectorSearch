/*
 * Copyright 2026 Intel Corporation
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

// GENERATED FILE -- DO NOT EDIT. Regenerated on every CMake configure from
// cmake/dispatch-surface.cmake; committed so a bare `-I include` compile works.

#pragma once

// clang-format off
// Reflowing would realign the escaped newlines and make the committed copy
// disagree with the one CMake writes -- an endless format/regenerate loop.

// Number of extents with a fixed-extent kernel, including svs::Dynamic.
#define SVS_SUPPORTED_DIM_COUNT 9

// Invokes M(extent) once per extent.
#define SVS_FOR_EACH_SUPPORTED_DIM(M) \
    M(64) \
    M(96) \
    M(100) \
    M(128) \
    M(160) \
    M(200) \
    M(512) \
    M(768) \
    M(svs::Dynamic) \
    /* end */

// Invokes M(extent, isa_level) once per (extent, ISA level) pair -- that is,
// once per kernel the library compiles ahead of time, modulo type pairs.
#define SVS_FOR_EACH_DISPATCH_TARGET(M) \
    M(64, AVX2) \
    M(96, AVX2) \
    M(100, AVX2) \
    M(128, AVX2) \
    M(160, AVX2) \
    M(200, AVX2) \
    M(512, AVX2) \
    M(768, AVX2) \
    M(svs::Dynamic, AVX2) \
    M(64, AVX512) \
    M(96, AVX512) \
    M(100, AVX512) \
    M(128, AVX512) \
    M(160, AVX512) \
    M(200, AVX512) \
    M(512, AVX512) \
    M(768, AVX512) \
    M(svs::Dynamic, AVX512) \
    /* end */

// Invokes M(isa_level) once per ISA level, weakest first. AVX_AVAILABILITY
// enumerators without a translation unit are absent: this is the surface.
#define SVS_FOR_EACH_ISA_LEVEL(M) \
    M(AVX2) \
    M(AVX512) \
    /* end */

// clang-format on
