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

#if defined(__x86_64__)
#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"

namespace svs::distance {

// Define every kernel for this ISA level at every generated extent. Extents come
// from cmake/dispatch-surface.cmake, type pairs from multi-arch/x86/preprocessor.h.
#define SVS_DEFINE_FOR_DIM(DIM) SVS_INSTANTIATE_DISTANCES(template, DIM, AVX512)
SVS_FOR_EACH_SUPPORTED_DIM(SVS_DEFINE_FOR_DIM)
#undef SVS_DEFINE_FOR_DIM

} // namespace svs::distance

#endif
