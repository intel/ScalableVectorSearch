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

#if defined(__x86_64__)
#include "svs/core/distance/inner_product_matrix.h"

namespace svs::distance {

SVS_VALIDATE_BOOL_ENV(SVS_AMX_TILE_BF16)
#if SVS_AMX_TILE_BF16
template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>;
#endif

} // namespace svs::distance

#endif
