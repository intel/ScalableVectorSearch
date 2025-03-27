/*
 * Copyright 2023 Intel Corporation
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

#include "svs/fallback/fallback_mode.h"

#ifndef USE_PROPRIETARY

#include "svs/fallback/lvq_fallback.h"
#include "svs/fallback/leanvec_fallback.h"

#else // USE_PROPRIETARY

#include "../../../../include/svs/quantization/lvq/lvq.h"
#include "../../../../include/svs/leanvec/leanvec.h"
#include "../../../../include/svs/flat/vamana/lvq.h"
#include "../../../../include/svs/flat/vamana/leanvec.h"
#include "../../../../include/svs/inverted/vamana/lvq.h"
#include "../../../../include/svs/inverted/vamana/leanvec.h"
#include "../../../../include/svs/extensions/vamana/lvq.h"
#include "../../../../include/svs/extensions/vamana/leanvec.h"

#endif // USE_PROPRIETARY
