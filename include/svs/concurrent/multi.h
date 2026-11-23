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
#include "svs/index/vamana/multi.h"

namespace svs::index::vamana::concurrent {

using vamana::auto_multi_dynamic_assemble;
using vamana::MultiBatchIterator;
using vamana::MultiMutableVamanaIndex;
using vamana::MultiMutableVamanaLoad;

} // namespace svs::index::vamana::concurrent
