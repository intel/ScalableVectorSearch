/*
 * Copyright 2024 Intel Corporation
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

// svs python bindings
#include "svs/python/svs_mkl.h"

// svs
#include "svs/lib/preprocessor.h"

// stl
#include <optional>

namespace svs::python {
bool have_mkl() { return false; }
std::optional<size_t> mkl_num_threads() { return std::nullopt; }
} // namespace svs::python
