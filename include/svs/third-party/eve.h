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

#include "eve/module/core.hpp"
#include "eve/wide.hpp"

namespace svs {

// Helper alias to cut down of visual clutter.
// Most internal uses of `wide` explicitly request the register width as well.
template <typename T, int64_t N> using wide_ = eve::wide<T, eve::fixed<N>>;

} // namespace svs
