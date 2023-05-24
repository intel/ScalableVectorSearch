/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

#include "eve/module/core.hpp"
#include "eve/wide.hpp"

namespace svs {

// Helper alias to cut down of visual clutter.
// Most internal uses of `wide` explicitly request the register width as well.
template <typename T, int64_t N> using wide_ = eve::wide<T, eve::fixed<N>>;

} // namespace svs
