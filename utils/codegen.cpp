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

// svs
#include "svs/quantization/lvq/compressed.h"

// stl
#include <span>

namespace lvq = svs::quantization::lvq;

void unpack_cv(
    std::span<int32_t> dst,
    lvq::CompressedVector<lvq::Unsigned, 8, 768, lvq::Turbo<16, 4>> cv
) {
    lvq::unpack(dst, cv);
}

void unpack_combined(
    std::span<int32_t> dst, lvq::Combined<4, 8, svs::Dynamic, lvq::Turbo<16, 8>> cv
) {
    lvq::unpack(dst, cv);
}
