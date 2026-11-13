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

#include <immintrin.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace svs::distance {

// The scalar-distance kernels (L2/IP/cosine) and QEmd/QChamfer/QChamferExactImpl dispatch
// through a runtime if-chain on svs::detail::avx_runtime_flags (euclidean.h, cosine.h,
// inner_product.h, gem.h), never through a switch or template argument naming
// AVX_AVAILABILITY::AMX; adding the AMX enumerator needs no new instantiation of any of
// them (checked: `git grep -n AVX_AVAILABILITY` finds no exhaustive switch or `AMX`
// branch to add outside InnerProductMatrix). AMX only applies to the InnerProductMatrix
// family below, whose public dispatch is the one exhaustive switch this change extends.

namespace {

// 64-byte tile-config struct the palette-1 AMX instructions read (Intel SDM Vol 1,
// 3.3.7): colsb/rows per tmm register 0-15, reserved bytes must stay zero.
struct TileConfig {
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved[14] = {};
    uint16_t colsb[16] = {};
    uint8_t rows[16] = {};
};

// Releases tile state on every exit path (normal return or exception): leaving tile
// state live costs later context switches and ordinary SSE/AVX code extra save/restore
// work, per Intel's optimization guidance for AMX.
struct TileGuard {
    TileGuard() = default;
    ~TileGuard() { _tile_release(); }
    TileGuard(const TileGuard&) = delete;
    TileGuard& operator=(const TileGuard&) = delete;
};

// tmm register assignment: tmm0 = A's current k-chunk (16 rows x 32 bf16, i.e. 16
// pairs); tmm1/tmm2 = B's two n-lane-tiles for the pair-chunk (pair-interleaved
// transposed bf16, see PackedOperand::tile_block); tmm3/tmm4 = the matching fp32
// accumulators. Two n-tiles in flight so their dpbf16ps chains are independent of each
// other (both read tmm0, but write distinct accumulators), per the task's "several
// accumulator tiles overlap" requirement. Named here for the tile-config array indexing
// below (ordinary C++, any expression is fine); every `_tile_*` call past this point
// uses the literal 0/1/2/3/4 directly, because gcc's tile intrinsics are macros that
// stringify their register argument (`#dst`), so a named constant used there would
// stringify to its identifier text, not a digit, and fail to assemble.
enum Tmm : int { kTmmA = 0, kTmmB0 = 1, kTmmB1 = 2, kTmmAcc0 = 3, kTmmAcc1 = 4 };

void configure_tiles(uint8_t a_acc_rows) {
    TileConfig cfg;
    cfg.palette_id = 1;
    cfg.rows[kTmmA] = a_acc_rows;
    cfg.colsb[kTmmA] = 64;
    cfg.rows[kTmmAcc0] = a_acc_rows;
    cfg.colsb[kTmmAcc0] = 64;
    cfg.rows[kTmmAcc1] = a_acc_rows;
    cfg.colsb[kTmmAcc1] = 64;
    // B's tiles are always a full 16 (pair-)rows: PackedOperand::tile_block zero-pads
    // every tile block to 16x16, so there is no partial-row case to configure for them.
    cfg.rows[kTmmB0] = 16;
    cfg.colsb[kTmmB0] = 64;
    cfg.rows[kTmmB1] = 16;
    cfg.colsb[kTmmB1] = 64;
    _tile_loadconfig(&cfg);
}

// Computes the dot products of up to 16 rows of A (starting at `a_row0`, already known
// not to read past `d` columns of any row via the tail copy below) against B's n-tiles
// `nt0` (and `nt1` when `has_nt1`) for every k-chunk, leaving the result in tmm3 (and
// tmm4). Tiles must already be configured by configure_tiles() for this call's row count.
//
// A's tail k-chunk (the last one, when d is not a multiple of 32) is copied through a
// zero-padded stack buffer rather than loaded directly: loading straight from `a_row0`
// would read up to 31 bf16 elements past column d-1 of the block's last real row, which
// is caller memory this kernel does not own past `d`.
void accumulate_group(
    const svs::BFloat16* a_row0,
    size_t rows,
    size_t stride_bytes,
    size_t full_k_chunks,
    size_t k_tail,
    const PackedOperand& b,
    size_t nt0,
    bool has_nt1,
    size_t nt1
) {
    _tile_zero(3);
    if (has_nt1) {
        _tile_zero(4);
    }
    const auto* base = reinterpret_cast<const char*>(a_row0);
    for (size_t kt = 0; kt < full_k_chunks; ++kt) {
        _tile_loadd(0, base + kt * 64, stride_bytes);
        _tile_loadd(1, b.tile_block(nt0, kt), 64);
        _tile_dpbf16ps(3, 0, 1);
        if (has_nt1) {
            _tile_loadd(2, b.tile_block(nt1, kt), 64);
            _tile_dpbf16ps(4, 0, 2);
        }
    }
    if (k_tail > 0) {
        alignas(64) uint16_t tail[16][32] = {};
        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(tail[r], base + r * stride_bytes + full_k_chunks * 64, k_tail * 2);
        }
        // _tile_loadd's asm has no memory operand or clobber, so nothing otherwise tells
        // the optimizer the memcpy above must finish before this reads `tail`: without
        // this barrier the store can be sunk past the load, observed as a silent
        // zero/garbage read on this compiler.
        __asm__ volatile("" ::: "memory");
        _tile_loadd(0, tail, 64);
        _tile_loadd(1, b.tile_block(nt0, full_k_chunks), 64);
        _tile_dpbf16ps(3, 0, 1);
        if (has_nt1) {
            _tile_loadd(2, b.tile_block(nt1, full_k_chunks), 64);
            _tile_dpbf16ps(4, 0, 2);
        }
    }
}

// Shared row-block/n-tile-group driver for both output forms below: walks A 16 rows at a
// time (reconfiguring tiles once more only for a final row-block narrower than 16, never
// per row-block in the steady state) and B 32 lanes (two n-tiles) at a time, handing each
// block's two 16x16 fp32 results to `emit(row0, rows, nt0, has_nt1, nt1, c0, c1)`.
template <typename Emit>
void for_each_block(
    const svs::BFloat16* a, size_t m, size_t d, const PackedOperand& b, Emit&& emit
) {
    size_t n_tiles = b.n_tiles();
    size_t full_k_chunks = d / 32;
    size_t k_tail = d % 32;
    size_t stride_bytes = d * 2;

    TileGuard guard;
    configure_tiles(16);
    bool configured_full = true;

    for (size_t i = 0; i < m; i += 16) {
        size_t rows = std::min<size_t>(16, m - i);
        if (rows < 16 && configured_full) {
            configure_tiles(static_cast<uint8_t>(rows));
            configured_full = false;
        }
        const svs::BFloat16* row0 = a + i * d;
        for (size_t nt0 = 0; nt0 < n_tiles; nt0 += 2) {
            bool has_nt1 = nt0 + 1 < n_tiles;
            size_t nt1 = has_nt1 ? nt0 + 1 : nt0;
            alignas(64) float c0[16][16];
            alignas(64) float c1[16][16];
            accumulate_group(
                row0, rows, stride_bytes, full_k_chunks, k_tail, b, nt0, has_nt1, nt1
            );
            _tile_stored(3, c0, 64);
            if (has_nt1) {
                _tile_stored(4, c1, 64);
            }
            // Same hazard as accumulate_group's tail load: _tile_stored's asm has no
            // output memory operand, so without this barrier the optimizer is free to
            // read c0/c1 in `emit` before the store above has actually run.
            __asm__ volatile("" ::: "memory");
            emit(i, rows, nt0, has_nt1, nt1, c0, c1);
        }
    }
}

} // namespace

// AMX table kernel for a BFloat16 A against a tile-packed B: writes out[i*n+j] exactly
// like the generic/AVX-512 forms, with the same <bf16(a), bf16(b)> values (summed in a
// different order, hence InnerProductMatrix's tolerance-based tests, not a bit-exact
// one).
void amx_table_bf16(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
    size_t n = b.n();
    for_each_block(
        a,
        m,
        b.d(),
        b,
        [&](size_t row0,
            size_t rows,
            size_t nt0,
            bool has_nt1,
            size_t nt1,
            auto& c0,
            auto& c1) {
            size_t lanes0 = std::min<size_t>(16, n - nt0 * 16);
            for (size_t r = 0; r < rows; ++r) {
                float* dst = out + (row0 + r) * n + nt0 * 16;
                std::memcpy(dst, c0[r], lanes0 * sizeof(float));
            }
            if (has_nt1) {
                size_t lanes1 = std::min<size_t>(16, n - nt1 * 16);
                for (size_t r = 0; r < rows; ++r) {
                    float* dst = out + (row0 + r) * n + nt1 * 16;
                    std::memcpy(dst, c1[r], lanes1 * sizeof(float));
                }
            }
        }
    );
}

// AMX column-max kernel for a BFloat16 A against a tile-packed B: folds each block's two
// 16x16 tiles into the running per-lane maximum with AVX-512 (the AMX level implies
// AVX-512F), reusing the same row-then-lane reduction shape as
// InnerProductMatrixImpl<AVX512>::column_max in avx512.cpp.
void amx_column_max_bf16(
    const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out
) {
    size_t n = b.n();
    std::fill(out, out + n, -std::numeric_limits<float>::infinity());
    if (m == 0) {
        return;
    }
    for_each_block(
        a,
        m,
        b.d(),
        b,
        [&](size_t /*row0*/,
            size_t rows,
            size_t nt0,
            bool has_nt1,
            size_t nt1,
            auto& c0,
            auto& c1) {
            auto fold = [&](size_t nt, auto& c) {
                size_t lanes = std::min<size_t>(16, n - nt * 16);
                alignas(64) float colmax[16] = {};
                for (size_t j = 0; j < 16; ++j) {
                    float best = -std::numeric_limits<float>::infinity();
                    for (size_t r = 0; r < rows; ++r) {
                        best = std::max(best, c[r][j]);
                    }
                    colmax[j] = best;
                }
                for (size_t j = 0; j < lanes; ++j) {
                    out[nt * 16 + j] = std::max(out[nt * 16 + j], colmax[j]);
                }
            };
            fold(nt0, c0);
            if (has_nt1) {
                fold(nt1, c1);
            }
        }
    );
}

SVS_VALIDATE_BOOL_ENV(SVS_AMX_TILE_BF16)
#if SVS_AMX_TILE_BF16

void InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
    const float* a, size_t m, const PackedOperand& b, float* out
) {
    // Never the AMX precision case (A is float32, not bf16): one source of truth is the
    // AVX-512 kernel.
    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(a, m, b, out);
}

void InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
    const svs::Float16* a, size_t m, const PackedOperand& b, float* out
) {
    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(a, m, b, out);
}

void InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
    const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out
) {
    if (m == 0 || b.n() == 0) {
        return;
    }
    if (!b.has_tile_layout()) {
        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(a, m, b, out);
        return;
    }
    amx_table_bf16(a, m, b, out);
}

void InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build(
    const float* a, size_t m, const float* b, size_t n, size_t d, float* out
) {
    // `a` is always float32 here (InnerProductMatrix::compute's only entry point to this
    // overload): the AMX precision case never applies, so this is a plain forward.
    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build(a, m, b, n, d, out);
}

void InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::column_max(
    const float* a, size_t m, const PackedOperand& b, float* out
) {
    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(a, m, b, out);
}

void InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::column_max(
    const svs::Float16* a, size_t m, const PackedOperand& b, float* out
) {
    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(a, m, b, out);
}

void InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::column_max(
    const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out
) {
    if (!b.has_tile_layout()) {
        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(a, m, b, out);
        return;
    }
    amx_column_max_bf16(a, m, b, out);
}

template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>;

#endif // SVS_AMX_TILE_BF16

} // namespace svs::distance

#endif
