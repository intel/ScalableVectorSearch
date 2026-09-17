#!/bin/bash
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Compare the ABI of two SVS build tarballs with napetrov/abicheck.
#
# One primitive, three callers, all asking the same temporal question -- is `new`
# a drop-in replacement for `old`? Only the baseline differs:
#   per-PR      old = the newest main build, new = this build (both repos)
#   nightly     old = the last release asset, new = today's main build
# The distinction is entirely in what the caller passes, so this lives in one
# place: this repo's build-cpp-runtime-bindings.yml and build-share-lib.yml in
# the innersource repo, which reaches it through the submodule path.
#
# Usage: abi-check.sh <old-label> <old-tarball> <new-label> <new-tarball>
#
# Overridable via environment:
#   LIBRARY        library basename to compare (default libsvs_runtime.so)
#   HEADER_SUBDIR  header root inside the tarball (default include/svs/runtime)
#   SUPPRESSIONS   suppression file (default .github/abi-suppressions.yml)
#   POLICY         abicheck policy (default strict_abi)
#   DEPTH          abicheck --depth (default: unset, i.e. abicheck's own
#                  'headers'). Set to 'binary' for the shared library, whose
#                  header set does not finish parsing in any usable time.
#   REPORT         markdown report path (default abi-report.md)
#   WORKDIR        scratch directory (default abi-work)
#
# Exit status is abicheck's: 0 compatible, 2 source-level API break, 4 binary
# ABI break, 5 budget exceeded, 16 not comparable, 64 invalid invocation. 77 is
# added to mean "an input was missing", which is not a finding and must never be
# reported as one.

set -euo pipefail

OLD_LABEL="${1:?Usage: abi-check.sh <old-label> <old-tarball> <new-label> <new-tarball>}"
OLD_TARBALL="${2:?missing old tarball}"
NEW_LABEL="${3:?missing new label}"
NEW_TARBALL="${4:?missing new tarball}"

LIBRARY="${LIBRARY:-libsvs_runtime.so}"
HEADER_SUBDIR="${HEADER_SUBDIR:-include/svs/runtime}"
SUPPRESSIONS="${SUPPRESSIONS:-.github/abi-suppressions.yml}"
POLICY="${POLICY:-strict_abi}"
DEPTH="${DEPTH:-}"
REPORT="${REPORT:-abi-report.md}"
WORKDIR="${WORKDIR:-abi-work}"

# Prints the resolved library path on success. Release tarballs and CI artifacts
# have differed on whether the unversioned symlink is included, so accept the
# versioned real file too.
unpack() {
    local tarball="$1" dest="$2" label="$3" lib

    if [ ! -f "$tarball" ]; then
        echo "::warning::$label: tarball not found at $tarball" >&2
        return 1
    fi

    rm -rf "$dest"
    mkdir -p "$dest"
    tar -xzf "$tarball" -C "$dest"

    lib=$(find "$dest" -name "$LIBRARY" -o -name "$LIBRARY.*" | sort | head -1)
    if [ -z "$lib" ]; then
        echo "::error::$label: $LIBRARY not found in $tarball" >&2
        return 1
    fi
    if [ ! -d "$dest/$HEADER_SUBDIR" ]; then
        echo "::error::$label: headers missing from $tarball ($HEADER_SUBDIR)" >&2
        return 1
    fi
    echo "$lib"
}

OLD_DIR="$WORKDIR/old"
NEW_DIR="$WORKDIR/new"

OLD_LIB=$(unpack "$OLD_TARBALL" "$OLD_DIR" "$OLD_LABEL") || exit 77
NEW_LIB=$(unpack "$NEW_TARBALL" "$NEW_DIR" "$NEW_LABEL") || exit 77

echo "Comparing $LIBRARY: [$OLD_LABEL] -> [$NEW_LABEL]"

suppress_args=()
[ -f "$SUPPRESSIONS" ] && suppress_args=(--suppress "$SUPPRESSIONS")

# Left unset by default so .abicheck.yml stays the single place that configures
# depth. The --header/--include arguments below are still passed at binary depth,
# where they are simply unused, so nothing here is conditional on this.
depth_args=()
[ -n "$DEPTH" ] && depth_args=(--depth "$DEPTH")

# Vendored deps ship under a versioned root (include/eve-2023.2.15/eve/...), so
# -I include alone cannot resolve their own `#include <eve/...>`.
#
# Only roots present on BOTH sides are added, in one sorted order. abicheck
# fingerprints the include sequence and refuses to compare (rc=16) when the two
# sides differ, so a root added to one side only voids the whole run -- and the
# private-source and public tarballs genuinely disagree about shipping eve.
include_args=(--include old="$OLD_DIR/include" --include new="$NEW_DIR/include")
for dir in "$OLD_DIR"/include/*-[0-9]*/; do
    [ -d "$dir" ] || continue
    vendored=$(basename "$dir")
    [ -d "$NEW_DIR/include/$vendored" ] || continue
    include_args+=(--include old="$OLD_DIR/include/$vendored" \
                   --include new="$NEW_DIR/include/$vendored")
done

# Headers that cannot be parsed as a translation unit, so they are never part of
# the compared surface. Only reachable when HEADER_SUBDIR is broad (the shared
# library legs); a no-op for the runtime bindings.
#   core.h / lib.h  documentation umbrellas, literally `static_assert(false, ...)`
#   cpuid.h         abicheck puts each parsed header's own directory on the
#                   include path, so include/svs shadows the system <cpuid.h>
#                   that svs/cpuid.h itself includes -- #pragma once then makes
#                   it a no-op and __cpuid is undeclared. Costs two inline
#                   svs::detail functions.
#   vendored trees  parsed transitively via -I where svs actually uses them; eve
#                   sweeps in ARM SVE headers that cannot compile on x86.
#   ivf             svs/index/ivf/common.h includes <mkl.h> unguarded and no
#                   tarball ships MKL headers, so IVF is compared at symbol level
#                   only. Install MKL headers on the runner to restore it.
exclude_args=()
# Every pattern is leading-* so it matches the full path, not just a path relative
# to HEADER_SUBDIR, which varies per leg.
for pattern in '*svs/core/core.h' '*svs/lib/lib.h' '*svs/cpuid.h' \
               '*svs/index/ivf/*' '*svs/extensions/ivf/*' '*svs/orchestrators/*ivf*' \
               '*/eve-*/*' '*/fmt/*' '*/spdlog/*' '*/tsl/*' '*/toml++/*'; do
    exclude_args+=(--exclude-header "$pattern")
done

rc=0
# Without pipefail the pipe into tee masks abicheck's exit status.
set -o pipefail
# The clang AST frontend and the C++20 standard come from .abicheck.yml. abicheck
# 0.6 dropped the --ast-frontend/--gcc-options flags, so that file is now the only
# place to set them -- do not reintroduce them here.
abicheck compare \
    "$OLD_LIB" \
    "$NEW_LIB" \
    --header old="$OLD_DIR/$HEADER_SUBDIR" \
    --header new="$NEW_DIR/$HEADER_SUBDIR" \
    "${include_args[@]}" \
    "${exclude_args[@]}" \
    --version old="$OLD_LABEL" \
    --version new="$NEW_LABEL" \
    --policy "$POLICY" \
    "${depth_args[@]}" \
    "${suppress_args[@]}" \
    --output markdown=- | tee "$REPORT" || rc=$?

# Only 2, 4 and 5 are findings. Anything else -- a removed flag, an unparseable
# header, a fingerprint mismatch (16), a crash -- means the comparison did not
# happen, and reporting that as an ABI break is worse than reporting nothing: it
# teaches reviewers to ignore a red ABI check.
case "$rc" in
    0)
        echo "ABI compatible: $OLD_LABEL -> $NEW_LABEL"
        ;;
    2 | 4 | 5)
        echo "::warning::ABI incompatibility: $NEW_LABEL differs from $OLD_LABEL (rc=$rc)"
        ;;
    *)
        echo "::error::abicheck could not complete the comparison of $OLD_LABEL ->" \
             "$NEW_LABEL (rc=$rc). This is a harness bug, not an ABI finding."
        ;;
esac

exit "$rc"
