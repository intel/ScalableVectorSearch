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

# Compare the ABI of two SVS build tarballs with abicheck.
#
# Every caller asks the same question -- is `new` a drop-in replacement for `old`?
# -- and differs only in the baseline it passes (newest main build per PR, previous
# release asset when cutting one), so the comparison lives in one place.
#
# Usage: abi-check.sh <old-label> <old-tarball> <new-label> <new-tarball>
#
# Overridable via environment:
#   LIBRARY        library basename to compare (default libsvs_runtime.so)
#   HEADER_SUBDIR  header root inside the tarball (default include/svs/runtime)
#   SUPPRESSIONS   suppression file (default .github/abi-suppressions.yml); a
#                  missing file is skipped silently, so callers that suppress
#                  nothing should still point at a real empty one
#   POLICY         abicheck policy (default strict_abi)
#   DEPTH          abicheck --depth (default unset, i.e. abicheck's 'headers');
#                  'binary' is the escape hatch for a header set too large to parse
#   REPORT         markdown report path (default abi-report.md)
#   WORKDIR        scratch directory (default abi-work)
#
# Exit status is abicheck's: 0 compatible, 2 source-level API break, 4 binary ABI
# break, 5 budget exceeded, 16 not comparable, 64 invalid invocation. 77 is added
# to mean "an input was missing", which is not a finding.

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

# Left unset by default so .abicheck.yml stays the single place that sets depth.
depth_args=()
[ -n "$DEPTH" ] && depth_args=(--depth "$DEPTH")

# Vendored deps ship under a versioned root (include/eve-2023.2.15/eve/...), so
# -I include alone cannot resolve their own `#include <eve/...>`.
#
# Only roots present on BOTH sides are added, in one sorted order: abicheck
# fingerprints the include sequence and refuses to compare (rc=16) when the two
# sides differ, so a root added to one side only voids the whole run. Tarballs do
# genuinely disagree about shipping eve, hence the intersection rather than a union.
include_args=(--include old="$OLD_DIR/include" --include new="$NEW_DIR/include")
for dir in "$OLD_DIR"/include/*-[0-9]*/; do
    [ -d "$dir" ] || continue
    vendored=$(basename "$dir")
    [ -d "$NEW_DIR/include/$vendored" ] || continue
    include_args+=(--include old="$OLD_DIR/include/$vendored" \
                   --include new="$NEW_DIR/include/$vendored")
done

# Headers that cannot be parsed as a translation unit. Inert for the narrow header
# roots the current callers pass; needed as soon as a whole-of-svs root is compared.
#   core.h / lib.h  documentation umbrellas, literally `static_assert(false, ...)`
#   cpuid.h         abicheck puts each parsed header's own directory on the include
#                   path, so include/svs shadows the system <cpuid.h> that
#                   svs/cpuid.h includes; #pragma once then makes it a no-op and
#                   __cpuid is undeclared. Costs two inline svs::detail functions.
#   vendored trees  parsed transitively via -I where svs uses them; eve sweeps in
#                   ARM SVE headers that cannot compile on x86.
#   ivf             svs/index/ivf/common.h includes <mkl.h> unguarded and no tarball
#                   ships MKL headers, so IVF is compared at symbol level only.
exclude_args=()
# Leading-* so each pattern matches the full path, not one relative to HEADER_SUBDIR.
for pattern in '*svs/core/core.h' '*svs/lib/lib.h' '*svs/cpuid.h' \
               '*svs/index/ivf/*' '*svs/extensions/ivf/*' '*svs/orchestrators/*ivf*' \
               '*/eve-*/*' '*/fmt/*' '*/spdlog/*' '*/tsl/*' '*/toml++/*'; do
    exclude_args+=(--exclude-header "$pattern")
done

rc=0
# The AST frontend, C++20 standard and compile options come from .abicheck.yml;
# abicheck 0.6 dropped the equivalent flags, so do not reintroduce them here.
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

# Only 2, 4 and 5 are findings. Anything else -- an unparseable header, a
# fingerprint mismatch (16), a crash -- means the comparison did not happen, and
# must not be reported as an ABI break.
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
