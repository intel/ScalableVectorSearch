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
# One primitive serves both questions asked of it:
#   temporal    old = a release tarball, new = this build -- did we break users?
#   equivalence old and new are the same source revision built two ways -- did
#               the build paths diverge?
# The distinction is in what the caller passes, not in the logic, so this lives
# in one place. It is called from this repo's build-cpp-runtime-bindings.yml and
# from build-share-lib.yml in the innersource repo via the submodule path.
#
# Usage: abi-check.sh <old-label> <old-tarball> <new-label> <new-tarball>
#
# Overridable via environment:
#   LIBRARY        library basename to compare (default libsvs_runtime.so)
#   HEADER_SUBDIR  header root inside the tarball (default include/svs/runtime)
#   SUPPRESSIONS   suppression file (default .github/abi-suppressions.yml)
#   POLICY         abicheck policy (default strict_abi)
#   REPORT         markdown report path (default abi-report.md)
#   WORKDIR        scratch directory (default abi-work)
#
# Exit status is abicheck's: 0 compatible, 2 source-level API break, 4 binary
# ABI break, 64 invalid invocation. 77 is added to mean "an input was missing",
# which is not a finding and must never be reported as one.

set -euo pipefail

OLD_LABEL="${1:?Usage: abi-check.sh <old-label> <old-tarball> <new-label> <new-tarball>}"
OLD_TARBALL="${2:?missing old tarball}"
NEW_LABEL="${3:?missing new label}"
NEW_TARBALL="${4:?missing new tarball}"

LIBRARY="${LIBRARY:-libsvs_runtime.so}"
HEADER_SUBDIR="${HEADER_SUBDIR:-include/svs/runtime}"
SUPPRESSIONS="${SUPPRESSIONS:-.github/abi-suppressions.yml}"
POLICY="${POLICY:-strict_abi}"
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

rc=0
# Without pipefail the pipe into tee masks abicheck's exit status.
set -o pipefail
# castxml is not installed on the runners; clang is. `auto` does not fall back to
# clang on its own. If header parsing degrades, abicheck still reports from the
# ELF tier -- which is where vtable-size findings come from anyway.
abicheck compare \
    "$OLD_LIB" \
    "$NEW_LIB" \
    --header old="$OLD_DIR/$HEADER_SUBDIR" \
    --header new="$NEW_DIR/$HEADER_SUBDIR" \
    --include old="$OLD_DIR/include" \
    --include new="$NEW_DIR/include" \
    --version old="$OLD_LABEL" \
    --version new="$NEW_LABEL" \
    --ast-frontend clang \
    --gcc-options "-std=c++20 -include cstddef" \
    --policy "$POLICY" \
    "${suppress_args[@]}" \
    --format markdown | tee "$REPORT" || rc=$?

case "$rc" in
    0)
        echo "ABI compatible: $OLD_LABEL -> $NEW_LABEL"
        ;;
    64)
        echo "::error::abicheck rejected the invocation (exit 64) comparing" \
             "$OLD_LABEL -> $NEW_LABEL. This is a harness bug, not an ABI finding."
        ;;
    *)
        echo "::warning::ABI incompatibility: $NEW_LABEL differs from $OLD_LABEL (rc=$rc)"
        ;;
esac

exit "$rc"
