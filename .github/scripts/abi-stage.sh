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

# Unpack two SVS build tarballs and resolve the paths abicheck/abicheck needs.
#
# This runs the comparison's inputs, not the comparison: the abicheck Action owns
# the toolchain, the invocation and the reporting. What is left here is the part no
# typed Action input can express, because it depends on what the tarballs contain.
#
# Usage: abi-stage.sh <old-tarball> <new-tarball>
#
# Overridable via environment:
#   LIBRARY        library basename to compare (default libsvs_runtime.so)
#   HEADER_SUBDIR  header root inside the tarball (default include/svs/runtime)
#   WORKDIR        scratch directory (default abi-work)
#
# Writes old-library, new-library, old-header, new-header, old-include, new-include
# and extra-args to $GITHUB_OUTPUT (stdout when run outside Actions).

set -euo pipefail

OLD_TARBALL="${1:?Usage: abi-stage.sh <old-tarball> <new-tarball>}"
NEW_TARBALL="${2:?missing new tarball}"

LIBRARY="${LIBRARY:-libsvs_runtime.so}"
HEADER_SUBDIR="${HEADER_SUBDIR:-include/svs/runtime}"
WORKDIR="${WORKDIR:-abi-work}"
GITHUB_OUTPUT="${GITHUB_OUTPUT:-/dev/stdout}"

# Prints the resolved library path on success. Release tarballs and CI artifacts
# have differed on whether the unversioned symlink is included, so accept the
# versioned real file too.
unpack() {
    local tarball="$1" dest="$2" side="$3" lib

    if [ ! -f "$tarball" ]; then
        echo "::error::$side: tarball not found at $tarball" >&2
        return 1
    fi

    rm -rf "$dest"
    mkdir -p "$dest"
    tar -xzf "$tarball" -C "$dest"

    lib=$(find "$dest" -name "$LIBRARY" -o -name "$LIBRARY.*" | sort | head -1)
    if [ -z "$lib" ]; then
        echo "::error::$side: $LIBRARY not found in $tarball" >&2
        return 1
    fi
    if [ ! -d "$dest/$HEADER_SUBDIR" ]; then
        echo "::error::$side: headers missing from $tarball ($HEADER_SUBDIR)" >&2
        return 1
    fi
    echo "$lib"
}

OLD_DIR="$WORKDIR/old"
NEW_DIR="$WORKDIR/new"

OLD_LIB=$(unpack "$OLD_TARBALL" "$OLD_DIR" old)
NEW_LIB=$(unpack "$NEW_TARBALL" "$NEW_DIR" new)

# Vendored deps ship under a versioned root (include/eve-2023.2.15/eve/...), so
# -I include alone cannot resolve their own `#include <eve/...>`.
#
# Only roots present on BOTH sides are added, in one sorted order: abicheck
# fingerprints the include sequence and refuses to compare (rc=16) when the two
# sides differ, so a root added to one side only voids the whole run. Tarballs do
# genuinely disagree about shipping eve, hence the intersection rather than a union.
old_includes="$OLD_DIR/include"
new_includes="$NEW_DIR/include"
for dir in "$OLD_DIR"/include/*-[0-9]*/; do
    [ -d "$dir" ] || continue
    vendored=$(basename "$dir")
    [ -d "$NEW_DIR/include/$vendored" ] || continue
    old_includes+=" $OLD_DIR/include/$vendored"
    new_includes+=" $NEW_DIR/include/$vendored"
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
#
# Passed through the Action's extra-args, which has no exclude-header input of its
# own and no config-file equivalent (sources.exclude scopes source collection, a
# different axis). Emitted unquoted on purpose: the Action appends extra-args with
# an unquoted `CMD+=($INPUT_EXTRA_ARGS)`, so quotes here would reach abicheck as
# literal characters -- and pathname expansion still applies, hence the guard.
#
# Leading-* so each pattern matches the full path, not one relative to HEADER_SUBDIR.
extra_args=
for pattern in '*svs/core/core.h' '*svs/lib/lib.h' '*svs/cpuid.h' \
               '*svs/index/ivf/*' '*svs/extensions/ivf/*' '*svs/orchestrators/*ivf*' \
               '*/eve-*/*' '*/fmt/*' '*/spdlog/*' '*/tsl/*' '*/toml++/*'; do
    # This cwd is the Action's cwd too, so a match here is a match there.
    # shellcheck disable=SC2086  # deliberately reproducing the Action's expansion
    set -- $pattern
    if [ "$#" -ne 1 ] || [ "$1" != "$pattern" ]; then
        echo "::error::exclude pattern $pattern matches paths under $PWD, so the" \
             "Action's extra-args expansion would glob it into argv instead of" \
             "passing it through. Move the match out of the workspace." >&2
        exit 1
    fi
    extra_args+=" --exclude-header $pattern"
done

{
    echo "old-library=$OLD_LIB"
    echo "new-library=$NEW_LIB"
    echo "old-header=$OLD_DIR/$HEADER_SUBDIR"
    echo "new-header=$NEW_DIR/$HEADER_SUBDIR"
    echo "old-include=$old_includes"
    echo "new-include=$new_includes"
    echo "extra-args=${extra_args# }"
} >> "$GITHUB_OUTPUT"

echo "Staged $LIBRARY: $OLD_LIB -> $NEW_LIB"
