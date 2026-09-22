#!/usr/bin/env bash
# Sourced by the driver scripts, never executed. Every path arrives as a command-line argument:
# a site-specific path committed to this repository would be a policy violation.

TRAIN=
TEST=
OUT=
CANDIDATE=
RELEASE=
INCUMBENT=
HARNESS_COMMON=()

# `usage` is supplied by the sourcing script, which knows which of these options it needs.
parse_common_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --train) TRAIN="$2"; shift 2 ;;
            --test) TEST="$2"; shift 2 ;;
            --out) OUT="$2"; shift 2 ;;
            --candidate) CANDIDATE="$2"; shift 2 ;;
            --release) RELEASE="$2"; shift 2 ;;
            --incumbent) INCUMBENT="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
        esac
    done
}

# Takes variable names rather than values, so a missing option can be named in its command-line
# spelling: require_args TRAIN CANDIDATE.
require_args() {
    local missing=0 name
    for name in "$@"; do
        if [ -z "${!name}" ]; then
            echo "missing required option: --$(echo "$name" | tr '[:upper:]_' '[:lower:]-')" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        usage
        exit 2
    fi
}

require_readable() {
    local path
    for path in "$@"; do
        if [ ! -r "$path" ]; then
            echo "cannot read: $path" >&2
            exit 1
        fi
    done
}

require_executable() {
    local path
    for path in "$@"; do
        if [ ! -x "$path" ]; then
            echo "not an executable: $path" >&2
            exit 1
        fi
    done
}

set_harness_common() {
    HARNESS_COMMON=(
        --train "$TRAIN" --test "$TEST"
        --build-size 100000
        --num-index-threads 8
        --graph-max-degree 64
        --search-window-size 30
        --searcher-sweep 8
        --warmup-seconds 10
        --window-seconds 30
        --num-windows 2
        --compact-every-seconds 0
    )
}

run_case() {
    local name="$1" bin="$2"
    shift 2
    local log="$OUT/${name}.log"
    if [ -s "${log}.gz" ]; then
        echo "=== $name done, skipping ==="
        return
    fi
    echo "=== $name starting $(date -Is) ==="
    local start=$SECONDS
    numactl --cpunodebind=0 --membind=0 "$bin" "${HARNESS_COMMON[@]}" "$@" > "$log" 2>&1
    # Captured on its own line: a command substitution in the banner below overwrites $?, which
    # made an earlier version of these scripts report every run as successful.
    local rc=$?
    echo "=== $name exit=$rc wall=$((SECONDS - start))s $(date -Is) ==="
    gzip -f "$log"
}
