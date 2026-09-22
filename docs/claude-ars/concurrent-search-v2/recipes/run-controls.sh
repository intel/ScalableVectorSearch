#!/usr/bin/env bash
# Two controls for the M4/M5 gates, ~12 minutes total.
#
# null: the candidate binary configured exactly as the release (full sweep at GC cadence). If the
# added machinery is inert when unused, these numbers must match the release baseline within noise.
# reuse: the memory criterion with empty-slot reuse enabled, which changes insert cost and so
# cannot ride along in a run measuring anything else.
set -u

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat >&2 <<'EOF'
Usage: run-controls.sh [OPTIONS]

Required options:
  --train FILE        Training dataset (e.g. cohere-1M_train.fvecs)
  --test FILE         Test dataset (e.g. cohere-1M_test.fvecs)
  --out DIR           Output directory for logs
  --candidate BIN     Candidate harness binary
  --release BIN       Release baseline harness binary

Options:
  -h, --help          Show this help message and exit
EOF
}

parse_common_args "$@"
require_args TRAIN TEST OUT CANDIDATE RELEASE
require_readable "$TRAIN" "$TEST"
require_executable "$CANDIDATE" "$RELEASE"

set_harness_common

mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "=== controls driver started $(date -Is) ==="
cat "$TRAIN" "$TEST" > /dev/null

for rep in 1 2; do
    run_case "null_candidate_rep${rep}" "$CANDIDATE" --search-rate 400 --write-ratio 100 --repair-mode full --repair-every 1000
    run_case "null_release_rep${rep}" "$RELEASE" --search-rate 400 --write-ratio 100 --repair-mode full --repair-every 1000
    run_case "reuse_candidate_rep${rep}" "$CANDIDATE" --search-rate 400 --write-ratio 100 --repair-mode slice --slice-k 10 --repair-every 100 --reuse-empty
done

echo "=== controls driver finished $(date -Is) ==="
