#!/usr/bin/env bash
# Saturation throughput, ~10 minutes. The paced runs cannot see a hot-loop regression because
# they deliver a fixed offered load; this one removes the throttle so QPS is the output.
#
# nullsat isolates the cost of reading neighbours through atomic loads: the candidate is
# configured exactly as the release, so the repair schedule is identical and only the hot loop
# differs. slicesat is the configuration actually proposed.
set -u

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat >&2 <<'EOF'
Usage: run-qps.sh [OPTIONS]

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
echo "=== qps driver started $(date -Is) ==="
cat "$TRAIN" "$TEST" > /dev/null

for rep in 1 2; do
    run_case "nullsat_candidate_rep${rep}" "$CANDIDATE" --repair-mode full --repair-every 1000
    run_case "nullsat_release_rep${rep}" "$RELEASE" --repair-mode full --repair-every 1000
    run_case "slicesat_candidate_rep${rep}" "$CANDIDATE" --repair-mode slice --slice-k 10 --repair-every 100
done

echo "=== qps driver finished $(date -Is) ==="
