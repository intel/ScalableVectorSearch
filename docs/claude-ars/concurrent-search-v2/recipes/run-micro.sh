#!/usr/bin/env bash
# Microbenchmark sweep: whole matrix must finish inside 60 minutes, so the index is small and
# the windows are short. Absolute latencies do not transfer to index scale; only the comparison
# between configurations does.
set -u

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REPLICATES=2
DEADLINE_MINUTES=60

usage() {
    cat >&2 <<'EOF'
Usage: run-micro.sh [OPTIONS]

Required options:
  --train FILE        Training dataset (e.g. cohere-1M_train.fvecs)
  --test FILE         Test dataset (e.g. cohere-1M_test.fvecs)
  --out DIR           Output directory for logs
  --candidate BIN     Candidate harness binary
  --release BIN       Release baseline harness binary
  --incumbent BIN     Incumbent baseline harness binary

Options:
  -h, --help          Show this help message and exit
EOF
}

# ce is the arm the 500k comparison lacked: same repair coverage per operation as the release
# arm, so a latency difference is the episode length and not the amount of work done.
config_harness() {
    case "$1" in
        ce|cf) echo "$CANDIDATE" ;;
        a) echo "$RELEASE" ;;
        b) echo "$INCUMBENT" ;;
    esac
}

config_flags() {
    case "$1" in
        ce) echo "--repair-mode slice --slice-k 10 --repair-every 100" ;;
        cf) echo "--repair-mode slice --slice-k 10 --repair-every 10" ;;
        a) echo "--repair-mode full --repair-every 1000" ;;
        b) echo "--repair-every 1000" ;;
    esac
}

point_flags() {
    case "$1" in
        r100) echo "--search-rate 400 --write-ratio 100" ;;
        r10) echo "--search-rate 400 --write-ratio 10" ;;
    esac
}

parse_common_args "$@"
require_args TRAIN TEST OUT CANDIDATE RELEASE INCUMBENT
require_readable "$TRAIN" "$TEST"
require_executable "$CANDIDATE" "$RELEASE" "$INCUMBENT"

set_harness_common
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "=== micro driver started $(date -Is) on $(hostname) ==="
started=$SECONDS

cat "$TRAIN" "$TEST" > /dev/null

# Abandoning the remaining runs beats overrunning the budget: a partial matrix is still readable,
# and the skip-if-done guard lets a later invocation finish it.
over_deadline() {
    elapsed=$(( (SECONDS - started) / 60 ))
    if [ "$elapsed" -ge "$DEADLINE_MINUTES" ]; then
        echo "!!! ${elapsed}m elapsed, at or past the ${DEADLINE_MINUTES}m ceiling: stopping early"
        return 0
    fi
    return 1
}

for rep in $(seq 1 "$REPLICATES"); do
    for point in r100 r10; do
        for config in ce cf a b; do
            over_deadline && break 3
            log="$OUT/${config}_${point}_rep${rep}.log"
            if [ -s "${log}.gz" ]; then echo "=== $config $point rep $rep done, skipping ==="; continue; fi
            echo "=== $config $point rep $rep starting $(date -Is) ==="
            start=$SECONDS
            harness=$(config_harness "$config")
            # shellcheck disable=SC2046
            numactl --cpunodebind=0 --membind=0 "$harness" "${HARNESS_COMMON[@]}" \
                $(config_flags "$config") $(point_flags "$point") > "$log" 2>&1
            rc=$?
            echo "=== $config $point rep $rep exit=$rc wall=$((SECONDS - start))s $(date -Is) ==="
            if [ "$rc" -ne 0 ]; then
                echo "!!! $config $point rep $rep exit=$rc, last 5 lines:"; tail -5 "$log"
            fi
            gzip -f "$log"
        done
    done
done

echo "=== micro driver finished $(date -Is) after $(( (SECONDS - started) / 60 ))m ==="
