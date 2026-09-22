# Measurement recipes

Every sweep here is sized to finish inside 60 minutes on one machine. Absolute latencies from a
100k-point index do not transfer to production scale; the ratios between implementations at a fixed
offered load are what these recipes establish.

## What each script answers

| Script | Question | Wall clock |
|---|---|---|
| `run-micro.sh` | Search and update tail latency, candidate versus both baselines, at the two mandated write:search mixes | 27 min |
| `run-controls.sh` | Is the default path unchanged when the rolling cursor is unused, and does empty-slot reuse change the footprint | 12 min |
| `run-qps.sh` | Saturation throughput, to catch a hot-loop regression that a paced run cannot see | 10 min |
| `summarize-micro.py` | Aggregates search p99/p99.9/max per configuration from the gzipped logs | — |

## Invocation

No script contains a path. Datasets, harness binaries and the output directory are all arguments,
and a missing one is a usage error rather than a silent fallback:

```
./run-micro.sh --train <dir>/cohere-1M_train.fvecs \
               --test <dir>/cohere-1M_test.fvecs \
               --out <results dir> \
               --candidate <candidate build>/examples/cpp/concurrency_latency_harness \
               --release <release build>/examples/cpp/concurrency_latency_harness \
               --incumbent <incumbent build>/examples/cpp/concurrency_latency_harness

./run-controls.sh --train ... --test ... --out ... --candidate ... --release ...
./run-qps.sh      --train ... --test ... --out ... --candidate ... --release ...

python3 summarize-micro.py '<results dir>/*.log.gz'
```

`--incumbent` is required only by `run-micro.sh`. Pass `--help` to any script for its own list.
Shared argument handling and the per-case runner live in `common.sh`, which is sourced and not run.

Each script skips a case whose compressed log already exists, so an interrupted sweep resumes by
re-invoking it with the same arguments. `run-micro.sh` additionally abandons its remaining cases at
the 60-minute mark rather than overrunning the budget; the skip guard lets a second invocation
finish the matrix.

## Three trees are required

The candidate is this tree. `run-controls.sh` and `run-qps.sh` additionally need a build of the
release baseline at the pre-change commit, and `run-micro.sh` also needs the incumbent
(`include/svs/concurrent/`). Each tree needs its own build directory and its own copy of the
harness, because the harness differs per tree: the incumbent's port drops `SearchExclusion`
entirely, and the release baseline's accepts only `--repair-mode full`.

## Two things to get right, or the numbers mislead

**Offered load must be absolute.** Pass `--search-rate` and `--write-ratio`, never
`--target-ratio`. Relative pacing is unreachable against a fast implementation, whose searches
outrun what one mutator can match, so it silently changes the workload per implementation instead of
holding it fixed. Check the `ACHIEVED RATES` line in every log: a configuration that did not deliver
its requested load produced a throughput result, not a latency result.

**Latency must be open-loop.** The harness measures from each operation's intended arrival time.
Measuring from after the pacing sleep is coordinated omission and inverts the comparison, because it
records a stall only against the queries already in flight and not against the ones whose turn
passed while the index was busy. A closed-loop benchmark cannot show this effect at all and will
rank the implementations the other way round; see the blind spots in `M5-VERDICT.md`.
