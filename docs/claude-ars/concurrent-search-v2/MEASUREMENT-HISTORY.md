# Measurement history — concurrent search v2 (AR2, AR5, AR7, AR7b)

This document replaces `AR2-RESULTS.md`, `AR5-RESULTS.md`, and `AR7-RESULTS.md`. It
records what four rounds of measurement on the concurrency latency harness
(`examples/cpp/concurrency_latency_harness.cpp`) established about spreading Vamana's
`consolidate()` repair across runtime instead of paying it as one stop-the-world pass,
and what each round abandoned or falsified along the way.

**Every absolute latency number reported by these four rounds is invalid and is not
reproduced here.** `M5-VERDICT.md` found that the harness measured service time: it
slept until each operation's scheduled arrival and only started the clock then, which is
coordinated omission. The bias grows with stall length, which flattered the stop-the-world
baseline specifically, and correcting it reversed the headline p99 comparison. Where a
latency figure below survives that invalidation, it is marked as service-time and
superseded and the point being made does not depend on its scale — for example, that a
write ratio was never delivered, or that an arm was abandoned because it starved writes.
No latency table from any of the four rounds is repeated here.

## Round 1 (AR2, commit 777fa619) — does the stall show up at all under mixed load?

The question was whether a harness built around one `std::shared_mutex`
(`SearchExclusion`, shared for search, exclusive for mutation/consolidate/compact) makes
the intended stall visible: search and mutation should go to zero completions for a
repair's whole duration. That part held — the harness's time-series buckets show clean
zero-completion windows spanning each repair.

Two things did not match the design brief and mattered more than the stall itself:

- **The `shared_mutex` baseline starves the writer outright, not just delays it.**
  Calibration at small scale showed mutator throughput collapsing by over 90% going from
  1 to 2 concurrent searcher threads, and to near-zero at 4 and 8. At the intended
  500k-vector, 16-searcher scale the same pathology showed up in slow motion: mutation
  throughput ran fast for the first ~40 seconds of a 240-second window, then collapsed to
  a trickle for the rest, with no further repair firing. 15 seconds of warmup was not long
  enough to reach a steady state, and the measured window mixed two throughput regimes.
  The likely mechanism — glibc's reader-preferring `pthread_rwlock` letting new readers
  cut in front of a waiting writer under continuous shared-lock churn — was not confirmed
  by reading glibc's source; it is inference from the symptom, not a verified cause.
- **`compact()` throws `Couldn't find key.`** when it is not called immediately after
  `consolidate()`. Root cause, read from `include/svs/index/vamana/dynamic_index.h`:
  `compact()` builds its old-to-new id remap from `nonmissing_indices()`, which — despite
  its own doc comment claiming to include soft-deleted entries — actually restricts to
  `Valid` slots only. `compact()` then remaps every `Valid` node's adjacency list, and any
  `Valid` node can still have an edge to a node that was soft-deleted but not yet
  consolidated; the remap lookup for that neighbor throws. Under continuous churn some
  node is soft-deleted at nearly every instant except the brief window right after a
  `consolidate()` call, so scheduling `compact()` on an independent wall-clock timer — the
  brief's own design — hits this almost every time under load. This was treated as a
  found defect, not worked around: every measurement round from here on ran with
  compaction disabled (`--compact-every-seconds 0`), so **compaction contributes no data
  to any of these four rounds.**

A third finding concerned the write:search ratio mechanism itself, which throttled the
mutator relative to measured search throughput. It never bound: achieved write rate came
in two to three orders of magnitude below target at both configured ratios, and — the
tell — the *less* restrictive nominal ratio (100 writes:100 searches) achieved a *lower*
write rate than the stricter one (10:100). That is the fingerprint of the starvation
finding above, not a ratio effect, and it made the two required runs uncontrolled samples
of the same starvation regime rather than a comparison. This round also noted that the
harness's baseline blocks search on repair just as much as it blocks writes, contradicting
the background assumption that search is affected "far less"; under a single
shared-vs-exclusive lock there is no such asymmetry.

**What this round settled:** the stall mechanism is real and visible; the baseline lock
as specified has a starvation failure mode, not just a bounded-delay one; and
`compact()` cannot be exercised under churn without hitting a genuine pre-existing bug in
`dynamic_index.h`.

## Round 2 (AR5, commit 24ac882a) — does fixing the lock fix the ratio?

`SearchExclusion` was changed to wrap the `shared_mutex` in a turnstile: a plain `mutex`
that every search and mutation call must acquire and release before contending for the
`shared_mutex`, so a waiting mutator blocks new searchers from even attempting the shared
lock. This bounds the mutator's wait by one round of in-flight searches rather than by
reader arrival rate.

Three replicates per ratio (six separate process runs, each with its own fresh build) were
used instead of the harness's own in-process replicate loop, so that `uptime` could be
polled at a clean process boundary immediately before and after each timed run, per the
task's machine-hygiene requirement.

The fix resolved the starvation pathology cleanly at 10:100 (achieved ratio converged to
within 0.001 of target, reproducibly across all three replicates) but not at 100:100
(achieved ratio held steady around 12.5:100 across all three replicates, a consistent
miss, not noise). The mechanism for the 100:100 miss is different in kind from Round 1's
starvation: repairs now fired on a steady cadence for the entire measurement window
instead of stalling out, so the mutator was not being denied the lock. The miss instead
tracked a mutator throughput ceiling — this single-mutator, single-lock design settled
around 800–850 update calls/second regardless of the ratio it was asked to hit, because
asking one mutator thread to match 16 searcher threads near 1:1 asks for a rate this
design was never going to reach. That explanation is inference from a repeated,
ratio-independent ceiling, not a confirmed isolated (zero-searcher) measurement.

A structural limitation surfaced here that later rounds addressed directly: because the
single mutator thread issues `consolidate()` itself, in-line, between its own mutation
calls, a mutation call can never queue behind a repair — it is the same thread doing both,
so there is nothing to wait for. This meant the harness's update-latency instrumentation
could not, by construction, show a repair's cost as part of update latency; the two were
structurally separated no matter how long a repair took.

**What this round settled:** the turnstile fix removes outright starvation; the residual
miss at high write pressure is a throughput ceiling of the single-mutator design, not a
lock-fairness defect; and the harness's own architecture (repair inline on the mutator
thread) hides repair cost from update-latency measurements by construction, which the next
round changed.

## Round 3 (AR7, commit 2777511a) — remove the throttle, move repair off the mutator thread

This round made three changes at once. It deleted the write:search throttle entirely,
since Round 2 showed a single sequential mutator can't be made to hit an arbitrary ratio —
the mutator now runs flat out and achieved ratio is reported as a measured output. It
swept searcher-thread count (1, 2, 4, 8, 16, 32, 64) against one index built once and
never rebuilt, three independent 120-second windows per point after a 30-second warmup.
And it moved `consolidate()` off the mutator thread onto a dedicated repair thread that
acquires the same `SearchExclusion` guard the mutator uses — no second lock — so that the
mutator's own update timer, which now starts *before* the guard is acquired, can finally
show repair-wait time in update latency.

A real defect was found and fixed in the harness itself along the way: `fmt::print`
writes through `stdout`, which becomes fully (not line) buffered once redirected to a file
under `nohup`. The first attempt at the full run produced an empty log for the first two
minutes even though the process was alive and printing internally — the buffer only
flushed once full. Fixed with `std::setvbuf(stdout, nullptr, _IOLBF, 0)`.

Pre-rework verification ran the existing test suite (`tests` target, filter
`[dynamic_index]`) at the prior commit: exit code 0, 123,793 assertions across 5 test
cases, all passed.

Non-latency findings, still valid:

- **Search throughput saturates between 32 and 64 searcher threads** — doubling thread
  count from 32 to 64 did not double measured QPS, consistent with the shared-lock
  `SearchExclusion` becoming the bottleneck at high concurrency.
- **Achieved write:search ratio falls monotonically as searcher count rises, by
  construction, not by defect** — with the throttle removed, the mutator's own achieved
  throughput stayed roughly flat (in a range of about 440–1020 update calls/second) while
  search throughput grew with thread count, so their ratio necessarily fell.
- **Repair call duration rose with searcher count up to 8 threads, then fell back from 16
  threads on**, while repair *count* fell monotonically throughout. The count trend is
  explained (mutator throughput falling under rising contention means fewer
  `repair-every` thresholds get crossed per window); the duration trend was not
  root-caused.
- **Update latency's rare/frequent split**: with repair now visible to update-latency
  instrumentation, only a tiny, stable fraction of updates (about 0.02%, independent of
  searcher count) were ever caught behind a repair at this cadence (`--repair-every
  5000`). That is small enough to show up in the maximum and in a dedicated slow-update
  count, but not in p99 or even p99.9 — a genuine methodological finding about where a
  rare-but-real tail event becomes visible at a given percentile, independent of the
  actual latency values involved.
- **Memory**: `graph_bytes` stayed exactly constant across the ~44-minute run (the
  blocked graph storage was sized for 500,000 live vertices at build time and the live
  count never changes, since every delete is paired with an insert). `data_bytes` grew
  roughly 3x (about 1.5 GiB to about 4.5 GiB): the blocked dataset storage only appends
  new blocks on insert and never frees a block vacated by a delete, and with compaction
  disabled throughout (per Round 1's finding) nothing ever reclaimed them — a real,
  measured memory-growth characteristic of running this workload uncompacted for tens of
  minutes, not a leak in the unreachable-memory sense. `metadata_bytes` grew a modest 12%,
  not root-caused. Separately, process-level `VmRSS`/`VmHWM` undercounts the index's true
  footprint because it excludes SVS's hugetlb-backed arenas; the index's own
  `get_memory_breakdown()` total is the figure to trust, not `VmRSS`/`VmHWM`.

**What this round settled:** removing the throttle and reporting achieved ratio as an
output is the right design; repair cost is now visible to update-latency instrumentation
but is rare enough at this cadence to appear only in the max and a dedicated slow-fraction
count, not in standard percentiles; and uncompacted churn grows the dataset's storage
substantially, which is an argument for compaction rather than evidence against this
change.

## Round 4 (AR7b, commits eeb4eb11 / 83060fbf) — does finer repair slicing help proportionally?

This round tested a specific premise: at a repair cadence fine enough to make the stall
show up in p99 (`--repair-every 100`, instead of Round 3's 5000), does a `consolidate_slice`
repairing a narrower sub-range of the graph reduce update-latency delay roughly in
proportion to how much narrower it is? Three arms at three searcher-count points (1, 8,
64): whole-index `consolidate()` ("full"), `consolidate_slice` at slice width
`ceil(n_nodes/10)` ("slice10"), and at `ceil(n_nodes/100)` ("slice100").

The premise was **not supported**. Full-to-either-slice was a large, real change in the
bulk of the update-latency distribution; slice10-to-slice100 — a further 10x change in
slice width — produced no measurable difference in the delayed-update fraction or in the
repair-call duration distribution at any of the three searcher-count points, and the two
slice arms' tails (p99.9, max) overlapped both each other's and full's, unlike their p99s.
The mechanistic evidence: slice10's average repair-call duration was within 1–2
milliseconds of slice100's at every point despite sweeping ten times as many nodes per
call, while both were consistently below full's average. The working explanation, stated
as inference and not confirmed by instrumenting `consolidate_slice` directly, is that a
slice's per-call cost is driven by how many deletions accumulated since the slice last
swept that sub-range, not by the sub-range's raw width — at `--repair-every 100` only
about 50 deletions accumulate between calls, and once a slice is wide enough to typically
contain most of them, narrowing it further buys nothing. This predicts K would start to
matter at a coarser `repair-every`; that was not tested.

Two robustness checks confirmed the three arms were comparable: achieved write:search
ratio was nearly identical across all three arms at every searcher count (within about
1.5% of each other), and end-of-run memory was byte-identical across all three arms,
confirming that repair strategy does not change how much the mutator inserts over a
fixed-duration run — `data_bytes` growth is purely a function of insert count, the same
uncompacted-growth mechanism Round 3 found.

**What this round settled:** slicing repair narrower than the point where it already
covers a cadence's typical deletion volume buys nothing further, at least at this
cadence; the "proportional to K" premise is falsified, while "slicing beats a whole-index
sweep" still holds. Only one `repair-every` value was tested, so the finding is scoped to
that cadence.

## What these rounds got wrong

**Coordinated omission.** All four rounds' latency instrumentation paced each operation to
its scheduled arrival time and only started the clock once that operation began — service
time, not response time. Because the bias grows with how long a stall lasts, it
systematically favored whichever implementation had rarer, longer stalls over one with
more frequent, shorter ones, which is exactly the comparison this whole effort exists to
make. `M5-VERDICT.md` corrected this by timing from each operation's intended arrival
instead, and the correction reversed the headline p99 comparison between the rolling-repair
candidate and the stop-the-world baseline. No percentile or maximum reported by any of the
four rounds above survives this correction as a result; only the qualitative and
non-latency findings do.

**The repair-coverage knob misreading.** The `--slice-k` flag sets a divisor of the node
count, not a node count — the width of a slice is `ceil(n_nodes / slice_k)`. It was first
read as if it specified the slice's width directly (i.e., as a node count), which made an
attempt at an equal-repair-coverage comparison between the candidate and a baseline look
like it differed by a factor of roughly 1000x in coverage when the two were actually
intended to match. This was caught and corrected before the M5 rung; anywhere `slice_k` or
"K" appears in the rounds above, it means the divisor. One repair call covers
`n_nodes / slice_k` nodes, so coverage per mutation is `n_nodes / (slice_k * repair_every)`
and it is that product, not `slice_k` alone, that has to match between two arms for a
comparison to hold repair work fixed.
