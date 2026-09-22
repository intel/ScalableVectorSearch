# M5 — Proxy verdict: rolling repair versus both baselines

Last updated 2026-09-21. Index and harness measured at `3952bf8`; the release baseline at
`5717f685`; the incumbent at `43c77ad1` (`dev/eglaser-finegrain-dmitry-copy`).

## A correction that invalidates every latency number reported before this document

Earlier results in this record, including the AR2/AR5/AR7 sweeps, measured **service time**: the
harness paced each operation to its scheduled arrival, slept until that moment, and only then
sampled the start clock. That is coordinated omission. When repair blocks searchers, only the
queries already in flight record a long latency; every query whose turn passed during the stall
starts its clock late and reads as fast. The bias is not uniform — it grows with stall *length* —
so it systematically flatters an implementation whose stalls are rare and long over one whose
stalls are frequent and short. That is exactly the comparison this work exists to make, and the
earlier numbers got its sign wrong.

Under service time the candidate appeared to **lose** p99 to the release baseline by 9× (50 ms
against 5.7 ms). That conclusion was wrong. Latency is now measured from each operation's intended
arrival time, and the ordering reverses.

## The rung

100k-point Cohere vectors, 8 index threads, degree 64, search window 30, 8 searcher threads, two
30-second windows after 10 seconds of warmup, two replicates, one NUMA node, compaction disabled.
Offered load is absolute and identical across implementations: 400 searches/s, with writes at
100 per 100 searches (`100:100`) and 10 per 100 searches (`10:100`), the two mixes §4.2 mandates.
Every configuration delivered its requested load to within 0.05%, so these are latency results and
not disguised saturation results.

Configurations. **Candidate** repairs `1/10` of the node range every 100 mutations, which is one
cursor revolution per 1000 mutations — the same repair coverage per operation as the release
baseline, so a latency difference is the length of a repair episode and not the amount of work
done. **Candidate (fine)** repairs the same slice every 10 mutations, i.e. 10× the coverage.
**Release** sweeps the whole graph every 1000 mutations. **Incumbent** is
`include/svs/concurrent/`, whose repair takes `compact_mutex_` shared and so blocks neither
readers nor writers.

## Search latency, response time, milliseconds (median over 4 window-samples)

| | p99 `100:100` | p99.9 | max | p99 `10:100` | p99.9 | max |
|---|---|---|---|---|---|---|
| Candidate | **54.4** | 73.4 | 88.9 | 4.88 | 35.5 | 48.7 |
| Candidate (fine) | **27.8** | 47.8 | 56.2 | 5.31 | 17.9 | 38.9 |
| Release | 222.3 | 298.3 | 321.1 | 4.69 | 141.8 | 163.6 |
| Incumbent | 1.44 | 1.59 | 1.84 | 1.29 | 1.45 | 1.62 |

## Update latency, throughput, memory

| | update p99 `100:100` | update p99 `10:100` | delivered searches/s | delivered writes/s | total bytes |
|---|---|---|---|---|---|
| Candidate | **62.7 ms** | 4.0 ms | 400.0 | 399.8 / 40.0 | 1 353 165 892 |
| Candidate (fine) | 816.8 ms | 4.2 ms | 400.0 | **378.4** / 40.0 | 1 353 165 892 |
| Release | 230.9 ms | 4.3 ms | 400.0 | 400.0 / 40.0 | 1 353 165 892 |
| Incumbent | 630.2 ms | 500.7 ms | 400.0 | 400.0 / 40.0 | 1 374 389 000 approx. |

The fine configuration buys its better search tail by starving writes: it delivers 378 of 400
requested writes/s and its update p99 is 817 ms, with a median of 191 ms, meaning the mutator is
chronically backlogged. It is not a viable setting and is reported only to show the shape of the
tradeoff. Everything below refers to the equal-coverage candidate.

## The three §4.2 criteria

**Superior p99 search latency — met against the release baseline at the write-heavy mix, at parity
at the read-heavy mix.** 54.4 ms against 222.3 ms is a 4.1× improvement at `100:100`. At `10:100`
the two are indistinguishable (4.88 against 4.69 ms, with the candidate's replicate range
4.76–9.45 ms overlapping the release's 4.58–4.86 ms); the deeper tail still improves 4× (35.5
against 141.8 ms). The honest statement is that the criterion is met where repair pressure exists
and neither implementation is stressed where it does not. **Not met against the incumbent**, which
is 38× better at 1.44 ms and is not beaten on this axis by anything that blocks searchers at all.

**Throughput at least on par — met.** A paced run cannot detect a hot-loop regression, and the
change reads neighbour lists through `std::atomic_ref` unconditionally on the default search path,
which §5 names as the dangerous kind of cost. Measured unthrottled, the candidate configured as the
release delivers 1826.4 searches/s against the release's 1821.6, with per-sample spreads of
1771–1871 and 1762–1869 that overlap almost exactly: the atomic loads cost nothing measurable. The
proposed slice configuration delivers 1803.5 searches/s, 1.0% below the release and within its own
spread, which is the cost of interleaving repair rather than of the hot loop.

**Memory increased as little as possible — met, with zero increase.** The candidate's total is
byte-identical to the release baseline's, because the mechanism adds no per-node metadata at all:
one cursor and one counter per index, against the reverse-edge list per node that §5 identifies as
the usual cost of making repair local. The incumbent carries roughly 1.6% more. Enabling
`--reuse-empty` changed the total not at all at this scale and improved search p99 slightly
(49.1 against 54.4 ms).

## Controls

**Null control.** The candidate binary configured exactly as the release — full sweep at the same
cadence — gives search p99 194.8 and 192.2 ms against the release's 192.7 and 192.7 ms, a 1.1%
spread with no consistent sign, and byte-identical memory. The added machinery is inert when the
rolling cursor is not used.

**Codegen.** The per-architecture distance libraries are byte-for-byte identical between the two
builds, but that is not evidence about the search path: greedy search is header-only and
instantiated in the consumer, and those objects contain no search code at all. The hot-loop
question is settled by the saturation measurement above, not by those hashes. Note that
`greedy_search` now reads its adjacency list through `get_node_atomic` unconditionally and iterates
it by index, because the atomic view deliberately exposes no iterators; the measurement says this
costs nothing, but it is a real change to the default path and not a dormant one.

**Test suite.** Two failures, neither attributable to this change. One is a leaked `spdlog` sink:
a lambda registered by the per-index logging test in `tests/svs/index/vamana/index.cpp` outlives
that test, and when upstream's `VamanaIndexParameters::load_legacy` later emits its legacy-config
warning, `spdlog` dispatches to the stale callback and dereferences a dangling capture. The whole
stack sits in test logging infrastructure and upstream loader code; this branch modifies neither
file, and the case passes in isolation. The other is `cannot remove: Directory not empty` on the
shared `data/temp` directory, which appeared in one run of two and not the other.

Both are confirmed pre-existing: the pristine baseline's suite, built from `5717f685`, fails at the
same two sites — `tests/svs/core/io/binary.cpp:138` and `tests/svs/index/vamana/index.cpp:95` — with
the same SIGSEGV and the same exit 139. The crash aborts the process, so neither tree runs its last
~50 registered cases. **This means "full suite green" cannot be demonstrated on this codebase at
all**, by this change or by anything else, until that leaked sink is fixed. The M4 gate is therefore
argued as parity with the baseline rather than as a green suite, and the untested tail of the suite
is a genuine blind spot in that argument.

## Fidelity and blind spots

- **The p99 win requires open-loop load, and that is a real limit on the claim.** Under closed-loop
  saturation, where each thread issues its next query only after the previous one returns, nothing
  can fall behind a schedule, and the trade inverts: the slice configuration shows p99 35.1 ms
  against the release's 5.2 ms, while still winning the deep tail at 64.8 against 211.6 ms. A
  serving system with independent arrivals is open-loop and sees the 4.1× p99 improvement; a
  benchmark harness spinning in a loop is closed-loop and will report the opposite. Any reviewer
  comparing against a closed-loop QPS benchmark must be told this, or the result will look wrong.
- **Absolute latencies do not transfer.** A 100k-point index is two to three orders of magnitude
  below production scale, and repair episode length scales with the node range being swept. The
  *ratios* between implementations at a fixed offered load are what this rung establishes.
- **One searcher count.** Everything here is at 8 searcher threads. The tail's dependence on
  reader concurrency is unmeasured, and the release baseline's stop-the-world episode should
  penalise it further as readers are added, so this likely understates the candidate's advantage.
- **Compaction is disabled.** `compact()` is a separate stop-the-world cost that this change does
  not address; a deployment that compacts on a timer still pays it.
- **Repair episode length is a tuning result, not a bound.** The candidate's p99 is roughly the
  duration of one slice. Nothing in the design guarantees a slice is short; the operator chooses
  coverage per operation, and choosing it too fine starves writes, as the fine configuration shows.
- **The incumbent's teardown aborts** with `Unmap failed!` at 500k points, though not at this
  scale. Its measurement data is complete before the abort. That is a defect in the code this
  change would let a deployment avoid needing.

## The four paragraphs

**What the problem was.** A dynamic Vamana index repairs its graph in `consolidate()`, a
stop-the-world pass over every node that runs at garbage-collection cadence. Every query that
arrives during it waits for all of it. At the write-heavy mix on this rung that is a 222 ms p99 and
a 321 ms worst case against a 3 ms median — the tail is not a property of the search, it is the
repair schedule showing through.

**What was built.** A cursor over the node range, and one method that advances it across a slice.
The index owns no scheduler and no thread: the caller decides when to call, which fits the three
cadences an embedder can actually offer — at garbage collection, every operation, or every N
operations. Deletions become reclaimable when the cursor completes a revolution rather than at the
end of a sweep. No reverse-edge index, no per-node metadata, no second index type: 255 insertions
across four headers, which is what the two previously rejected attempts could not manage.

**What was measured.** At equal repair coverage and identical offered load, search p99 falls 4.1×
at the write-heavy mix and the deep tail falls 4× at both mixes; update p99 falls 3.7× against the
release baseline and is 10× better than the incumbent's; memory is byte-identical to the release
baseline; saturation throughput is within 1%, with the hot-loop change itself costing nothing
measurable; and the candidate configured as the release reproduces it to within 1% on latency and
byte-for-byte on memory. Getting here required correcting the harness, which had been measuring
service time and so had the sign of the p99 comparison backwards.

**Why a maintainer should prefer this.** It moves the tail without buying anything with memory or
with a second implementation of the index. The incumbent has a better search tail and will keep it,
but it costs a duplicated index type, more memory, a worse update tail, and a teardown defect; this
change costs one cursor and one method on the existing type, and it is inert until a caller asks
for it.
