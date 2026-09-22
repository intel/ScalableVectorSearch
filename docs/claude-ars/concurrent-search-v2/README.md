# Concurrent search under a mutating index: rolling graph repair

The design record for the change that spreads Vamana's `consolidate()` repair across runtime
instead of paying it as one stop-the-world pass. Four documents, in reading order.

| Document | What it holds |
|---|---|
| `SUMMARY.md` | The case for the change: the mechanism, why it is this shape and not a reverse-edge index or a second index type, and the low-hanging fruit left over. Start here. |
| `M5-VERDICT.md` | The measured verdict against both baselines, criterion by criterion, with the controls and the blind spots. |
| `MEASUREMENT-HISTORY.md` | The four earlier measurement rounds: what each asked, what it falsified, and the two methodological errors they contained. |
| `recipes/` | The runnable sweeps that produce the verdict's numbers, and how to read their output. |

Two facts that govern how every number here should be read, both established in `M5-VERDICT.md`:

**Latency is response time, measured from each operation's intended arrival.** Measuring from after
the pacing sleep is coordinated omission, and because its bias grows with stall length it flatters a
stop-the-world baseline specifically. Every latency number in `MEASUREMENT-HISTORY.md` predates that
correction and is marked superseded there.

**The p99 improvement requires open-loop arrivals.** Under closed-loop saturation the p99 comparison
inverts while the deep tail still favours the candidate. A reviewer comparing against a closed-loop
QPS benchmark will see the opposite ranking and should be told why.

No file in this directory contains a machine-local path. The measurement scripts take every path —
datasets, harness binaries, output directory — as a command-line argument; see `recipes/README.md`.
