# Rolling graph repair: what it does, why it works, and why it is the right shape

This is the short form of the case for the change. `M5-VERDICT.md` holds the measured verdict and
its blind spots; `MEASUREMENT-HISTORY.md` holds the earlier rounds and what they falsified;
`recipes/` holds the runnable sweeps. Read this one first.

## The goal, and why the current design cannot meet it

A `MutableVamanaIndex` deletes lazily. A delete marks a slot, and the graph keeps its edges into
that slot until `consolidate()` walks the node range and rewrites every adjacency list that points
at a deleted node. Only then is the slot reclaimable. That pass is stop-the-world: it takes the
index's exclusive lock, so no search runs for its duration.

The consequence is a search tail that has nothing to do with search. On the measured rung — 100k
Cohere vectors, 8 searcher threads, 400 searches/s offered, one write per search — the median
search is around 3 ms and the p99 is 222 ms, with a 321 ms worst case. Two orders of magnitude
between median and p99, and the gap is the repair schedule showing through the lock. A serving
deployment cannot tune its way out of this: repairing less often makes each episode longer, and
repairing more often multiplies the number of episodes. Either way a query that arrives during a
sweep waits for the whole sweep.

The goal is therefore a narrow one: cut the search tail under mutation, without giving up search
throughput and without spending memory to do it.

## What was built

A cursor over the node range, and one method that advances it:

```
consolidate_slice(batch_size)   // repair [cursor, cursor + batch_size), advance the cursor
```

When the cursor reaches the end of the range it wraps, and that wraparound — one full revolution —
is the moment the previous revolution's deletions become reclaimable. `consolidate()` keeps its
existing meaning: a full sweep, which resets the cursor and the deletion counter, so the default
path is unchanged and the mechanism is inert until a caller reaches for it.

The index owns no scheduler and no thread. The caller decides when to call, which is what lets the
same mechanism serve the three cadences an embedder can actually offer: at garbage collection, on
every mutation, or every N mutations. State added per index: one cursor, one counter. State added
per node: none.

## Why this achieves the goal

The argument is about episode length, not about work.

A blocking repair contributes to the search tail in proportion to how long one episode lasts, not
how much repair happens in total. Under open-loop arrivals, a stall of duration D delays every
query that arrives during it, and the worst of those waits very nearly D. So the tail a repair
schedule imposes is set by the *longest single episode*, while the amount of repair is set by the
*product* of episode length and episode frequency.

Those two are independent, and that is the whole lever. Splitting one sweep of N nodes into k
slices of N/k, each fired k times as often, leaves total repair work identical and divides the
worst blocking episode by k. Nothing is traded away: the same edges get rewritten, in the same
order, under the same lock, just in shorter turns.

The measurement was designed to test exactly that and nothing else. The candidate repairs 1/10 of
the range every 100 mutations — one cursor revolution per 1000 mutations, which is the same repair
coverage per mutation as the baseline's full sweep every 1000. Equal work, different episode
length. Search p99 falls from 222.3 ms to 54.4 ms, a factor of 4.1, and the deep tail falls by
about the same factor at both mandated write:search mixes. Update p99 falls 3.7×, because a mutator
waiting behind a repair is waiting behind the same shortened episode. Memory is byte-identical to
the baseline, to the byte, because there is no per-node metadata to pay for. Saturation throughput
is within 1%, and the hot-loop change on its own — reading adjacency lists through atomic loads —
costs nothing measurable at 1826 against 1821 searches/s.

The honest boundary: at the read-heavy mix neither implementation is under repair pressure and the
p99s are indistinguishable, with the improvement showing only in the deeper tail. The change helps
where the problem exists.

## Why this is the right shape, and not one of the alternatives

**Against a reverse-edge index.** The textbook way to make repair local is to store, per node, the
list of nodes pointing at it, so a delete can be repaired by touching only its in-neighbours. It
works, and it costs memory proportional to the graph on a structure that already dominates the
footprint — against a criterion that asks for footprint to grow as little as possible. The cursor
buys the same locality in *time* instead of in *space*, and the space it costs is two words per
index. That is why the measured memory delta is zero rather than small.

**Against a second index type.** The incumbent `include/svs/concurrent/` reaches a 1.44 ms search
p99 and will keep that crown; nothing that blocks searchers at all competes with it on that axis.
What it costs is a duplicated index implementation to maintain, roughly 1.6% more memory, an update
p99 an order of magnitude worse than the candidate's, and a teardown that aborts with
`Unmap failed!` at 500k points. This change is 255 insertions across four existing headers, adds no
type, and leaves one code path to maintain.

**Against a background repair thread.** Putting repair on its own thread inside the index makes the
index own a thread and a policy, and it still blocks searchers for whatever episode length it
chooses. The cursor is orthogonal to where repair runs: it shortens the episode whoever calls it.
An embedder that already has a maintenance thread can drive the cursor from it.

**Against doing nothing and tuning `repair_every`.** This is the alternative worth stating plainly,
because it is free. It does not work, for the reason in the section above: tuning the existing knob
moves episode length and episode frequency together, in opposite directions. The cursor is what
makes them independent.

The design also declines a generalisation it could easily have made. There is no policy object, no
strategy interface, no scheduler abstraction — the caller passes a batch size and that is the entire
contract. A single caller does not justify an interface, and the two earlier attempts at this change
that were abandoned both foundered on machinery, not on the idea.

## The one thing a reader must not skip

The p99 result requires open-loop load, and this is a real limit rather than a caveat. Under
closed-loop saturation — each thread issuing its next query only after the previous returns —
nothing can fall behind a schedule, and the p99 comparison inverts: 35.1 ms for the candidate
against 5.2 ms for the baseline, while the deep tail still favours the candidate at 64.8 against
211.6 ms. A serving system with independent arrivals is open-loop and sees the 4.1× improvement. A
benchmark harness spinning in a loop is closed-loop and will report the opposite ranking.

Related, and the reason every number in `MEASUREMENT-HISTORY.md` is marked superseded: the harness
originally measured service time, starting each operation's clock after its pacing sleep. That is
coordinated omission, its bias grows with stall length, and it therefore flattered the
stop-the-world baseline specifically. It had the sign of the headline comparison backwards.

## Low-hanging fruit

1. **Size the batch by accumulated deletions instead of by node count.** Round 4 in
   `MEASUREMENT-HISTORY.md` found that a repair call's cost tracks how many deletions accumulated
   since that sub-range was last swept, not the sub-range's width — which is why narrowing the
   slice tenfold changed nothing. The index already holds the deletion counter the principled knob
   needs. This is arithmetic inside `consolidate_slice`, and it would remove the sharpest edge in
   the current design: the operator picks coverage blind, and picking it too fine starves writes.
2. **Fix the leaked `spdlog` sink in `tests/svs/index/vamana/index.cpp`.** A lambda registered by
   the per-index logging test outlives the test and is still a sink when a later case emits a
   legacy-config warning, so the suite SIGSEGVs and roughly 50 registered cases never run. This is
   pre-existing and reproduces on the pristine baseline, which means "full suite green" is currently
   not demonstrable for *any* change to this repository. It looks like a few lines, and fixing it
   unblocks a gate every future change has to argue past.
3. **Sweep searcher count.** Everything measured is at 8 searcher threads. The baseline's
   stop-the-world episode should penalise it further as readers are added, so the 4.1× is probably a
   floor rather than a ceiling. The recipe already sweeps this dimension; it is one run, not a code
   change.
4. **Confirm and consider defaulting `--reuse-empty`.** It improved search p99 from 54.4 to 49.1 ms
   at no measured memory cost. That was one observation and it is unexplained, which is exactly why
   it is cheap to check rather than cheap to adopt.
5. **Point the same cursor at `compact()`.** Compaction remains a separate stop-the-world cost that
   this change does not touch, and a deployment that compacts on a timer still pays it in full. The
   prerequisite is already done here: `compact()` used to throw when it ran with unrepaired
   deletions outstanding, and that is fixed on this branch. This is the largest of the five and the
   least hanging; it is listed because it is the obvious next increment of the same idea.
