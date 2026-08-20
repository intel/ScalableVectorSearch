# `svs::index::vamana::concurrent` — concurrent dynamic Vamana index

A dynamic Vamana index that supports **lock-free search concurrent with mutation**:
searches, `add_points`, `delete_entries`, and `consolidate` may all be in flight at the
same time, from any number of threads.

This is a **separate index**. The pre-existing `svs::index::vamana::VamanaIndex` and
`svs::index::vamana::MutableVamanaIndex` are not modified, and neither is anything else
under `include/svs/`. The only pre-existing file this stack touches at all is
`tests/CMakeLists.txt`, to register the new tests.

## Why a separate index

The functionality being reproduced here originates in
[`razdoburdin:ScalableVectorSearch:seqlock`][seqlock], which implements it by editing
`MutableVamanaIndex` and its collaborators in place. That is invasive: the graph, the ID
translator, the search buffer, the pruning heuristics, `greedy_search`, and the build
driver all change shape, and the static index shares most of them. Reproducing the same
functionality as a parallel stack keeps the existing indexes bit-for-bit unchanged, so the
two can coexist and be compared.

[seqlock]: https://github.com/intel/ScalableVectorSearch/compare/main...razdoburdin:ScalableVectorSearch:seqlock

## How the separation works

Everything lives in `svs::index::vamana::concurrent`, a namespace nested inside the one it
shadows. C++ name lookup then gives a **delta layer** for free: a name declared inside
`concurrent` hides the same name in the enclosing namespace, while a name *not* declared
inside `concurrent` resolves to the enclosing namespace's. Each header here therefore only
needs to carry what actually changed; `VamanaSearchParameters`,
`VamanaBuildParameters`, `SearchScratchspace`, the `extensions` customization points, and
the rest are picked up from upstream unchanged.

Two consequences are easy to get wrong, and both are load-bearing:

1. **Do not redeclare an entity this stack does not change.** A redeclaration inside
   `concurrent` is a *distinct type* that merely looks identical, and values crossing the
   boundary stop converting. Redeclaring `GreedySearchPrefetchParameters`, for instance,
   makes `SearchScratchspace::prefetch_parameters` (an upstream type, reused verbatim)
   fail to bind. Use a `using`-declaration instead — see the block at the top of
   `greedy_search.h`, which aliases `GreedySearchPrefetchParameters`,
   `GreedySearchTracker`, `NullTracker`, `EntryPointInitializer`, and `NeighborBuilder`.

2. **Qualify intra-namespace calls.** Once those using-declarations make the upstream
   overloads visible, ADL on an upstream argument type makes an unqualified
   `greedy_search(...)` ambiguous. Calls inside this stack are written
   `concurrent::greedy_search(...)`.

## The concurrency design

### Sequence locks on adjacency lists

`lib::SeqLockCounter` / `lib::SeqLockArray` (`svs/lib/concurrency/seqlock.h`) give each
node an even/odd version counter. A writer bumps it odd, mutates, bumps it even; a reader
snapshots it (`read_begin`), reads, and re-checks (`read_validate`), retrying on
disagreement. `greedy_search` wraps its per-node neighbor expansion in that retry loop, so
searches take no locks at all. Stale neighbors from an invalidated read are harmless: they
carry valid ids and distances, and the search buffer dedupes by id.

Every adjacency slot is accessed through `SimpleGraphBase::relaxed_load` /
`relaxed_store`, which are relaxed `std::atomic_ref` operations. The *ordering* comes from
the sequence-lock counters, not from these accesses; relaxed atomics compile to plain
loads and stores on the platforms SVS targets, so the only thing they buy is the removal
of a formal data race — which is exactly what makes the TSan run below meaningful.

### Grow-stable storage

Appending must never relocate what a lock-free reader is holding. `lib::SegmentedVector`
(`svs/lib/segmented_vector.h`) is a two-level array whose existing elements keep stable
addresses across growth; it backs the per-node locks, the sequence counters, the slot
metadata, and the reverse-edge lists.

The dataset needs the same property. Rather than change `svs::data::Blocked`, this stack
adds an allocator tag `SegmentedBlocked<Alloc>` and an **additive** partial specialization
`svs::data::SimpleData<T, Extent, SegmentedBlocked<Alloc>>` (`blocked_data.h`), identical
to the `Blocked` specialization except that the outer block directory is a
`SegmentedVector`. Because the result is still a `SimpleData`, every generic facility
written against that template — the dataset concepts, the `extensions` customization
points, `compact_data`, the save/load serializer — applies with no further work.

### Reverse edges: O(|deleted|) consolidation

`graphs::ReverseEdges` (`reverse_edges.h`) keeps a per-node in-neighbor list `R(n)`, so
`consolidate()` visits only the in-neighbors of deleted nodes instead of scanning the whole
graph. It is off (null) by default; only the dynamic index enables it, so the static index
and the compaction scratch graphs pay a single null check per mutator.

The maintained invariant is `R(d) ⊇ in(d)`: `record` runs on every created edge, so `R`
may hold stale or duplicated entries but never misses a live in-edge. See *Divergence 7*
below for why the tempting weaker invariant does not work.

### Lock discipline

Three mutexes, with a global acquisition order:

```
compact_mutex_ -> slot_alloc_mutex_
compact_mutex_ -> translator_mutex_
```

`slot_alloc_mutex_` and `translator_mutex_` are never held simultaneously.

- `compact_mutex_` — shared by readers and by `add_points`; exclusive only by `compact()`,
  which *shrinks* storage and so must drain readers. Growth needs no exclusion.
- `slot_alloc_mutex_` — held only for Phase 1 of `add_points` (reserving slots). Because
  `add_points` holds `compact_mutex_` *shared* for its whole duration and
  `slot_alloc_mutex_` only briefly, **`add_points` may be called concurrently from
  multiple threads**.
- `translator_mutex_` — guards the ID translator's hash maps.

`std::shared_mutex` is not recursive, so every translation operation comes in two
flavours: `foo(...)` takes the shared lock itself, and `unsafe_foo(...)` requires the
caller to already hold it (via `lock_for_translation()`). Batch paths and
`BatchIterator::next` take the lock once and use the `unsafe_` variants; anything else
should use the plain form. Calling a self-locking accessor from a context that already
holds the lock is a latent deadlock, not merely slow: a writer arriving between the two
shared acquisitions blocks the second one.

### Slot lifecycle

`SlotMetadata` gains a fourth state, `Pending`: a slot reserved by an in-flight
`add_points` whose vector is copied but whose adjacency list is still being built. Pending
slots are invisible to search, to `consolidate`, and to subsequent `add_points` until
promoted to `Valid`.

## Deliberate divergences from the source branch

The source branch was reproduced feature-for-feature. Where this stack differs, it is for
one of two reasons: (a) it must avoid editing a pre-existing file, or (b) the source
branch has a defect. Both kinds are listed.

**1. Search-path extension (avoids editing `extensions.h`).** The source branch edits
`svs::index::vamana::extensions` directly. Here the equivalent behaviour is an
`svs_invoke` override for `single_search` plus a `supplement_search_buffer` step on the
concurrent index itself. The rewrite also fixes two problems in the original: a recursive
`shared_mutex` acquisition that can deadlock, and an iteration over the ID translator with
no lock held.

**2. Grow-stable dataset (avoids editing `core/data/simple.h`).** The source branch
changes `SimpleData<T, Extent, Blocked<Alloc>>` in place, which alters an existing dataset
type. Here it is a new allocator tag plus an additive partial specialization; see
*Grow-stable storage* above.

**3. Spin lock (avoids editing `lib/spinlock.h`).** The source branch extends
`svs::lib::SpinLock`. Here `concurrent::SpinLock` (`spinlock.h`) subclasses it and adds
what this stack needs.

**4. `capacity()` on scalar quantization — omitted.** The source branch adds a
`capacity()` accessor to `quantization/scalar/scalar.h`. It is used only for memory
reporting and nothing in this stack calls it, so it is left out rather than editing a
pre-existing header.

**5. `NullLockGuard` and the Python GIL changes — omitted.** The source branch adds a
`NullLockGuard` to `index/vamana/index.h` and releases the GIL around some Python
bindings. Both are about integrating the *modified in place* index into existing call
paths; a separate index does not need them. The Python bindings continue to expose the
pre-existing dynamic index.

**6. `PruneState` test expectations — corrected.** The source branch's test asserts
`reenable(Candidate) == Candidate`. The implementation returns `Available`, which is what
the two-round pruning heuristic requires. `tests/svs/concurrent/prune.cpp` asserts the
correct values (`reenable(Pruned) == Pruned`, `reenable(Candidate) == Available`,
`excluded(Candidate) == true`).

**7. Reverse-edge rebuild — bug fixed.** `rebuild_reverse_edges` in the source branch
records `src` into `R(dst)` only when the reverse edge `dst -> src` does not also exist,
halving the index on the reasoning that `gather_work_set` visits `out(d) ∪ R(d)` and so
already covers symmetric in-neighbors. That invariant — *"for every edge `u -> d`: `u ∈
R(d)` **or** the edge `d -> u` exists"* — holds immediately after a rebuild but is
**not maintainable**: the second disjunct is falsified the moment consolidation rewires
`d` and drops `d -> u`, at which point `u`'s in-edge is invisible, and a later deletion of
`d` leaves `u` pointing at a retired slot.

The failure is masked in the source branch because every one of its tests pairs
`consolidate()` with `compact()`, and `compact()` rebuilds the index from scratch.
Repeated `consolidate()` *without* `compact()` — the cheap maintenance path, and the
normal one — corrupts the graph within two rounds, which
`debug_check_invariants()` reports as `Node number N has an invalid (Empty) neighbor`.
This stack records unconditionally, giving the strictly stronger `R(d) ⊇ in(d)`, which no
edge *removal* can break.

**8. Unsynchronized translator reads — bug fixed.** `translate_external_id`,
`translate_external_id_or`, `has_id`, `translate_internal_id`, and `on_ids` read the
translator's `tsl::robin_map`s with no lock in the source branch, while `add_points`
inserts and `consolidate` erases under `translator_mutex_`. A comment there anticipates
reading a stale value, but the actual hazard is worse: an insert can rehash and free the
bucket array a reader is walking. These are now the two-flavour operations described under
*Lock discipline*, and TSan reports the original as a race on every search that overlaps
an insert.

**9. Greedy-search scaffolding via `using`-declarations.** See *How the separation works*.
The source branch has no analogue because it edits upstream in place and so never crosses
a namespace boundary.

## Layout

| File | Contents |
| --- | --- |
| `spinlock.h` | `concurrent::SpinLock` |
| `blocked_data.h` | `SegmentedBlocked<Alloc>` tag + `SimpleData` specialization |
| `graph_concepts.h` | graph concepts for the concurrent graph API |
| `reverse_edges.h` | `graphs::ReverseEdges` in-neighbor index |
| `graph.h` | `SimpleGraphBase`, `SimpleBlockedGraph`, `AddEdgeResult` |
| `translation.h` | `IDTranslator` |
| `dynamic_search_buffer.h` | `MutableBuffer`, `PredicatedSearchNeighbor` |
| `prune.h` | pruning heuristics and `PruneStrategy` |
| `greedy_search.h` | `greedy_search` with the SeqLock retry loop |
| `vamana_build.h` | `VamanaBuilder` |
| `consolidate.h` | reverse-edge-driven and full-scan consolidation |
| `dynamic_index.h` | `MutableVamanaIndex`, `auto_dynamic_assemble` |
| `iterator.h` | `BatchIterator` |
| `multi.h` | `MultiMutableVamanaIndex`, `MultiBatchIterator` |

Shared building blocks that are not Vamana-specific live under `svs/lib/`:
`lib/segmented_vector.h`, `lib/concurrency/seqlock.h`, `lib/concurrency/atomic_span.h`.

## Tests

`tests/svs/concurrent/` mirrors the upstream Vamana tests against this stack —
`translation.cpp`, `graph.cpp`, `prune.cpp`, `consolidate.cpp`, `dynamic_index.cpp`,
`dynamic_index_2.cpp`, `iterator.cpp`, `multi.cpp` — plus `concurrency.cpp`, which is new:
it runs searches against the index while writers insert, delete, and consolidate, and
checks id round-trips, result-set consistency, and post-mutation recall. The source branch
has no multi-threaded test.

`tests/svs/lib/segmented_vector.cpp` covers `lib::SegmentedVector` directly, including
address stability under concurrent growth.

Every test case here is tagged `[concurrent]`, so the whole set runs as

```sh
ctest -L "concurrent|segmented_vector"     # 24 tests
./tests/tests "[concurrent],[segmented_vector]"
```

`tests/svs/concurrent/dynamic_index_2.cpp` diverges from the upstream file it was ported
from in one respect: the upstream logging tests push a capturing sink onto the
*process-global* logger and never remove it, so the sink outlives by reference the vector it
captures and every later global-logger statement is a use-after-free. The port scopes the
push with a `ScopedGlobalSink` guard. The same leak exists at six sites in pre-existing test
files (`svs/index/flat/flat.cpp`, `svs/index/inverted/{memory_based,clustering}.cpp`,
`svs/index/vamana/{index,dynamic_index_2}.cpp`) and is left alone there; it is why a full
`tests` run can abort with a SIGSEGV inside an unrelated test that happens to log a warning
(commonly `Vamana Index Parameters`). That crash reproduces in a binary built without any of
this stack's sources.

### ThreadSanitizer

The correctness of this stack rests almost entirely on memory ordering, which an
uninstrumented test can only fail to disprove. TSan targets are opt-in because they cost
roughly an order of magnitude in time and memory:

```sh
cmake -DSVS_EXPERIMENTAL_ENABLE_CONCURRENT_TSAN=YES ...
ctest -L tsan
```

This builds two targets. `concurrent_tsan` must come out clean. `concurrent_tsan_negative`
is a **negative control**: it defines `SVS_CONCURRENT_UNSAFE_PLAIN_GRAPH_ACCESS`, which
degrades the adjacency-slot accessors to plain loads and stores, and is registered with
`WILL_FAIL TRUE`. A clean positive run only means something if the same run reports races
once the atomics are taken away — otherwise it is equally consistent with TSan watching
the wrong memory. (Never define that macro in a real build.)
