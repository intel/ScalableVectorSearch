# `svs::concurrent` — a dynamic Vamana index with lock-free search

A second dynamic Vamana index type whose searches never block, and never need to be excluded
from concurrent insertion or deletion.

**Nothing outside this directory changes.** No existing header is modified. The static
`VamanaIndex` and the existing `MutableVamanaIndex` are bit-for-bit unaffected and pay
nothing — no extra atomic load, no extra indirection, no extra byte per node.

## What it gives you, and what it does not

| | `svs::index::vamana::MutableVamanaIndex` | `svs::concurrent::MutableVamanaIndex` |
|---|---|---|
| search ‖ search | yes | yes |
| search ‖ insert | **caller must exclude** | **lock-free** |
| search ‖ delete | **caller must exclude** | **lock-free** |
| insert ‖ insert | caller must exclude | serialized internally |
| `consolidate()` / `compact()` | caller must exclude | stop-the-world (internally) |
| multi-label (`MultiMutableVamanaIndex`) | yes | `static_assert` — not supported |
| save / load | yes | **not supported** (`supports_saving = false`) |

The trade is deliberate: writers are serialized behind one mutex so that the only
synchronization on the *search* path is a per-node sequence lock. Applications where query
throughput matters and ingest is a single background stream get lock-free search for a
one-line type change. Applications that need concurrent writers or serialization should keep
using the existing index.

## How it works

1. **`SeqLockGraph`** (`graph.h`) stores adjacency lists in a `lib::SegmentedVector`, so
   growing the graph never moves an existing node's slot — a reader holding a pointer into
   node *i* is unaffected by a concurrent `unsafe_resize`. Each node carries a 1-byte
   `SeqLockCounter`. Elements are accessed through relaxed `std::atomic_ref`, which is a
   plain `MOV` on x86 but makes the concurrent access race-free rather than UB.

2. **`seqlock_greedy_search`** (`greedy_search.h`) wraps each node expansion in
   `read_begin` / `read_validate`. A rejected read is safe to retry: the graph never
   publishes a degree covering an unwritten slot, so anything already inserted into the
   search buffer has a valid ID and a correctly computed distance, and `insert` dedupes by
   ID. A retry costs redundant work, never correctness.

3. **`SeqLockGraphView`** (`graph_view.h`) pushes that same retry down into `get_node`,
   returning a certified snapshot copied into per-thread scratch. This is what lets
   **unmodified** upstream code — notably the 340-line `BatchIterator` — read the graph
   safely with no changes at all.

4. **`MutableVamanaIndex`** (`mutable_vamana_index.h`) holds one `writer_mutex_` to
   serialize writers, and two `WriterPriorityMutex`es held *shared* by searches: one for
   structural changes (dataset capacity growth) and one for ID-translator remaps.

`SeqLockGraph` satisfies the **unmodified** `svs::graphs::MemoryGraph` concept — including
`add_edge`'s `size_t` return — which is why `VamanaBuilder`, `prune` and `GraphConsolidator`
all work against it with no edits. There are `static_assert`s to that effect in
`tests/svs/concurrent/graph.cpp`; if a future concept change breaks the arrangement, those
fire at the concept boundary rather than deep inside a template.

## Using it

```cpp
#include "svs/concurrent/mutable_vamana_index.h"

using Index = svs::concurrent::MutableVamanaIndex<
    uint32_t, svs::data::BlockedData<float>, svs::distance::DistanceL2>;

// Build with the *stock* builder over a SeqLockGraph.
auto graph = svs::concurrent::SeqLockGraph<uint32_t>{data.size(), max_degree};
auto builder = svs::index::vamana::VamanaBuilder{
    graph, data, distance, parameters, threadpool, prefetch};
builder.construct(alpha, entry_point);

auto index = Index{
    std::move(graph), std::move(data), entry_point, distance, ids, threadpool};

// From here on, `index.search(...)` may run on any number of threads while another thread
// calls `add_points` / `delete_entries`. No external lock.
```

`consolidate()` and `compact()` are the exceptions: they take the structure lock
exclusively, so searches stall for their duration.

## Testing

`tests/svs/concurrent/{graph,mutable_vamana_index}.cpp`, tagged `[concurrent]`.

Because the correctness argument here is almost entirely about memory ordering, a
non-instrumented test can only fail to disprove it. Two opt-in ThreadSanitizer targets close
that gap:

```sh
cmake -DSVS_BUILD_TESTS=YES -DSVS_EXPERIMENTAL_ENABLE_CONCURRENT_TSAN=YES ...
ctest -L tsan
```

`concurrent_tsan` must be clean. `concurrent_tsan_negative` compiles the same graph test
with `-DSVS_CONCURRENT_UNSAFE_PLAIN_GRAPH_ACCESS`, which swaps the relaxed `atomic_ref`
accessors for plain loads and stores, and is registered `WILL_FAIL` — it must report races
in `SeqLockGraph::load_`. Without that negative control a clean run proves only that TSan
was not looking at the interesting memory.

## Known limitations

* **No serialization.** SVS's save/load is templated on the graph type; `SeqLockGraph` needs
  its own serializer and a matching loader. Mechanical, but not done.
* **No allocator.** `SegmentedVector` takes no allocator, so graph memory is invisible to
  allocator-based accounting.
* **`SeqLockCounter` is a `uint8_t`.** A reader descheduled across ≥128 writes to the *same*
  node can observe a matching counter after wraparound and accept a torn read. Widening it is
  a one-line change in `lib/concurrency/seqlock.h`.
* **Uncompressed storage only.** The analysis of which accesses may race was done against
  `svs::data::SimpleData<..., Blocked<...>>`. LVQ and LeanVec have their own layouts and
  growth behaviour and have not been reviewed.
* **`WriterPriorityMutex` has a slower fallback off glibc.** See
  `lib/concurrency/writer_priority_mutex.h`: the glibc path keeps `std::shared_mutex`'s
  atomic fast path, the portable path takes an uncontended `std::mutex` per `lock_shared`.
* **Code duplication.** The graph and the search loop are parallel copies of upstream's and
  will drift. This is the standing cost of keeping the change additive.
