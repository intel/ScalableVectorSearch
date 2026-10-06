<!--
  ~ Copyright 2026 Intel Corporation
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Version
Actualized for commit 107071c

# Concurrent dynamic Vamana index

A dynamic Vamana index that you can search and change at the same time.
Searches take no lock on the graph.

Everything lives in the namespace `svs::index::vamana::concurrent`. The main
classes are `MutableVamanaIndex` ([dynamic_index.h](dynamic_index.h)) and
`MultiMutableVamanaIndex` ([multi.h](multi.h)).

## 1. The concurrency model

There are two different questions here. First: is it **safe** to call this from
many threads? Second: does it then really **run in parallel**? For this index the
answers are not the same. Read both parts below.

### 1a. What is safe to call together

All of these are safe from any number of threads. No data race, and no broken
graph. Section 2 explains how this works.

| Operation | Note |
| --- | --- |
| `search` - one query or a batch | Takes no lock on the graph. |
| `BatchIterator::next` | Same search path. |
| `add_points` | Also from several threads at once. |
| `delete_entries` | Soft delete. It only marks slots. |
| `consolidate()`, `consolidate(ids)` | Unlinks deleted nodes. |
| `has_id`, `translate_*`, `on_ids`, `size`, `external_ids` | Id lookups. |
| `get_datum`, `get_distance`, `reconstruct_at` | Read stored vectors. |
| `replace_external_id` | Renames an id. |

### 1b. What really runs in parallel

The index has only **one** thread pool. And some pools take their own mutex, and
hold it for a whole job. A second job then waits for the first one to finish.

That is why two calls can both be safe, as 1a says, and still not run at the same
time.

Only some operations use the pool:

| Uses the pool | Does not use the pool |
| --- | --- |
| `search(results, queries, sp)` - a batch | `search(query, scratch)` - one query |
| `add_points` | `BatchIterator::next` |
| `consolidate`, `compact` | `delete_entries` |
| `translate_to_external` | id lookups |
| `reconstruct_at`, `exhaustive_search` | `get_datum`, `get_distance` |

The right column always runs in parallel. For the left column, the pool decides.
A job splits into `n = min(work, pool size)` tasks.

The pool does not have to be an SVS class. SVS accepts any external pool with two
methods, `size()` and `parallel_for()`. See the `ThreadPool` concept in
[../lib/threads/threadpool.h](../lib/threads/threadpool.h).

If a pool runs a job **inline** - on the calling thread, and without taking its own lock - then jobs
with `n == 1` run in parallel. Each caller works on its own thread, and they
share nothing.

### 1c. `compact()` stops everything

You may call `compact()` at any time, but it blocks all other work. It makes the
storage smaller, so it must wait for every reader to finish first. It is the only
caller that takes `compact_mutex_` exclusive. It runs `consolidate_locked()`
before it compacts.

### 1d. Not safe at all

Call these only when no other thread uses the index:

- `save()` - it holds no lock while it writes the files. It calls `consolidate()`
  and `compact()` first, but it releases both locks before writing.
- `set_alpha`, `set_prune_to`, `set_max_candidates`,
  `set_construction_window_size`, `set_threadpool` - plain writes, no lock.

`set_search_parameters` and `get_search_parameters` **are** safe. That member is
a `lib::ReadWriteProtected`, which holds its own lock.

## 2. Patterns used for synchronisation

### Seqlock - a search reads a node with no lock

A seqlock is a version counter. The writer makes it odd, edits the data, then
makes it even again. The reader reads the counter, reads the data, then reads the
counter again. If the counter changed, the reader tries once more.

See `SeqLockCounter` and `SeqLockArray` in
[../lib/concurrency/seqlock.h](../lib/concurrency/seqlock.h). There is one
counter per graph node.

- Readers call `read_begin()`, then `read_validate()`. `read_begin()` returns
  nothing if a write is in progress.
- Writers call `begin_write()`, then `end_write()`.

`greedy_search` ([greedy_search.h](greedy_search.h)) wraps every node in this
retry loop, so a search never blocks. Old neighbours from a failed read do no
harm. The ids are still real, and the search buffer drops duplicates. `has_edge`
([graph.h](graph.h)) and `consolidate` ([consolidate.h](consolidate.h)) use the
same loop.

A seqlock does **not** keep two writers apart. That is the spinlock's job.

### Spinlock - one writer per node

A spinlock makes the thread wait in a busy loop. This is cheaper than sleeping
when the wait is very short. `concurrent::SpinLock` ([spinlock.h](spinlock.h))
adds copy and move to `svs::SpinLock`, so a `SegmentedVector` can hold it.

There are three groups:

- `SimpleGraphBase::node_locks_` ([graph.h](graph.h)) - one lock per node.
  `add_edge` and `clear_node` take it. `lock_node(i)` gives it to the caller, who
  can then read, prune, and write one node as a single step. `consolidate`,
  `VamanaBuilder`, and `delete_entry` work this way.
- `ReverseEdges::locks_` ([reverse_edges.h](reverse_edges.h)) - one lock per
  node. `record`, `remove`, `collect`, and `reset_node` touch one node's list
  only.
- `BackedgeBuffer::bucket_locks_` ([vamana_build.h](vamana_build.h)) - the same
  idea, but one `std::mutex` per *group* of node ids, not per node.

### `std::shared_mutex` - many readers or one writer

| Lock | Protects | Taken exclusive by |
| --- | --- | --- |
| `compact_mutex_` | the storage stays alive | `compact()` only |
| `translator_mutex_` | the two id hash maps | `add_points`, `consolidate`, `compact`, `replace_external_id` |
| `pending_insertions_mutex_` | `pending_insertions_` | `add_points` |
| `slot_alloc_mutex_` (plain `std::mutex`) | the scan for free slots | `add_points` |

`MultiMutableVamanaIndex` adds `l2e_mutex_` and `e2l_mutex_` for its own label
maps ([multi.h](multi.h)).

**Why writers take `compact_mutex_` shared.** Growing the storage is safe, see
section 3. Only shrinking is not. So `add_points`, `delete_entries`, and
`consolidate` take this lock *shared*. Only `compact()` takes it exclusive, and
so it waits for all of them.

**Lock order.** Always take locks in the same order. Then threads cannot
deadlock.

```
compact_mutex_ -> slot_alloc_mutex_
compact_mutex_ -> translator_mutex_
l2e_mutex_     -> e2l_mutex_          (multi index)
```

`slot_alloc_mutex_` and `translator_mutex_` are never held together.

**The `unsafe_` prefix.** `std::shared_mutex` is not recursive. So every id
operation has two forms:

- `foo(...)` takes the shared lock itself. Use this by default.
- `unsafe_foo(...)` needs the caller to hold the lock already, through
  `lock_for_translation()`. Use this to translate a whole batch under one lock.

Do not call `foo(...)` when you already hold the lock. This can deadlock.

**Nodes that are still being built.** A single-vector `add_points` puts its new
node id into `pending_insertions_` after it copies the vector. Another
`add_points` reads this set, so it can link to a node that is almost ready. Batch
inserts do not use the set.

### Atomic slot states

`status_` holds one byte per slot. It is read and written with `std::atomic_ref`.
There are four states:

| State | Meaning |
| --- | --- |
| `Empty` | Free. |
| `Valid` | Live. A search may return it. |
| `Deleted` | Soft deleted. Still in the graph, but never returned. |
| `Pending` | An `add_points` is still filling it. |

`Pending` is the key to concurrent adds. A search skips such a slot, because
`ValidBuilder` accepts only `Valid`. Other writers skip it too. A thread claims a
slot with `compare_exchange_strong`, so only one thread wins it.

`first_empty_`, `first_reusable_`, `num_valid_`, and the entry point are plain
atomics. The helpers `detail::atomic_min` and `detail::atomic_max` move a counter
in one direction only.

### Relaxed atomics on neighbour slots

Every neighbour id is read and written through `relaxed_load` and `relaxed_store`
([graph.h](graph.h)). The ordering comes from the seqlock counters, not from
these accesses. Relaxed atomics compile to a plain load or store, so they cost
nothing.
## 3. Grow-stable storage

The rule: a search may hold a pointer into the storage while another thread adds
points. So adding must never move the elements that are already there.

`lib::SegmentedVector` ([../lib/segmented_vector.h](../lib/segmented_vector.h))
is a two-level array. Growth adds a new segment, so old elements keep the same
address. It holds `status_`, the seqlock counters, the node spinlocks, and both
arrays in `ReverseEdges`.

The dataset needs the same property. It gets it from the allocator tag
`SegmentedBlocked<Alloc>` and the matching `SimpleData` specialisation
([blocked_data.h](blocked_data.h)). This is the same as the `Blocked` version,
except that the outer block directory is a `SegmentedVector`. The result is still
a `SimpleData`, so the dataset concepts, the `extensions` hooks, `compact_data`,
and save/load all work with no extra code.

`resize()` publishes the new size with a release store, and `size()` reads it
with an acquire load. A search that overlaps an `add_points` therefore sees
either the old size or the new one, never a broken value.
