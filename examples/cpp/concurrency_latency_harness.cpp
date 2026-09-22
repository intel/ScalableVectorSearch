/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

///
/// @file
/// Concurrency latency harness for `svs::index::vamana::MutableVamanaIndex`.
///
/// This is a measurement instrument, not a feature: its one job is to make the
/// consolidate()/compact() stall visible and quantify it separately for searches and for
/// updates, across a sweep of searcher-thread counts. It reproduces today's upstream
/// behaviour, in which the index is not safe for concurrent search during mutation, by
/// serializing access through a single writer-preferring `std::shared_mutex` (see
/// `SearchExclusion` below). A later change is expected to remove the exclusion for
/// searches; that change should only need to touch `SearchExclusion`, not the thread
/// bodies that use it.
///
/// The mutator runs flat out (no write:search throttle - see `mutator_loop`); repair
/// (`consolidate()`) runs on its own thread and serializes against the mutator through the
/// same `SearchExclusion` mutation guard, so a mutator update's measured latency now
/// includes any time it spends waiting behind an in-progress repair.
///

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/lib/exception.h"

// benchmark utility (header-only)
#include "svs-benchmark/latency.h"

// example support
#include "svsmain.h"

// stl
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
inline double to_seconds(Clock::duration d) {
    return std::chrono::duration<double>(d).count();
}

// Sleeps toward `deadline`, checking `should_stop` at least every 50ms so a long-horizon
// deadline (a slow paced rate) does not delay shutdown.
template <class ShouldStop>
void sleep_until_paced(Clock::time_point deadline, ShouldStop&& should_stop) {
    while (!should_stop()) {
        auto now = Clock::now();
        if (now >= deadline) {
            return;
        }
        std::this_thread::sleep_until(std::min(deadline, now + std::chrono::milliseconds(50)));
    }
}

///// ------------------------------------------------------------------------------------
///// Command-line configuration
///// ------------------------------------------------------------------------------------

std::vector<size_t> parse_size_list(const std::string& csv) {
    std::vector<size_t> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(static_cast<size_t>(std::stoull(item)));
        }
    }
    return out;
}

struct Config {
    std::string train_path;
    std::string test_path;

    size_t build_size = 500'000;
    size_t insert_pool_size = 0; // 0 => derive from remaining vectors in the train file.

    // With this false (the default), every insert appends a new slot instead of reusing a
    // freed one, so the end-of-run memory breakdown never differs from the start-of-run one.
    bool reuse_empty = false;

    std::vector<size_t> searcher_sweep = {1, 2, 4, 8, 16, 32, 64};
    size_t num_index_threads = 8;

    size_t graph_max_degree = 64;
    size_t search_window_size = 30;

    // Repair (consolidate) fires every this-many mutator operations (each delete or each
    // insert counts as one), from a dedicated thread - see file header.
    size_t repair_every = 200;
    double compact_every_seconds = 5.0;

    double warmup_seconds = 30.0;
    double window_seconds = 120.0;
    size_t num_windows = 3;

    // Grace period after a window's nominal end before the main thread trusts that every
    // worker thread's writes into that window's recorder are visible (see the heartbeat
    // acquire/release discussion on `Heartbeats`).
    double window_grace_ms = 100.0;

    // Update latency above this is counted toward the "fraction of updates that stalled"
    // report; repair now shows up in update latency, which is what this is meant to catch.
    double slow_update_threshold_s = 0.100;

    // "full" calls consolidate() (repairs every deleted slot in one sweep); "slice"
    // calls consolidate_slice() over 1/slice_k of the node range each time, so the same
    // repair_every cadence delays more updates by less each. K=1 is a full sweep.
    std::string repair_mode = "full";
    size_t slice_k = 100;

    // Mutations per 100 completed searches the mutator paces itself to, in the current
    // measurement window; 0 (default) means no throttle, i.e. today's flat-out behaviour.
    double target_ratio = 0.0;

    // Absolute aggregate searches/s offered across all searcher threads; 0 (default) =
    // unthrottled. When set, takes precedence over target_ratio (see parse_config).
    double search_rate = 0.0;

    // Mutations per 100 searches, applied against search_rate (not the achieved rate) to
    // derive an absolute mutation rate; 0 (default) = unthrottled.
    double write_ratio = 0.0;
};

class ArgMap {
  public:
    ArgMap(const std::vector<std::string>& args) {
        for (size_t i = 1; i + 1 < args.size(); i += 2) {
            if (args[i].rfind("--", 0) != 0) {
                throw ANNEXCEPTION("Expected a flag starting with '--', got {}", args[i]);
            }
            map_[args[i].substr(2)] = args[i + 1];
        }
    }

    std::string get(const std::string& key, std::string fallback) const {
        auto it = map_.find(key);
        return it == map_.end() ? fallback : it->second;
    }

    size_t get(const std::string& key, size_t fallback) const {
        auto it = map_.find(key);
        return it == map_.end() ? fallback : static_cast<size_t>(std::stoull(it->second));
    }

    double get(const std::string& key, double fallback) const {
        auto it = map_.find(key);
        return it == map_.end() ? fallback : std::stod(it->second);
    }

    bool get(const std::string& key, bool fallback) const {
        auto it = map_.find(key);
        return it == map_.end() ? fallback : (it->second == "1" || it->second == "true");
    }

  private:
    std::unordered_map<std::string, std::string> map_;
};

Config parse_config(const std::vector<std::string>& args) {
    ArgMap m(args);
    Config c;
    // Required rather than defaulted: a dataset location is site-specific and must not be baked
    // into a published source file.
    c.train_path = m.get("train", c.train_path);
    if (c.train_path.empty()) {
        throw ANNEXCEPTION("--train is required, e.g. --train <dir>/cohere-1M_train.fvecs");
    }
    c.test_path = m.get("test", c.test_path);
    if (c.test_path.empty()) {
        throw ANNEXCEPTION("--test is required, e.g. --test <dir>/cohere-1M_test.fvecs");
    }
    c.build_size = m.get("build-size", c.build_size);
    c.insert_pool_size = m.get("insert-pool-size", c.insert_pool_size);
    c.reuse_empty = m.get("reuse-empty", c.reuse_empty);
    {
        std::string sweep = m.get("searcher-sweep", std::string());
        if (!sweep.empty()) {
            c.searcher_sweep = parse_size_list(sweep);
        }
    }
    c.num_index_threads = m.get("num-index-threads", c.num_index_threads);
    c.graph_max_degree = m.get("graph-max-degree", c.graph_max_degree);
    c.search_window_size = m.get("search-window-size", c.search_window_size);
    c.repair_every = m.get("repair-every", c.repair_every);
    c.compact_every_seconds = m.get("compact-every-seconds", c.compact_every_seconds);
    c.warmup_seconds = m.get("warmup-seconds", c.warmup_seconds);
    c.window_seconds = m.get("window-seconds", c.window_seconds);
    c.num_windows = m.get("num-windows", c.num_windows);
    c.window_grace_ms = m.get("window-grace-ms", c.window_grace_ms);
    c.slow_update_threshold_s = m.get("slow-update-threshold-s", c.slow_update_threshold_s);
    c.repair_mode = m.get("repair-mode", c.repair_mode);
    if (c.repair_mode != "full" && c.repair_mode != "slice") {
        throw ANNEXCEPTION(
            "--repair-mode must be 'full' or 'slice', got '{}'", c.repair_mode
        );
    }
    c.slice_k = m.get("slice-k", c.slice_k);
    if (c.slice_k == 0) {
        throw ANNEXCEPTION("--slice-k must be >= 1!");
    }
    c.target_ratio = m.get("target-ratio", c.target_ratio);
    c.search_rate = m.get("search-rate", c.search_rate);
    c.write_ratio = m.get("write-ratio", c.write_ratio);
    return c;
}

///// ------------------------------------------------------------------------------------
///// The single, easily-replaceable exclusion point.
///// ------------------------------------------------------------------------------------

/// Models today's upstream behaviour: `MutableVamanaIndex` is not safe for concurrent
/// search during mutation, so every search takes the lock SHARED and every mutation
/// (insert, delete, consolidate, compact) takes it EXCLUSIVE. A finer-grained scheme that
/// stops excluding searches only needs to change this class; callers only ever ask for a
/// search guard or a mutation guard.
///
/// A bare `std::shared_mutex` is reader-preferring under sustained shared-lock traffic
/// (glibc's `pthread_rwlock` default), so with many continuously-searching threads a
/// waiting writer can starve indefinitely rather than merely wait its turn. `turnstile_`
/// fixes that: both a searcher and the mutator must acquire-then-release it before (for
/// searchers) or while waiting to acquire (for the mutator) `mutex_`. A waiting mutator
/// holds the turnstile for its whole wait, so no new searcher can even attempt `mutex_`
/// until the ones already in flight drain and the mutator gets in - bounding the
/// mutator's wait by one round of in-flight searches rather than by reader arrival rate.
/// Repair (consolidate) is just another mutation-guard caller: it serializes against the
/// mutator here, not through a second lock.
class SearchExclusion {
  public:
    using SearchGuard = std::shared_lock<std::shared_mutex>;
    using MutationGuard = std::unique_lock<std::shared_mutex>;

    SearchGuard lock_for_search() {
        std::lock_guard<std::mutex> turn(turnstile_);
        return SearchGuard{mutex_};
    }

    MutationGuard lock_for_mutation() {
        std::lock_guard<std::mutex> turn(turnstile_);
        return MutationGuard{mutex_};
    }

  private:
    std::mutex turnstile_;
    std::shared_mutex mutex_;
};

///// ------------------------------------------------------------------------------------
///// Per-thread latency recording: histogram + running sum/min/max. No shared structure is
///// touched on the hot path; each thread owns its own recorder for the window it is
///// currently in, and cross-thread visibility at a window boundary is established
///// separately (see `Heartbeats`), not by any lock here.
///// ------------------------------------------------------------------------------------

class LatencyRecorder {
  public:
    void record(double latency_s) {
        hist_.record(latency_s);
        sum_ += latency_s;
        min_ = std::min(min_, latency_s);
        max_ = std::max(max_, latency_s);
        ++count_;
    }

    const svsbenchmark::LatencyHistogram& histogram() const { return hist_; }
    double sum() const { return sum_; }
    double min() const { return count_ == 0 ? 0.0 : min_; }
    double max() const { return max_; }
    uint64_t count() const { return count_; }

    // Merge `other` into `*this`. Used to combine per-searcher-thread recorders after a
    // window closes; never called on the hot path.
    void merge(const LatencyRecorder& other) {
        hist_.merge(other.hist_);
        sum_ += other.sum_;
        min_ = std::min(min_, other.min_);
        max_ = std::max(max_, other.max_);
        count_ += other.count_;
    }

  private:
    svsbenchmark::LatencyHistogram hist_{};
    double sum_ = 0.0;
    double min_ = std::numeric_limits<double>::infinity();
    double max_ = 0.0;
    uint64_t count_ = 0;
};

struct PercentileReport {
    uint64_t count;
    double mean;
    double min;
    double max;
    double p50;
    double p95;
    double p99;
    double p999;
};

PercentileReport summarize(const LatencyRecorder& r) {
    const auto& h = r.histogram();
    PercentileReport out{};
    out.count = r.count();
    out.mean = out.count == 0 ? 0.0 : r.sum() / static_cast<double>(out.count);
    out.min = r.min();
    out.max = r.max();
    out.p50 = h.percentile(50.0);
    out.p95 = h.percentile(95.0);
    out.p99 = h.percentile(99.0);
    out.p999 = h.percentile(99.9);
    return out;
}

void print_percentiles(const std::string& label, const PercentileReport& p) {
    fmt::print(
        "{:<10s} count={:<8d} mean={:>9.6f}s min={:>9.6f}s p50={:>9.6f}s p95={:>9.6f}s "
        "p99={:>9.6f}s p99.9={:>9.6f}s max={:>9.6f}s\n",
        label,
        p.count,
        p.mean,
        p.min,
        p.p50,
        p.p95,
        p.p99,
        p.p999,
        p.max
    );
}

///// ------------------------------------------------------------------------------------
///// Achieved write:search ratio. There is no throttle and no target any more (see file
///// header): this is purely a measured output of a window, computed from that window's
///// own recorder counts so warmup and other-window activity cannot leak into it.
///// ------------------------------------------------------------------------------------

struct RatioReport {
    uint64_t search_count;
    uint64_t update_count;
    double achieved_writes_per_100_searches;
};

RatioReport compute_ratio_report(uint64_t search_count, uint64_t update_count) {
    RatioReport r{};
    r.search_count = search_count;
    r.update_count = update_count;
    r.achieved_writes_per_100_searches =
        search_count == 0
            ? 0.0
            : static_cast<double>(update_count) * 100.0 / static_cast<double>(search_count);
    return r;
}

void print_ratio_report(const std::string& label, const RatioReport& r) {
    fmt::print(
        "[{}] ACHIEVED WRITE:SEARCH RATIO -- {} updates / {} searches = {:.4f} "
        "writes/100 searches\n",
        label,
        r.update_count,
        r.search_count,
        r.achieved_writes_per_100_searches
    );
}

///// ------------------------------------------------------------------------------------
///// Record of a single repair (consolidate) call.
///// ------------------------------------------------------------------------------------

struct CallRecord {
    double start_s; // seconds since the window's own start (may be slightly negative if
                    // the call started just before the window boundary).
    double duration_s;
};

///// ------------------------------------------------------------------------------------
///// One operating point's window schedule, published to the continuously-running mutator
///// and repair threads (searcher threads get it by plain reference - see below).
///// ------------------------------------------------------------------------------------

struct WindowBounds {
    std::array<Clock::time_point, 16> starts{};
    std::array<Clock::time_point, 16> ends{};
    size_t num_windows = 0;

    // Returns the window `t` falls in, or -1 if `t` is in warmup or a point transition
    // gap. Windows are contiguous and non-overlapping, so at most one can match.
    int index_for(Clock::time_point t) const {
        for (size_t i = 0; i < num_windows; ++i) {
            if (t >= starts[i] && t < ends[i]) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }
};

struct UpdateWindowData {
    LatencyRecorder recorder;
    uint64_t slow_count = 0; // count of updates whose latency exceeded the threshold.
};

// Owns one operating point's window bounds plus the mutator's and repair thread's
// per-window results. Retained for the whole program (see `all_point_contexts` below):
// the mutator/repair threads publish a pointer to the *current* one, and a stale reader
// mid-transition must land on valid memory, not a freed one.
struct PointContext {
    WindowBounds bounds;
    std::vector<UpdateWindowData> update_windows;
    std::vector<std::vector<CallRecord>> repairs;
    // Live per-window completed-search count, bumped by searchers and read by the mutator
    // for --target-ratio pacing; a fresh PointContext per point resets it with the window.
    std::vector<std::atomic<uint64_t>> search_completed;

    explicit PointContext(size_t num_windows)
        : update_windows(num_windows)
        , repairs(num_windows)
        , search_completed(num_windows) {}
};

///// ------------------------------------------------------------------------------------
///// Heartbeats: each worker thread (searcher, mutator, repair) owns one atomic counter
///// that it bumps with release ordering immediately after writing a sample into its
///// current window's recorder. The main thread, after sleeping past a window's nominal
///// end plus a grace period, does one acquire load of every relevant heartbeat before
///// reading that window's data - the release/acquire pair on the SAME atomic is what makes
///// the just-written recorder contents visible; without it this would be an unsynchronized
///// cross-thread read.
///// ------------------------------------------------------------------------------------

using Heartbeat = std::atomic<uint64_t>;

void bump(Heartbeat& h) { h.fetch_add(1, std::memory_order_release); }
void wait_for_visibility(Heartbeat& h) { (void)h.load(std::memory_order_acquire); }

///// ------------------------------------------------------------------------------------
///// Shared run state, for the whole program lifetime.
///// ------------------------------------------------------------------------------------

struct GlobalState {
    std::atomic<bool> stop{false};   // stops the mutator and repair threads for good.
    std::atomic<bool> failed{false}; // set by any thread on an unhandled exception.
    // Total mutator operations (each delete or each insert counts as one); read by the
    // repair thread (relaxed, just a threshold poll) and by main (relaxed, informational).
    std::atomic<uint64_t> mutation_count{0};

    std::mutex error_mutex;
    std::vector<std::string> errors;

    void report_error(const std::string& who, const std::exception& e) {
        std::lock_guard<std::mutex> lock(error_mutex);
        errors.push_back(who + ": " + e.what());
        failed.store(true, std::memory_order_relaxed);
        stop.store(true, std::memory_order_relaxed);
    }
};

///// ------------------------------------------------------------------------------------
///// Index type aliases. Both the graph and the dataset must be blocked storage: a
///// searcher can be holding a span into `data_` (via `get_datum`) while the mutator
///// grows the dataset for an insert. Non-blocked storage reallocates on growth, which
///// would turn that into a use-after-free; blocked storage only appends new blocks, so
///// existing spans stay valid. That matters for a later, less-exclusive locking scheme;
///// under this harness's exclusive-lock baseline it also happens to be moot, since no
///// search runs concurrently with a mutation - but building on the wrong storage type
///// now would make that later change unsafe by construction.
///// ------------------------------------------------------------------------------------

using GraphType = svs::graphs::SimpleBlockedGraph<uint32_t>;
using DataType = svs::data::BlockedData<float>;
using DistType = svs::DistanceL2;
using Index = svs::index::vamana::MutableVamanaIndex<GraphType, DataType, DistType>;

///// ------------------------------------------------------------------------------------
///// Searcher thread. Spawned fresh for each operating point, so it takes its point's
///// `WindowBounds` by plain reference - no atomic needed, the object outlives the thread.
///// ------------------------------------------------------------------------------------

void searcher_loop(
    size_t thread_id,
    const Index& index,
    const svs::data::SimpleData<float>& queries,
    const svs::index::vamana::VamanaSearchParameters& search_params,
    SearchExclusion& exclusion,
    GlobalState& gstate,
    const std::atomic<bool>& point_stop,
    const WindowBounds& bounds,
    std::vector<LatencyRecorder>& window_recorders, // sized bounds.num_windows
    std::vector<std::atomic<uint64_t>>& search_completed, // sized bounds.num_windows
    size_t num_threads,
    double rate_per_thread,
    Clock::time_point pace_epoch,
    Heartbeat& heartbeat
) {
    try {
        auto scratch = index.scratchspace(search_params);
        size_t num_queries = queries.size();
        size_t qi = thread_id;
        uint64_t op_index = 0;

        // Aggregate rate divided across num_threads; phase-offsetting each thread's
        // deadline sequence by thread_id/num_threads spreads arrivals instead of bursting
        // every thread at the same instant.
        //
        // Returns the intended arrival time, which is where latency must be measured from: a
        // query held back because the index was busy was already waiting, and starting its
        // clock after the wait hides one long stall behind every query it delayed.
        auto pace_for_rate = [&]() -> Clock::time_point {
            if (rate_per_thread <= 0.0) {
                return Clock::now();
            }
            double phase = static_cast<double>(thread_id) / static_cast<double>(num_threads);
            auto deadline = pace_epoch + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(
                    (static_cast<double>(op_index) + phase) / rate_per_thread
                )
            );
            sleep_until_paced(deadline, [&] {
                return gstate.stop.load(std::memory_order_relaxed) ||
                       point_stop.load(std::memory_order_relaxed);
            });
            ++op_index;
            return deadline;
        };

        while (!gstate.stop.load(std::memory_order_relaxed) &&
               !point_stop.load(std::memory_order_relaxed)) {
            auto query = queries.get_datum(qi % num_queries);
            qi += 1;

            auto t0 = pace_for_rate();
            {
                auto guard = exclusion.lock_for_search();
                index.search(query, scratch);
            }
            auto t1 = Clock::now();

            int w = bounds.index_for(t1);
            if (w >= 0) {
                window_recorders[static_cast<size_t>(w)].record(to_seconds(t1 - t0));
                search_completed[static_cast<size_t>(w)].fetch_add(
                    1, std::memory_order_relaxed
                );
            }
            bump(heartbeat);
        }
    } catch (const std::exception& e) {
        gstate.report_error(fmt::format("searcher[{}]", thread_id), e);
    }
}

///// ------------------------------------------------------------------------------------
///// Mutator thread: runs flat out for the whole program, across every operating point.
///// No write:search throttle - see file header. Repair no longer happens here; it is the
///// dedicated repair thread's job, serializing against this thread through the same
///// `SearchExclusion` mutation guard.
///// ------------------------------------------------------------------------------------

// Bookkeeping the harness owns entirely; the index is never asked for the set of live
// ids. `live_ids` starts as every id used at build time and grows/shrinks by exactly one
// entry per delete/insert, so the index's valid-entry count stays pinned at `build_size`.
struct MutatorDataState {
    std::deque<size_t> live_ids;
    size_t next_id;
    size_t pool_cursor = 0;
};

void mutator_loop(
    Index& index,
    const svs::data::SimpleData<float>& insert_pool,
    SearchExclusion& exclusion,
    GlobalState& gstate,
    double compact_every_seconds,
    double slow_update_threshold_s,
    bool reuse_empty,
    double target_ratio,
    bool use_absolute_rates,
    double absolute_mutation_rate,
    MutatorDataState& dstate,
    std::atomic<PointContext*>& current_ctx,
    Heartbeat& heartbeat
) {
    try {
        auto& live_ids = dstate.live_ids;
        size_t pool_size = insert_pool.size();
        auto last_compact = Clock::now();

        auto record_update = [&](Clock::time_point t0, Clock::time_point t1) {
            PointContext* ctx = current_ctx.load(std::memory_order_acquire);
            if (ctx == nullptr) {
                return;
            }
            int w = ctx->bounds.index_for(t1);
            if (w < 0) {
                return;
            }
            double latency_s = to_seconds(t1 - t0);
            auto& uw = ctx->update_windows[static_cast<size_t>(w)];
            uw.recorder.record(latency_s);
            if (latency_s > slow_update_threshold_s) {
                ++uw.slow_count;
            }
        };

        // Blocks the mutator, strictly before its next t0, until its own update count in
        // the current window is no longer ahead of target_ratio/100 * that window's
        // completed searches; a no-op once caught up, or always when target_ratio <= 0.
        auto pace_for_ratio = [&]() {
            if (target_ratio <= 0.0) {
                return;
            }
            while (!gstate.stop.load(std::memory_order_relaxed)) {
                PointContext* ctx = current_ctx.load(std::memory_order_acquire);
                if (ctx == nullptr) {
                    return;
                }
                int w = ctx->bounds.index_for(Clock::now());
                if (w < 0) {
                    return; // Warmup or a point-transition gap: no window to pace against.
                }
                uint64_t searches =
                    ctx->search_completed[static_cast<size_t>(w)].load(
                        std::memory_order_relaxed
                    );
                uint64_t updates = ctx->update_windows[static_cast<size_t>(w)].recorder.count();
                if (static_cast<double>(updates) <
                    target_ratio / 100.0 * static_cast<double>(searches)) {
                    return;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        };

        auto rate_epoch = Clock::now();
        uint64_t rate_op_index = 0;

        // Deadline scheduling: op i fires at rate_epoch + i/rate, so a long-running op
        // does not push every later one back by its own overrun the way a fixed
        // inter-op sleep would.
        auto pace_for_absolute_rate = [&]() -> Clock::time_point {
            if (absolute_mutation_rate <= 0.0) {
                return Clock::now();
            }
            auto deadline = rate_epoch + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(
                    static_cast<double>(rate_op_index) / absolute_mutation_rate
                )
            );
            sleep_until_paced(deadline, [&] {
                return gstate.stop.load(std::memory_order_relaxed);
            });
            ++rate_op_index;
            return deadline;
        };

        // --search-rate takes precedence over --target-ratio (see parse_config / banner).
        // Ratio pacing has no arrival schedule, so it can only report service time.
        auto pace = [&]() -> Clock::time_point {
            if (use_absolute_rates) {
                return pace_for_absolute_rate();
            }
            pace_for_ratio();
            return Clock::now();
        };

        auto do_compact = [&]() {
            auto t0 = Clock::now();
            {
                auto guard = exclusion.lock_for_mutation();
                index.compact();
            }
            auto t1 = Clock::now();
            (void)t0;
            (void)t1; // Compaction is not under test here (--compact-every-seconds 0).
        };

        while (!gstate.stop.load(std::memory_order_relaxed)) {
            // --- Delete the oldest live id. -------------------------------------------
            size_t old_id = live_ids.front();
            live_ids.pop_front();
            auto t0 = pace();
            {
                auto guard = exclusion.lock_for_mutation();
                index.delete_entries(std::array<size_t, 1>{old_id});
            }
            auto t1 = Clock::now();
            record_update(t0, t1);
            gstate.mutation_count.fetch_add(1, std::memory_order_release);
            bump(heartbeat);

            // --- Insert a fresh vector from the held-back pool. -----------------------
            size_t new_id = dstate.next_id++;
            auto vec = insert_pool.get_datum(dstate.pool_cursor);
            dstate.pool_cursor = (dstate.pool_cursor + 1) % pool_size;
            auto view = svs::data::ConstSimpleDataView<float>(vec.data(), 1, vec.size());
            t0 = pace();
            {
                auto guard = exclusion.lock_for_mutation();
                index.add_points(view, std::array<size_t, 1>{new_id}, reuse_empty);
            }
            t1 = Clock::now();
            record_update(t0, t1);
            live_ids.push_back(new_id);
            gstate.mutation_count.fetch_add(1, std::memory_order_release);
            bump(heartbeat);

            // --- Compaction on its own, independent, wall-clock schedule. -------------
            if (compact_every_seconds > 0.0) {
                auto now = Clock::now();
                if (to_seconds(now - last_compact) >= compact_every_seconds) {
                    do_compact();
                    last_compact = Clock::now();
                }
            }
        }
    } catch (const std::exception& e) { gstate.report_error("mutator", e); }
}

///// ------------------------------------------------------------------------------------
///// Repair thread: watches the mutation counter and calls consolidate() (or, in "slice"
///// mode, consolidate_slice() over 1/slice_k of the node range) every `repair_every`
///// mutations, through the same SearchExclusion mutation guard the mutator uses. It is a
///// writer like any other; it does not get a second lock.
///// ------------------------------------------------------------------------------------

void repair_loop(
    Index& index,
    SearchExclusion& exclusion,
    GlobalState& gstate,
    size_t repair_every,
    const std::string& repair_mode,
    size_t slice_k,
    std::atomic<PointContext*>& current_ctx,
    Heartbeat& heartbeat
) {
    try {
        uint64_t next_threshold = repair_every;
        while (!gstate.stop.load(std::memory_order_relaxed)) {
            uint64_t count = gstate.mutation_count.load(std::memory_order_relaxed);
            if (count < next_threshold) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            auto t0 = Clock::now();
            {
                auto guard = exclusion.lock_for_mutation();
                if (repair_mode == "slice") {
                    // The slice size is this experiment's independent variable: it must
                    // come from the index's own node count, not be reconstructed from
                    // the mutator's bookkeeping (which is a coincidence of today's
                    // 1-mutation-per-op counting, not a contract).
                    size_t n_nodes = index.view_graph().n_nodes();
                    size_t batch = (n_nodes + slice_k - 1) / slice_k;
                    index.consolidate_slice(batch);
                } else {
                    index.consolidate();
                }
            }
            auto t1 = Clock::now();
            next_threshold += repair_every;

            PointContext* ctx = current_ctx.load(std::memory_order_acquire);
            if (ctx != nullptr) {
                int w = ctx->bounds.index_for(t1);
                if (w >= 0) {
                    double window_start_s =
                        to_seconds(t0 - ctx->bounds.starts[static_cast<size_t>(w)]);
                    ctx->repairs[static_cast<size_t>(w)].push_back(
                        {window_start_s, to_seconds(t1 - t0)}
                    );
                }
            }
            bump(heartbeat);
        }
    } catch (const std::exception& e) { gstate.report_error("repair", e); }
}

///// ------------------------------------------------------------------------------------
///// Memory + RSS reporting.
///// ------------------------------------------------------------------------------------

void print_memory_breakdown(
    const std::string& label, const svs::index::vamana::MemoryBreakdown& mb
) {
    fmt::print(
        "[{}] MEMORY -- graph_bytes={} data_bytes={} metadata_bytes={} total={}\n",
        label,
        mb.graph_bytes,
        mb.data_bytes,
        mb.metadata_bytes,
        mb.total()
    );
}

// VmRSS/VmHWM from /proc/self/status. Both exclude SVS's 1 GiB hugetlb pages (see the
// harness's results writeup); reported for the record, not as the memory figure to trust.
void print_proc_status_rss() {
    std::ifstream in("/proc/self/status");
    if (!in) {
        fmt::print("[proc] could not open /proc/self/status\n");
        return;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmHWM:", 0) == 0) {
            fmt::print("[proc] {}\n", line);
        }
    }
}

} // namespace

int svs_main(std::vector<std::string> args) {
    // stdout is fully buffered rather than line buffered once redirected to a file (as it
    // is under nohup); without this, a run killed mid-flight leaves the log empty even
    // though every window printed - defeating the point of printing per-window results.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Config cfg = parse_config(args);

    fmt::print("=== Configuration ===\n");
    fmt::print("train_path            = {}\n", cfg.train_path);
    fmt::print("test_path             = {}\n", cfg.test_path);
    fmt::print("build_size            = {}\n", cfg.build_size);
    fmt::print("insert_pool_size      = {} (0 = derive)\n", cfg.insert_pool_size);
    fmt::print("reuse_empty           = {}\n", cfg.reuse_empty);
    {
        std::string sweep_str;
        for (size_t i = 0; i < cfg.searcher_sweep.size(); ++i) {
            sweep_str += std::to_string(cfg.searcher_sweep[i]);
            if (i + 1 < cfg.searcher_sweep.size()) {
                sweep_str += ",";
            }
        }
        fmt::print("searcher_sweep        = {}\n", sweep_str);
    }
    fmt::print("num_index_threads     = {}\n", cfg.num_index_threads);
    fmt::print("graph_max_degree      = {}\n", cfg.graph_max_degree);
    fmt::print("search_window_size    = {}\n", cfg.search_window_size);
    fmt::print("repair_every          = {} mutation calls\n", cfg.repair_every);
    fmt::print("repair_mode           = {}\n", cfg.repair_mode);
    fmt::print("slice_k               = {}\n", cfg.slice_k);
    fmt::print("compact_every_seconds = {}\n", cfg.compact_every_seconds);
    fmt::print("warmup_seconds        = {}\n", cfg.warmup_seconds);
    fmt::print("window_seconds        = {}\n", cfg.window_seconds);
    fmt::print("num_windows           = {}\n", cfg.num_windows);
    fmt::print("slow_update_threshold_s = {}\n", cfg.slow_update_threshold_s);
    fmt::print("target_ratio          = {} mutations per 100 searches\n", cfg.target_ratio);
    fmt::print(
        "search_rate           = {} searches/s (aggregate; 0 = unthrottled)\n",
        cfg.search_rate
    );
    fmt::print(
        "write_ratio           = {} mutations per 100 searches (of search_rate)\n",
        cfg.write_ratio
    );
    if (cfg.search_rate > 0.0) {
        fmt::print("search_rate > 0: --target-ratio is ignored; pacing uses absolute rates.\n");
    }

    if (cfg.num_windows > 16) {
        throw ANNEXCEPTION(
            "num_windows ({}) exceeds WindowBounds' fixed capacity (16)!", cfg.num_windows
        );
    }

    // --- Load the training vectors once, plain (non-blocked) storage; this is staging
    // only, never handed to the index. The initial build_size vectors are copied into
    // blocked storage for the index; the remainder sources the insert pool directly. ---
    fmt::print("\nLoading {} ...\n", cfg.train_path);
    auto raw = svs::data::SimpleData<float>::load(cfg.train_path);
    fmt::print("Loaded {} vectors of dimension {}.\n", raw.size(), raw.dimensions());

    size_t remaining = raw.size() > cfg.build_size ? raw.size() - cfg.build_size : 0;
    size_t pool_size = cfg.insert_pool_size == 0 ? remaining : cfg.insert_pool_size;
    if (cfg.build_size + pool_size > raw.size()) {
        throw ANNEXCEPTION(
            "build_size ({}) + insert_pool_size ({}) exceeds available training vectors "
            "({})!",
            cfg.build_size,
            pool_size,
            raw.size()
        );
    }
    if (pool_size == 0) {
        throw ANNEXCEPTION("No training vectors held back for the insert pool!");
    }

    // Held-back pool: copy the tail of `raw` out into its own buffer so the mutator has
    // an explicit, narrow dependency (a plain `SimpleData<float>`) rather than reaching
    // back into the loader's staging array.
    auto insert_pool_owned = svs::data::SimpleData<float>(pool_size, raw.dimensions());
    for (size_t i = 0; i < pool_size; ++i) {
        insert_pool_owned.set_datum(i, raw.get_datum(cfg.build_size + i));
    }

    fmt::print(
        "Build set: [0, {}). Insert pool: [{}, {}) ({} vectors).\n",
        cfg.build_size,
        cfg.build_size,
        cfg.build_size + pool_size,
        pool_size
    );

    auto data = svs::data::BlockedData<float>(cfg.build_size, raw.dimensions());
    for (size_t i = 0; i < cfg.build_size; ++i) {
        data.set_datum(i, raw.get_datum(i));
    }

    std::vector<size_t> ids(cfg.build_size);
    std::iota(ids.begin(), ids.end(), size_t{0});

    svs::index::vamana::VamanaBuildParameters build_params{
        1.2f,                 // alpha
        cfg.graph_max_degree, // graph max degree
        128,                  // construction search window size
        1024,                 // max candidate pool size
        60,                   // prune to degree
        true,                 // full search history
    };

    fmt::print("\nBuilding index over {} vectors ...\n", cfg.build_size);
    auto build_start = Clock::now();
    Index index(build_params, std::move(data), ids, DistType(), cfg.num_index_threads);
    fmt::print("Build took {:.3f}s.\n", to_seconds(Clock::now() - build_start));

    print_memory_breakdown("start of run", index.get_memory_breakdown());

    fmt::print("Loading {} ...\n", cfg.test_path);
    auto queries = svs::data::SimpleData<float>::load(cfg.test_path);
    fmt::print("Loaded {} queries.\n", queries.size());
    if (queries.dimensions() != raw.dimensions()) {
        throw ANNEXCEPTION(
            "Query dimensions ({}) do not match training dimensions ({})!",
            queries.dimensions(),
            raw.dimensions()
        );
    }

    svs::index::vamana::VamanaSearchParameters search_params{};
    search_params.buffer_config_ =
        svs::index::vamana::SearchBufferConfig(cfg.search_window_size);
    // Leave search_buffer_visited_set_ at its default (false): a visited-set buffer
    // sized once at thread start would go stale as the dataset grows via add_points.
    // Without it, the per-thread scratchspace built once below stays valid for the
    // whole run regardless of how much the index grows or shrinks.

    MutatorDataState dstate;
    for (size_t i = 0; i < cfg.build_size; ++i) {
        dstate.live_ids.push_back(i);
    }
    dstate.next_id = cfg.build_size;

    SearchExclusion exclusion;
    GlobalState gstate;

    // Retained for the whole program: the mutator/repair threads read `current_ctx`
    // across point transitions, and it must never point at freed memory.
    std::vector<std::unique_ptr<PointContext>> all_point_contexts;
    std::atomic<PointContext*> current_ctx{nullptr};
    Heartbeat mutator_heartbeat{0};
    Heartbeat repair_heartbeat{0};

    bool use_absolute_rates = cfg.search_rate > 0.0;
    double absolute_mutation_rate =
        (use_absolute_rates && cfg.write_ratio > 0.0)
            ? (cfg.write_ratio / 100.0 * cfg.search_rate)
            : 0.0;

    std::thread mutator_thread(
        mutator_loop,
        std::ref(index),
        std::cref(insert_pool_owned),
        std::ref(exclusion),
        std::ref(gstate),
        cfg.compact_every_seconds,
        cfg.slow_update_threshold_s,
        cfg.reuse_empty,
        cfg.target_ratio,
        use_absolute_rates,
        absolute_mutation_rate,
        std::ref(dstate),
        std::ref(current_ctx),
        std::ref(mutator_heartbeat)
    );
    std::thread repair_thread(
        repair_loop,
        std::ref(index),
        std::ref(exclusion),
        std::ref(gstate),
        cfg.repair_every,
        std::cref(cfg.repair_mode),
        cfg.slice_k,
        std::ref(current_ctx),
        std::ref(repair_heartbeat)
    );

    bool aborted = false;
    for (size_t point_idx = 0; point_idx < cfg.searcher_sweep.size() && !aborted;
         ++point_idx) {
        size_t n = cfg.searcher_sweep[point_idx];
        fmt::print(
            "\n=== Operating point {}/{}: {} searcher threads, {:.0f}s warmup + {} x "
            "{:.0f}s measurement windows ===\n",
            point_idx + 1,
            cfg.searcher_sweep.size(),
            n,
            cfg.warmup_seconds,
            cfg.num_windows,
            cfg.window_seconds
        );

        auto ctx = std::make_unique<PointContext>(cfg.num_windows);
        auto point_start = Clock::now();
        auto warmup_end =
            point_start + std::chrono::duration_cast<Clock::duration>(
                              std::chrono::duration<double>(cfg.warmup_seconds)
                          );
        ctx->bounds.num_windows = cfg.num_windows;
        for (size_t w = 0; w < cfg.num_windows; ++w) {
            ctx->bounds.starts[w] =
                warmup_end +
                std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(
                    static_cast<double>(w) * cfg.window_seconds
                ));
            ctx->bounds.ends[w] = ctx->bounds.starts[w] +
                                  std::chrono::duration_cast<Clock::duration>(
                                      std::chrono::duration<double>(cfg.window_seconds)
                                  );
        }
        PointContext* ctx_raw = ctx.get();
        all_point_contexts.push_back(std::move(ctx));
        current_ctx.store(ctx_raw, std::memory_order_release);

        std::atomic<bool> point_stop{false};
        std::vector<std::vector<LatencyRecorder>> search_window_recorders(n);
        for (auto& v : search_window_recorders) {
            v.resize(cfg.num_windows);
        }
        std::vector<Heartbeat> search_heartbeats(n);

        double rate_per_thread =
            use_absolute_rates ? cfg.search_rate / static_cast<double>(n) : 0.0;

        std::vector<std::thread> searchers;
        searchers.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            searchers.emplace_back(
                searcher_loop,
                i,
                std::cref(index),
                std::cref(queries),
                std::cref(search_params),
                std::ref(exclusion),
                std::ref(gstate),
                std::cref(point_stop),
                std::cref(ctx_raw->bounds),
                std::ref(search_window_recorders[i]),
                std::ref(ctx_raw->search_completed),
                n,
                rate_per_thread,
                point_start,
                std::ref(search_heartbeats[i])
            );
        }

        auto sleep_until_or_abort = [&](Clock::time_point deadline) {
            while (Clock::now() < deadline) {
                if (gstate.failed.load(std::memory_order_relaxed)) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return true;
        };

        if (!sleep_until_or_abort(warmup_end)) {
            aborted = true;
        } else {
            for (size_t w = 0; w < cfg.num_windows && !aborted; ++w) {
                auto grace = std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(cfg.window_grace_ms / 1000.0)
                );
                if (!sleep_until_or_abort(ctx_raw->bounds.ends[w] + grace)) {
                    aborted = true;
                    break;
                }

                for (auto& h : search_heartbeats) {
                    wait_for_visibility(h);
                }
                wait_for_visibility(mutator_heartbeat);
                wait_for_visibility(repair_heartbeat);

                LatencyRecorder merged_search;
                for (auto& v : search_window_recorders) {
                    merged_search.merge(v[w]);
                }
                const auto& uw = ctx_raw->update_windows[w];

                std::string label = fmt::format(
                    "point {} ({} searchers) / window {}", point_idx + 1, n, w + 1
                );
                fmt::print("\n--- {} ---\n", label);
                fmt::print("-- Searches --\n");
                print_percentiles("searches", summarize(merged_search));
                fmt::print("-- Updates --\n");
                print_percentiles("updates", summarize(uw.recorder));
                fmt::print(
                    "slow updates (> {:.3f}s): {} / {} ({:.4f}%)\n",
                    cfg.slow_update_threshold_s,
                    uw.slow_count,
                    uw.recorder.count(),
                    uw.recorder.count() == 0 ? 0.0
                                             : 100.0 * static_cast<double>(uw.slow_count) /
                                                   static_cast<double>(uw.recorder.count())
                );
                print_ratio_report(
                    label, compute_ratio_report(merged_search.count(), uw.recorder.count())
                );
                fmt::print(
                    "[{}] ACHIEVED RATES -- searches/s = {:.3f} updates/s = {:.3f} "
                    "(requested {:.3f} / {:.3f})\n",
                    label,
                    static_cast<double>(merged_search.count()) / cfg.window_seconds,
                    static_cast<double>(uw.recorder.count()) / cfg.window_seconds,
                    cfg.search_rate,
                    absolute_mutation_rate
                );
                fmt::print(
                    "search QPS = {:.3f}  update ops/s = {:.3f}\n",
                    static_cast<double>(merged_search.count()) / cfg.window_seconds,
                    static_cast<double>(uw.recorder.count()) / cfg.window_seconds
                );
                const auto& repairs = ctx_raw->repairs[w];
                double repair_total = 0.0;
                fmt::print("-- Repair (consolidate) calls: {} --\n", repairs.size());
                for (size_t i = 0; i < repairs.size(); ++i) {
                    fmt::print(
                        "  [{}] start={:.3f}s duration={:.3f}s\n",
                        i,
                        repairs[i].start_s,
                        repairs[i].duration_s
                    );
                    repair_total += repairs[i].duration_s;
                }
                fmt::print("  total repair time = {:.3f}s\n", repair_total);
            }
        }

        point_stop.store(true, std::memory_order_relaxed);
        for (auto& t : searchers) {
            t.join();
        }
    }

    gstate.stop.store(true, std::memory_order_relaxed);
    mutator_thread.join();
    repair_thread.join();

    if (gstate.failed.load(std::memory_order_relaxed)) {
        std::cerr << "\n=== Harness aborted: a thread threw ===\n";
        for (const auto& e : gstate.errors) {
            std::cerr << e << "\n";
        }
        return EXIT_FAILURE;
    }

    print_memory_breakdown("end of run", index.get_memory_breakdown());
    print_proc_status_rss();

    return EXIT_SUCCESS;
}

SVS_DEFINE_MAIN()
