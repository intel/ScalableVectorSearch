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

#pragma once

#include "svs/concepts/distance.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <type_traits>

namespace svs::index::vamana {

/// Build-time measurement counters. All distance counts and durations are summed over
/// every worker thread; `max_search_distances_per_insert` is a max over inserts instead.
struct BuildStats {
    uint64_t inserts = 0;
    uint64_t search_distances = 0;
    uint64_t pool_distances = 0;
    uint64_t prune_distances = 0;
    uint64_t backedge_prune_distances = 0;
    uint64_t search_ns = 0;
    uint64_t prune_ns = 0;
    uint64_t backedge_prune_ns = 0;
    uint64_t max_search_distances_per_insert = 0;

    BuildStats& operator+=(const BuildStats& other) {
        inserts += other.inserts;
        search_distances += other.search_distances;
        pool_distances += other.pool_distances;
        prune_distances += other.prune_distances;
        backedge_prune_distances += other.backedge_prune_distances;
        search_ns += other.search_ns;
        prune_ns += other.prune_ns;
        backedge_prune_ns += other.backedge_prune_ns;
        max_search_distances_per_insert = std::max(
            max_search_distances_per_insert, other.max_search_distances_per_insert
        );
        return *this;
    }
};

namespace detail {
inline std::mutex& build_stats_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline BuildStats& build_stats_accumulator() {
    static BuildStats stats{};
    return stats;
}
} // namespace detail

/// Return a copy of the process-wide accumulator.
inline BuildStats build_stats_snapshot() {
    std::lock_guard lock(detail::build_stats_mutex());
    return detail::build_stats_accumulator();
}

/// Zero the process-wide accumulator.
inline void build_stats_reset() {
    std::lock_guard lock(detail::build_stats_mutex());
    detail::build_stats_accumulator() = BuildStats{};
}

/// Fold a thread's local counters into the process-wide accumulator.
/// Call once per thread, after that thread's `parallel_for` body finishes, so the hot
/// path itself touches no shared state.
inline void build_stats_merge(const BuildStats& stats) {
    std::lock_guard lock(detail::build_stats_mutex());
    detail::build_stats_accumulator() += stats;
}

/// Nanoseconds elapsed since `start`, for summing into a `BuildStats` duration field.
inline uint64_t build_stats_elapsed_ns(std::chrono::steady_clock::time_point start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
    );
}

/// Wraps `Dist` by reference, counting each distance::compute call into an unsynchronized
/// uint64_t& counter; forwards compare/implicit_broadcast/must_fix_argument from Dist.
/// Select PruneStrategy from the unwrapped Dist: it has no specialization for this type.
template <typename Dist> class CountingDistance {
  public:
    using compare = distance::compare_t<Dist>;
    static constexpr bool implicit_broadcast = distance::implicitly_broadcastable<Dist>();
    static constexpr bool must_fix_argument = distance::fix_argument_mandated<Dist>();

    CountingDistance(Dist& distance, uint64_t& counter)
        : distance_{distance}
        , counter_{counter} {}

    template <typename A, typename B> float compute(A a, B b) {
        ++counter_;
        return distance::compute(distance_, a, b);
    }

    // Constrained to Dist's own fix_argument support, so ShouldFix matches Dist exactly.
    template <typename A>
        requires requires(Dist& d, A a) { d.fix_argument(a); }
    void fix_argument(A a) {
        distance_.fix_argument(a);
    }

    auto comparator() const { return distance::comparator(distance_); }

  private:
    Dist& distance_;
    uint64_t& counter_;
};

/// No-op build-stats collector: every member is empty, so binding it into the build hot
/// path costs nothing. Mirrors `CountingBuildStats`'s interface exactly.
class NullBuildStats {
  public:
    struct ScopedTimer {};

    ScopedTimer time_search() { return {}; }
    ScopedTimer time_prune() { return {}; }
    ScopedTimer time_backedge_prune() { return {}; }

    template <typename Dist> Dist& wrap_search(Dist& dist) { return dist; }
    template <typename Dist> Dist& wrap_pool(Dist& dist) { return dist; }
    template <typename Dist> Dist& wrap_prune(Dist& dist) { return dist; }
    template <typename Dist> Dist& wrap_backedge_prune(Dist& dist) { return dist; }

    void finish_search() {}
    void finish_pool() {}
    void finish_prune() {}
    void merge() {}
};

/// Build-stats collector that actually counts: one instance per worker thread, merged
/// into the process-wide accumulator once that thread's work is done.
class CountingBuildStats {
  public:
    // Adds its own elapsed time into `target` on destruction; never copied or moved, so
    // that add happens exactly once per timed scope.
    class ScopedTimer {
      public:
        explicit ScopedTimer(uint64_t& target)
            : target_{&target}
            , start_{std::chrono::steady_clock::now()} {}
        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;
        ScopedTimer(ScopedTimer&&) = delete;
        ScopedTimer& operator=(ScopedTimer&&) = delete;
        ~ScopedTimer() { *target_ += build_stats_elapsed_ns(start_); }

      private:
        uint64_t* target_;
        std::chrono::steady_clock::time_point start_;
    };

    ScopedTimer time_search() { return ScopedTimer{current_.search_ns}; }
    ScopedTimer time_prune() { return ScopedTimer{current_.prune_ns}; }
    ScopedTimer time_backedge_prune() { return ScopedTimer{current_.backedge_prune_ns}; }

    // Resets its own per-call counter, so the caller must invoke this once per
    // search/pool/prune step rather than reusing a previously returned wrapper.
    template <typename Dist> CountingDistance<Dist> wrap_search(Dist& dist) {
        search_count_ = 0;
        return CountingDistance<Dist>{dist, search_count_};
    }
    template <typename Dist> CountingDistance<Dist> wrap_pool(Dist& dist) {
        pool_count_ = 0;
        return CountingDistance<Dist>{dist, pool_count_};
    }
    template <typename Dist> CountingDistance<Dist> wrap_prune(Dist& dist) {
        prune_count_ = 0;
        return CountingDistance<Dist>{dist, prune_count_};
    }

    // Backedge counting spans the whole thread's work, not one node, so this is called
    // once per thread and the returned wrapper is reused across every source vertex.
    template <typename Dist> CountingDistance<Dist> wrap_backedge_prune(Dist& dist) {
        return CountingDistance<Dist>{dist, backedge_count_};
    }

    void finish_search() {
        current_.search_distances += search_count_;
        current_.max_search_distances_per_insert =
            std::max(current_.max_search_distances_per_insert, search_count_);
        ++current_.inserts;
    }
    void finish_pool() { current_.pool_distances += pool_count_; }
    void finish_prune() { current_.prune_distances += prune_count_; }

    /// Call once per thread, after that thread's loop over nodes finishes.
    void merge() {
        current_.backedge_prune_distances = backedge_count_;
        build_stats_merge(current_);
    }

  private:
    BuildStats current_{};
    uint64_t search_count_ = 0;
    uint64_t pool_count_ = 0;
    uint64_t prune_count_ = 0;
    uint64_t backedge_count_ = 0;
};

/// The collector `VamanaBuilder` uses by default. Both collectors above always compile,
/// so only this selection depends on the build-time flag.
using DefaultBuildStats = std::
    conditional_t<static_cast<bool>(SVS_ENABLE_STATS), CountingBuildStats, NullBuildStats>;

} // namespace svs::index::vamana
