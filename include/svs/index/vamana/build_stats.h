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

// Compiled out unless SVS_ENABLE_STATS is set, so including this header is always free.
#if SVS_ENABLE_STATS

#include "svs/concepts/distance.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>

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

} // namespace svs::index::vamana

#endif // SVS_ENABLE_STATS
