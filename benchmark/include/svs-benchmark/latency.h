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

// svs
#include "svs/lib/exception.h"

// stl
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace svsbenchmark {

///
/// Percentile of a sorted sample using the nearest-rank method: for a percentile `p` in
/// `[0, 100]` and `n` ascending samples, the result is `sorted[k]` with
/// `k = ceil(p / 100 * n) - 1` clamped to `[0, n - 1]`. No interpolation between
/// neighboring samples is performed.
///
/// @param sorted Samples in ascending order. Behavior is undefined if unsorted.
/// @param percentile Value in `[0, 100]`.
///
inline double nearest_rank_percentile(std::span<const double> sorted, double percentile) {
    auto n = sorted.size();
    if (n == 0) {
        return 0.0;
    }
    auto rank =
        static_cast<int64_t>(std::ceil(percentile / 100.0 * static_cast<double>(n)));
    auto k = std::clamp(rank - 1, int64_t{0}, static_cast<int64_t>(n) - 1);
    return sorted[static_cast<size_t>(k)];
}

///
/// A log-linear histogram for merging per-thread latency samples and recovering
/// arbitrary percentiles without retaining the raw samples.
///
/// Values are binned by octave (power-of-two range) and linearly subdivided within each
/// octave, giving a bounded relative error of `1 / (2 * subdivisions)` regardless of the
/// value's magnitude - the same trade-off HdrHistogram makes, without its sub-bucket
/// index arithmetic. Values outside `[lowest, highest]` are clamped into the nearest
/// octave, which biases extreme outliers but keeps the bucket count finite.
///
class LatencyHistogram {
  public:
    // Linear subdivisions per power-of-two octave; bounds the worst-case relative
    // recovery error to 1 / (2 * subdivisions_per_octave).
    static constexpr size_t subdivisions_per_octave = 256;
    static constexpr double default_lowest_trackable_value = 1e-9;
    static constexpr double default_highest_trackable_value = 1000.0;

    LatencyHistogram(
        double lowest_trackable_value = default_lowest_trackable_value,
        double highest_trackable_value = default_highest_trackable_value
    )
        : lowest_{lowest_trackable_value}
        // Validating here, rather than in the constructor body, guarantees the throw
        // happens before `octaves_`/`counts_` are sized from unchecked values.
        , highest_{check_bounds(lowest_trackable_value, highest_trackable_value)}
        , octaves_{compute_octaves(lowest_trackable_value, highest_trackable_value)}
        , counts_(octaves_ * subdivisions_per_octave, 0) {}

    // Record `value`, clamping into `[lowest, highest]` first.
    void record(double value) {
        auto clamped = std::clamp(value, lowest_, highest_);
        ++counts_[bucket_for(clamped)];
        ++total_count_;
    }

    // Add `other`'s counts into this histogram. Both histograms must share the same
    // bucket layout - merging across differing `lowest`/`highest` would silently
    // misattribute counts to the wrong value ranges.
    void merge(const LatencyHistogram& other) {
        if (lowest_ != other.lowest_ || highest_ != other.highest_) {
            throw ANNEXCEPTION("Cannot merge LatencyHistograms with differing bounds!");
        }
        for (size_t i = 0; i < counts_.size(); ++i) {
            counts_[i] += other.counts_[i];
        }
        total_count_ += other.total_count_;
    }

    // Recover `percentile` (in `[0, 100]`) via the nearest-rank method applied to the
    // histogram's cumulative counts, returning the midpoint of the resolved bucket.
    // Returns 0.0 for an empty histogram.
    double percentile(double pct) const {
        if (total_count_ == 0) {
            return 0.0;
        }
        auto rank =
            static_cast<uint64_t>(std::ceil(pct / 100.0 * static_cast<double>(total_count_))
            );
        rank = std::clamp(rank, uint64_t{1}, total_count_);

        uint64_t cumulative = 0;
        for (size_t i = 0; i < counts_.size(); ++i) {
            cumulative += counts_[i];
            if (cumulative >= rank) {
                return bucket_midpoint(i);
            }
        }
        return bucket_midpoint(counts_.size() - 1);
    }

    uint64_t count() const { return total_count_; }

  private:
    static double check_bounds(double lowest, double highest) {
        if (!(lowest > 0.0) || !(highest > lowest)) {
            throw ANNEXCEPTION("LatencyHistogram requires 0 < lowest_trackable_value < "
                               "highest_trackable_value!");
        }
        return highest;
    }

    static size_t compute_octaves(double lowest, double highest) {
        return static_cast<size_t>(std::ceil(std::log2(highest / lowest))) + 1;
    }

    size_t bucket_for(double value) const {
        auto ratio = value / lowest_;
        // `ratio` can round to just under 1.0 when `value == lowest_`, so clamp the
        // signed octave to 0 before the unsigned cast rather than let it wrap around.
        auto octave_signed = static_cast<int64_t>(std::floor(std::log2(ratio)));
        auto octave = static_cast<size_t>(
            std::clamp(octave_signed, int64_t{0}, static_cast<int64_t>(octaves_) - 1)
        );
        auto frac = std::max(0.0, ratio / std::exp2(static_cast<double>(octave)) - 1.0);
        auto sub = static_cast<size_t>(frac * static_cast<double>(subdivisions_per_octave));
        sub = std::min(sub, subdivisions_per_octave - 1);
        return octave * subdivisions_per_octave + sub;
    }

    double bucket_midpoint(size_t bucket) const {
        auto octave = bucket / subdivisions_per_octave;
        auto sub = bucket % subdivisions_per_octave;
        auto octave_base = lowest_ * std::exp2(static_cast<double>(octave));
        auto width = octave_base / static_cast<double>(subdivisions_per_octave);
        return octave_base + width * (static_cast<double>(sub) + 0.5);
    }

    double lowest_;
    double highest_;
    size_t octaves_;
    std::vector<uint64_t> counts_;
    uint64_t total_count_ = 0;
};

///
/// Summary statistics for a collection of latency samples, named to match the
/// end-to-end benchmark's vocabulary so proxy and E2E numbers can sit side by side.
///
struct LatencyStats {
    size_t count;
    double mean_time;
    double std_time;
    double min_time;
    double max_time;
    double p50_time;
    double p95_time;
    double p99_time;
    LatencyHistogram latency_hdr;
};

///
/// Compute `LatencyStats` from raw samples. The `p50_time`/`p95_time`/`p99_time` fields
/// use `nearest_rank_percentile` (see its docstring for the exact rule); `latency_hdr`
/// is built by recording every sample so it can recover the same percentiles later.
///
/// An empty `latencies` is defined behavior: `count` is 0 and every time field is 0.0.
///
inline LatencyStats compute_latency_stats(
    std::span<const double> latencies,
    double hdr_lowest_trackable_value = LatencyHistogram::default_lowest_trackable_value,
    double hdr_highest_trackable_value = LatencyHistogram::default_highest_trackable_value
) {
    auto hdr = LatencyHistogram(hdr_lowest_trackable_value, hdr_highest_trackable_value);
    auto n = latencies.size();
    if (n == 0) {
        return LatencyStats{0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::move(hdr)};
    }

    for (double v : latencies) {
        hdr.record(v);
    }

    auto sorted = std::vector<double>(latencies.begin(), latencies.end());
    std::sort(sorted.begin(), sorted.end());

    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    double mean = sum / static_cast<double>(n);
    double sq_sum = 0.0;
    for (double v : sorted) {
        sq_sum += (v - mean) * (v - mean);
    }
    double std_dev = std::sqrt(sq_sum / static_cast<double>(n));

    return LatencyStats{
        n,
        mean,
        std_dev,
        sorted.front(),
        sorted.back(),
        nearest_rank_percentile(sorted, 50.0),
        nearest_rank_percentile(sorted, 95.0),
        nearest_rank_percentile(sorted, 99.0),
        std::move(hdr)};
}

} // namespace svsbenchmark
