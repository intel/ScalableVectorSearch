/*
 * Copyright 2024 Intel Corporation
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
#include "svs/core/logging.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/search_params.h"
#include "svs/lib/threads/types.h"
#include "svs/lib/timing.h"

// third-party
#include "svs/third-party/fmt.h"

// stl
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

namespace svs::index::vamana {

// Algorithm overview:
//
// The calibration algorithm currently tunes the following two parameters:
// * search_window_size: The effective size of the search buffer that determines when
//     a greedy search terminates.
//
// * search_buffer_capacity: The number of candidates maintained in the search buffer
//     during the greedy search.
//
// Invariants:
// * `search_buffer_capacity >= search_window_size`: This constraint is fairly
//      straight-forward and is imposed by the `SearchBufferConfig` class.
//
// * `search_buffer_capacity >= num_neighbors`: The index implementations currently
//      detect when this contraints if violated and silently set the search window size
//      to `num_neighbors` in this case because they depend on the search buffer
//      containing at least `num_neighbors`.
//
//      This introduces an abrupt change in recall behavior if we ever set
//      `search_buffer_capacity` less thaN `num_neighbors`, so try to avoid that situation.
// Assumptions:
// * Recall is monotonic in `search_window_size` when not using a split buffer. This is
//      a fairly solid assumption as searches with a smaller search buffer are (almost)
//      guraenteed to be prefixes of searches with a larger buffer.
//
// * At a fixed `search_window_size`, recall is monotonic with `search_buffer_capacity`.
//      This assumption is a little less solid as reranking occasionally interacts poorly
//      with recall. We're using this for now, but may need to revisit.
//
// It is useful to set the parameter `search_buffer_capacity` to a value less than
// `search_window_size` in two cases:
//
// 1. The dataset uses reranking after the primary graph search to refine distance
//    computations. Tracking more neighbors increases the probability that a true
//    neighbor is captures.
// 2. The search window size in the non-split buffer configuration is equal to the
//    number of neighbors (due to the clamping effect implemented by the graph indexes).
//    In this case, we can benefit by decreasing `search_window_size` but keeping
//    `search_buffer_capacity` fixed at `num_neighbors`.
//
// The algorithm first conducts a binary search over `search_window_size` for a non-split
// buffer to determing an upper bound on the search window size.
//
// Then determine if a split buffer may be useful by either checking for an extension to
// `svs::index::vamana::extensions::UsesReranking` or by detecting if we're operating
// in case 2 above.
//
// If a split buffer is to be used, then we test successively smaller search window sizes
// with a binary search over `search_buffer_capacity`. When a valid combination if found,
// it is benchmarked and compared with the current best.
//
// This process continues until the target recall can no longer be achieved, at which point
// the algorithm terminates.

struct CalibrationParameters {
    enum class SearchBufferOptimization { Disable, All, ROIOnly, ROITuneUp };

    ///// Parameters controlling bounds on aspects of the algorithm.

    /// The maximum search window size to try.
    size_t search_window_size_upper_ = 1000;
    /// The maximum search window capacity to try.
    size_t search_window_capacity_upper_ = 1000;
    /// The maximum number of search iterations to use when determining search time.
    size_t timing_iterations_ = 5;
    /// The number of seconds before search times-out.
    double search_timeout_ = 0.125;
    /// The steps to use when training prefetchers.
    std::vector<size_t> prefetch_steps_ = {1, 2, 4};

    ///// Flags determining which aspects of the algorithm will be run.

    /// What aspect of the search buffer will be optimized.
    SearchBufferOptimization search_buffer_optimization_ = SearchBufferOptimization::All;
    /// Do we train the prefetchers as well?
    bool train_prefetchers_ = true;
    /// Should we obtain untrained parameters from default values or from the index.
    bool use_existing_parameter_values_ = true;

    ///// Member Functions
    bool should_optimize_search_buffer() const {
        return search_buffer_optimization_ != SearchBufferOptimization::Disable;
    }
};

namespace calibration {

template <typename DoSearch>
double get_search_time(
    const CalibrationParameters& calibration_parameters,
    const DoSearch& do_search,
    const VamanaSearchParameters& parameters
) {
    double min_time = std::numeric_limits<double>::max();
    auto start_time = lib::now();

    auto max_iterations = calibration_parameters.timing_iterations_;
    auto search_timeout = calibration_parameters.search_timeout_;

    for (size_t i = 0; i < max_iterations; ++i) {
        auto tic = lib::now();
        do_search(parameters);
        auto toc = lib::now();
        min_time = std::min(min_time, lib::time_difference(toc, tic));

        // Detect timeout.
        double elapsed = lib::time_difference(toc, start_time);
        if (elapsed > search_timeout) {
            break;
        }
    }
    return min_time;
}

template <typename F>
VamanaSearchParameters optimize_split_buffer_using_binary_search(
    double target_recall, VamanaSearchParameters current, const F& compute_recall
) {
    size_t current_capacity = current.buffer_config_.get_total_capacity();
    auto range = svs::threads::UnitRange<size_t>(1, current_capacity);

    auto search_window_size = *std::lower_bound(
        range.begin(),
        range.end(),
        target_recall,
        [&](size_t window_size, double recall) {
            current.buffer_config({window_size, current_capacity});
            double this_recall = compute_recall(current);
            return this_recall < recall;
        }
    );
    current.buffer_config({search_window_size, current_capacity});
    return current;
}

template <typename F, typename DoSearch>
VamanaSearchParameters optimize_split_buffer(
    const CalibrationParameters& calibration_parameters,
    size_t num_neighbors,
    double target_recall,
    VamanaSearchParameters current,
    const F& compute_recall,
    const DoSearch& do_search
) {
    auto logger = svs::logging::get();
    svs::logging::trace(logger, "Entering split buffer optimization routine");
    assert(
        current.buffer_config_.get_search_window_size() ==
        current.buffer_config_.get_total_capacity()
    );

    // Get the timing for the baseline search.
    double min_search_time = get_search_time(calibration_parameters, do_search, current);

    // Now, start experimenting.
    size_t sws = current.buffer_config_.get_search_window_size();
    svs::logging::trace(
        logger, "Search time with uniform buffer with size {}: {}s", sws, min_search_time
    );
    svs::logging::trace(logger, "Trying to achieve recall {}", target_recall);

    // Copy the current state of the search parameters so we only tweak the buffer config.
    size_t search_window_capacity_upper =
        calibration_parameters.search_window_capacity_upper_;
    auto sp = current;
    while (sws > 1) {
        --sws;
        // First, try using the largest search window capacity.
        // If that fails, then we shouldn't see any better progress by further decreasing
        // the search window size and we can terminate now.
        sp.buffer_config({sws, search_window_capacity_upper});
        svs::logging::trace(logger, "Trying search window size {} ...", sws);
        if (compute_recall(sp) < target_recall) {
            svs::logging::trace(logger, "Search window size {} failed", sws);
            return current;
        }
        svs::logging::trace(logger, "Search window size {} succeeded", sws);

        // Otherwise, this search window size has a chance of working.
        // Use a binary search to determine the smallest capacity that achieves the desired
        // recall.
        // Then time that configuration.
        // If it's faster than what we've found so far, then update the current best.
        auto capacity_lower_bound = std::max<size_t>(sws, num_neighbors);
        auto range =
            threads::UnitRange<size_t>(capacity_lower_bound, search_window_capacity_upper);
        auto best_capacity = *std::lower_bound(
            range.begin(),
            range.end(),
            target_recall,
            [&](size_t capacity, double recall) {
                sp.buffer_config({sws, capacity});
                auto r = compute_recall(sp);
                svs::logging::trace(logger, "recall = {}", r);
                return r < recall;
            }
        );
        sp.buffer_config({sws, best_capacity});
        double search_time = get_search_time(calibration_parameters, do_search, sp);
        svs::logging::trace(
            logger, "Best capacity: {}, Search time: {}", best_capacity, search_time
        );
        if (search_time < min_search_time) {
            min_search_time = search_time;
            current = sp;
        }
    }
    return current;
}

// Return a pair of the best search parameters found so far and the convergence state.
template <typename Index, typename ComputeRecall, typename DoSearch>
std::pair<VamanaSearchParameters, bool> optimize_search_buffer(
    const CalibrationParameters& calibration_parameters,
    VamanaSearchParameters current,
    size_t num_neighbors,
    double target_recall,
    const ComputeRecall& compute_recall,
    const DoSearch& do_search
) {
    using enum CalibrationParameters::SearchBufferOptimization;
    using dataset_type = typename Index::data_type;
    auto logger = svs::logging::get();

    double max_recall = std::numeric_limits<double>::lowest();
    const size_t current_capacity = current.buffer_config_.get_total_capacity();
    auto state = calibration_parameters.search_buffer_optimization_;
    bool use_current_capacity = (state == ROITuneUp);
    auto configure_current_buffer = [&](size_t search_window_size) {
        if (use_current_capacity) {
            size_t this_capacity = std::max(search_window_size, current_capacity);
            current.buffer_config({search_window_size, this_capacity});
        } else {
            current.buffer_config(search_window_size);
        }
    };

    // Compute the lower bound for the search window size.
    // If we are using the current buffer capacity, then we can set the search window size
    // all the way to "1" is the capacity is greater-than or equal to the number of
    // neighbors.
    size_t range_lower =
        (use_current_capacity && current_capacity >= num_neighbors) ? 1 : num_neighbors;

    auto range = svs::threads::UnitRange<size_t>(
        range_lower, calibration_parameters.search_window_size_upper_
    );

    // In all cases - we're going to optimize the search window size.
    auto search_window_size = *std::lower_bound(
        range.begin(),
        range.end(),
        target_recall,
        [&](size_t window_size, double recall) {
            configure_current_buffer(window_size);
            double this_recall = compute_recall(current);
            svs::logging::trace(
                logger, "Trying {}, got {}. Target: {}", window_size, this_recall, recall
            );
            max_recall = std::max(max_recall, this_recall);
            return this_recall < recall;
        }
    );

    // Determine if we want to optimize for split-buffer operation as well.

    // Force exit.
    bool converged = max_recall >= target_recall;
    bool exit_now = (state != All) || !converged;

    // Continuing could be helpful.
    bool maybe_oversized = (search_window_size == num_neighbors);
    bool dataset_uses_reranking = extensions::calibration_uses_reranking<dataset_type>();
    bool split_buffer_could_be_helpful = maybe_oversized || dataset_uses_reranking;

    // Configure the search parameters to the best window size found so far.
    // Then determine how we will proceed.
    configure_current_buffer(search_window_size);
    if (exit_now || !split_buffer_could_be_helpful) {
        // Return the best found parameters.
        return std::make_pair(current, converged);
    }

    // If the dataset does not use reranking, then we're in a situation where the
    // search window size is less than the number of neighbors.
    //
    // In this case, we can be faster and binary search to the window size with the
    // capacity fixed to the number of neighbors.
    if (!dataset_uses_reranking) {
        current = optimize_split_buffer_using_binary_search(
            target_recall, current, compute_recall
        );
    } else {
        // Optimize the split-buffer using a generic exhaustive search.
        current = optimize_split_buffer(
            calibration_parameters,
            num_neighbors,
            target_recall,
            current,
            compute_recall,
            do_search
        );
    }
    return std::make_pair(current, converged);
}

template <typename Index, typename DoSearch>
VamanaSearchParameters tune_prefetch(
    const CalibrationParameters& calibration_parameters,
    Index& index,
    VamanaSearchParameters search_parameters,
    const DoSearch& do_search
) {
    auto logger = svs::logging::get();
    svs::logging::trace(logger, "Tuning prefetch parameters");
    const auto& prefetch_steps = calibration_parameters.prefetch_steps_;
    size_t max_lookahead = index.max_degree();

    // Start with no prefetching.
    search_parameters.prefetch_lookahead_ = 0;
    search_parameters.prefetch_step_ = 0;
    double min_search_time =
        get_search_time(calibration_parameters, do_search, search_parameters);
    svs::logging::trace(logger, "Time with no prefetching: {}s", min_search_time);

    // Create a local copy of `search_parameters` to mutate.
    auto sp = search_parameters;

    auto visited_lookaheads = std::unordered_map<int64_t, double>{};
    auto search_with_refinement =
        [&](int64_t lookahead_lower, int64_t lookahead_upper, int64_t lookahead_step) {
            double best_time = std::numeric_limits<double>::max();
            auto best_l = std::numeric_limits<int64_t>::max();
            for (auto l = lookahead_lower; l < lookahead_upper; l += lookahead_step) {
                // Look if we have a cached run-time.
                auto itr = visited_lookaheads.find(l);
                if (itr == visited_lookaheads.end()) {
                    // Compute a new run-time;
                    sp.prefetch_lookahead_ = l;
                    auto run_time = get_search_time(calibration_parameters, do_search, sp);
                    itr = visited_lookaheads.insert(itr, {l, run_time});
                }

                // Access cached run-time.[
                double time = itr->second;
                svs::logging::trace(logger, "Tried {}, got {}", l, time);
                if (time < best_time) {
                    best_time = time;
                    best_l = l;
                }
            }
            return std::make_pair(best_l, best_time);
        };

    for (auto step : prefetch_steps) {
        sp.prefetch_step_ = step;
        svs::logging::trace(logger, "Trying prefetch step {}", step);
        visited_lookaheads.clear();

        int64_t lookahead_step = lib::narrow<int64_t>(max_lookahead) / 4;
        int64_t lookahead_start = 1;
        int64_t lookahead_stop = lib::narrow<int64_t>(max_lookahead);

        // First - try with the maximum lookahead value.
        {
            sp.prefetch_lookahead_ = max_lookahead;
            auto search_time = get_search_time(calibration_parameters, do_search, sp);
            if (search_time < min_search_time) {
                search_parameters.prefetch_lookahead_ = sp.prefetch_lookahead_;
                search_parameters.prefetch_step_ = sp.prefetch_step_;
            }
        }

        // Perform successive refinement.
        while (lookahead_step != 0) {
            svs::logging::trace(
                logger,
                "Running refinement with {}:{}:{}",
                lookahead_start,
                lookahead_step,
                lookahead_stop
            );
            auto [best_lookahead, search_time] = search_with_refinement(
                std::max<int64_t>(lookahead_start, 1),
                std::min<int64_t>(lookahead_stop, max_lookahead),
                lookahead_step
            );

            if (search_time < min_search_time) {
                min_search_time = search_time;
                search_parameters.prefetch_lookahead_ = best_lookahead;
                search_parameters.prefetch_step_ = step;
                svs::logging::trace(
                    logger,
                    "Replacing prefetch parameters to {}, {} at {}s",
                    search_parameters.prefetch_lookahead_,
                    search_parameters.prefetch_step_,
                    search_time
                );
            }

            // Refine the search parameters in the local area of the bes results.
            lookahead_step /= 2;
            lookahead_start = best_lookahead - 2 * lookahead_step;
            lookahead_stop = best_lookahead + 2 * lookahead_step;
        }
    }
    return search_parameters;
}

} // namespace calibration

// TODO: Use calibration parameters to allow finer-control of the search space.
///
/// @brief Calibrate the search parameters for maximum performance.
///
/// @param calibration_parameters Configuration class with the hyper-parameters for search
///     optimization.
/// @param index The index being calibrated.
/// @param num_neighbors The number of neighbors desired.
/// @param target_recall The desired `num_neighbors` recall at `num_neighbors`.
/// @param compute_recall A zero-argument callable that returns the recall of the
///     currently configured index.
///
///     It must internally contain a reference to ``index`` so that changes to the
///     search parameters are visible.
/// @param do_search A zero-argument callable that simply performs a search over the index
///     using the current configuration. This callable will be used for benchmarking to
///     determine the performance of various configurations.
///
/// @returns The search parameters discovered that yield the highest QPS over the
///     implicit query-set while meeting the target recall.
///
/// When calibration terminates, ``index`` will be configured with the best discovered
/// search parameters
///
/// If the desired recall is unreachable for any configuration, the index will instead
/// be configured in the setup that achieves the highest recall.
///
template <typename Index, typename F, typename DoSearch>
VamanaSearchParameters calibrate(
    const CalibrationParameters& calibration_parameters,
    Index& index,
    size_t num_neighbors,
    double target_recall,
    F&& compute_recall,
    DoSearch&& do_search
) {
    // Get the existing parameters and the default values decide which to use as the seed.
    auto default_parameters = VamanaSearchParameters();
    auto preset_parameters = index.get_search_parameters();

    auto current = calibration_parameters.use_existing_parameter_values_
                       ? preset_parameters
                       : default_parameters;

    // Step 1: Optimize aspects of the search buffer if desired.
    if (calibration_parameters.should_optimize_search_buffer()) {
        svs::logging::trace("Optimizing search buffer.");
        auto [best, converged] = calibration::optimize_search_buffer<Index>(
            calibration_parameters,
            current,
            num_neighbors,
            target_recall,
            compute_recall,
            do_search
        );
        current = best;

        if (!converged) {
            svs::logging::warn(
                "Target recall could not be achieved. Exiting optimization early."
            );
            return current;
        }
    }

    // Step 2: Optimize prefetch parameters.
    if (calibration_parameters.train_prefetchers_) {
        svs::logging::trace("Training Prefetchers.");
        current =
            calibration::tune_prefetch(calibration_parameters, index, current, do_search);
    }

    // Finish up.
    return current;
}

} // namespace svs::index::vamana

// print search-parameters enums.
template <>
struct fmt::formatter<svs::index::vamana::CalibrationParameters::SearchBufferOptimization>
    : svs::format_empty {
    using SBO = svs::index::vamana::CalibrationParameters::SearchBufferOptimization;
    using enum svs::index::vamana::CalibrationParameters::SearchBufferOptimization;

    static std::string_view get_name(SBO x) {
        switch (x) {
            case Disable: {
                return "Disable";
            }
            case All: {
                return "All";
            }
            case ROIOnly: {
                return "ROIOnly";
            }
            case ROITuneUp: {
                return "ROITuneUp";
            }
        }
        throw ANNEXCEPTION("Unreachable reached!");
    }

    auto format(const auto& x, auto& ctx) const {
        return fmt::format_to(ctx.out(), "{}", get_name(x));
    }
};
