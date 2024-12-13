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

// svs-benchmark
#include "svs-benchmark/index_traits.h"
#include "svs-benchmark/inverted/memory/build.h"

// svs
// #include "svs/index/inverted/memory_based.h"
#include "svs/orchestrators/inverted.h"

namespace svsbenchmark {

// Implement IndexTraits for the inverted index.
template <> struct IndexTraits<svs::Inverted> {
    using index_type = svs::Inverted;

    // Helpers.
    using config_type = svs::index::inverted::InvertedSearchParameters;
    using state_type = svsbenchmark::inverted::memory::MemoryInvertedState;

    static std::string name() { return "static inverted index"; }

    // Configuration Space.
    static void apply_config(index_type& index, const config_type& config) {
        index.set_search_parameters(config);
    }

    template <svs::data::ImmutableMemoryDataset Queries>
    static auto search(
        index_type& index,
        const Queries& queries,
        size_t num_neighbors,
        const config_type& config
    ) {
        apply_config(index, config);
        return index.search(queries, num_neighbors);
    }

    static state_type report_state(const index_type& index) {
        return state_type(index.get_search_parameters(), index.get_num_threads());
    }

    template <svs::data ::ImmutableMemoryDataset Queries, typename Groundtruth>
    static config_type calibrate(
        index_type& index,
        const Queries& queries,
        const Groundtruth& groundtruth,
        size_t num_neighbors,
        double target_recall,
        svsbenchmark::CalibrateContext SVS_UNUSED(ctx),
        svsbenchmark::Placeholder SVS_UNUSED(placeholder)
    ) {
        // First, we find valid combinations of refinement epsilon and the inner search
        // window size.
        //
        // Once we've collected a list of valid configurations, we profile each
        // configuration to find the one with the lowest latency.
        //
        // This final configuration is the one that is returned.

        // The baseline configuration that modifications will be made to.
        config_type baseline = index.get_search_parameters();

        auto valid_configurations = std::vector<config_type>();

        // Search over epsilon search space first.
        auto epsilons = std::vector<double>({0.8, 1.0, 2.0, 5.0, 10.0, 20.0});
        auto sws_range = svs::threads::UnitRange<size_t>(1, 300);
        for (auto epsilon : epsilons) {
            auto copy = baseline;
            copy.refinement_epsilon_ = epsilon;
            copy.primary_parameters_.buffer_config_ = *std::lower_bound(
                sws_range.begin(),
                sws_range.end(),
                target_recall,
                [&](size_t window_size, double recall) {
                    copy.primary_parameters_.buffer_config_ = window_size;
                    index.set_search_parameters(copy);

                    auto result = index.search(queries, num_neighbors);
                    auto this_recall = svs::k_recall_at_n(groundtruth, result);
                    return this_recall < recall;
                }
            );

            valid_configurations.push_back(copy);
        }

        // Loop through each valid configuration - find the fastest.
        size_t best_config = std::numeric_limits<size_t>::max();
        double lowest_latency = std::numeric_limits<double>::max();

        size_t config_index = 0;
        for (auto& config : valid_configurations) {
            apply_config(index, config);
            auto latencies = std::vector<double>();
            for (size_t i = 0; i < 5; ++i) {
                auto tic = svs::lib::now();
                index.search(queries, num_neighbors);
                latencies.push_back(svs::lib::time_difference(tic));
            }

            auto min_latency = *std::min_element(latencies.begin(), latencies.end());

            std::cout << svs::lib::save_to_table(config) << '\n';
            SVS_SHOW(min_latency);

            if (min_latency < lowest_latency) {
                best_config = config_index;
                lowest_latency = min_latency;
            }
            ++config_index;
        }

        return valid_configurations[best_config];
    }

    template <svs::data::ImmutableMemoryDataset Queries, typename Groundtruth>
    static config_type calibrate_with_hint(
        index_type& index,
        const Queries& queries,
        const Groundtruth& groundtruth,
        size_t num_neighbors,
        double target_recall,
        svsbenchmark::CalibrateContext ctx,
        const config_type& SVS_UNUSED(preset),
        svsbenchmark::Placeholder placeholder
    ) {
        return calibrate(
            index, queries, groundtruth, num_neighbors, target_recall, ctx, placeholder
        );
    }
};

} // namespace svsbenchmark
