/*
 * Copyright 2025 Intel Corporation
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

// svs
#include "svs/orchestrators/ivf.h"

// stl
#include <functional>
#include <string>

namespace svsbenchmark {

template <> struct IndexTraits<svs::IVF> {
    using index_type = svs::IVF;
    using config_type = svs::index::ivf::IVFSearchParameters;
    using state_type = svsbenchmark::ivf::IVFState;
    static std::string name() { return "static ivf index"; }

    // Configuration Space.
    static void apply_config(svs::IVF& index, const config_type& config) {
        index.set_search_parameters(config);
    }

    template <svs::data::ImmutableMemoryDataset Queries>
    static auto search(
        svs::IVF& index,
        const Queries& queries,
        size_t num_neighbors,
        const config_type& config
    ) {
        apply_config(index, config);
        return index.search(queries, num_neighbors);
    }

    static state_type report_state(const svs::IVF& index) { return state_type(index); }

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
        config_type baseline = index.get_search_parameters();

        auto valid_configurations = std::vector<config_type>();

        // Search over epsilon search space first.
        auto k_reorders = std::vector<size_t>({1, 4, 10});
        auto n_probes_range = svs::threads::UnitRange<size_t>(1, 200);
        for (auto k_reorder : k_reorders) {
            auto copy = baseline;
            copy.k_reorder_ = k_reorder;

            copy.n_probes_ = *std::lower_bound(
                n_probes_range.begin(),
                n_probes_range.end(),
                target_recall,
                [&](size_t n_probes, double recall) {
                    copy.n_probes_ = n_probes;
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
