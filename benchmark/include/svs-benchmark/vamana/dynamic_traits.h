#pragma once

// svs-benchmark
#include "svs-benchmark/index_traits.h"

// svs
#include "svs/index/vamana/dynamic_index.h"

namespace svsbenchmark {

template <typename Graph, typename Data, typename Dist>
struct IndexTraits<svs::index::vamana::MutableVamanaIndex<Graph, Data, Dist>> {
    using index_type = svs::index::vamana::MutableVamanaIndex<Graph, Data, Dist>;

    // Search window size.
    using config_type = svs::index::vamana::VamanaSearchParameters;
    using state_type = svsbenchmark::vamana::VamanaState;

    static std::string name() { return "dynamic vamana index"; }

    // Dynamic Operations
    template <svs::data::ImmutableMemoryDataset Points>
    static void
    add_points(index_type& index, const Points& points, const std::vector<size_t>& ids) {
        index.add_points(points, ids);
    }

    static void delete_points(index_type& index, const std::vector<size_t>& ids) {
        index.delete_entries(ids);
    }

    static void consolidate(index_type& index) {
        index.consolidate();
        index.compact();
    }

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
        return svs::index::search_batch(index, queries, num_neighbors);
    }

    static state_type report_state(const index_type& index) { return state_type(index); }

    template <svs::data::ImmutableMemoryDataset Queries, typename Groundtruth>
    static config_type calibrate(
        index_type& index,
        const Queries& queries,
        const Groundtruth& groundtruth,
        size_t num_neighbors,
        double target_recall,
        svsbenchmark::CalibrateContext ctx,
        svsbenchmark::vamana::DynamicOptimizationLevel SVS_UNUSED(opt_level)
    ) {
        // This method may only be called in the initial training set context.
        if (ctx != CalibrateContext::InitialTrainingSet) {
            throw ANNEXCEPTION("Default static calibration may only be performed on the "
                               "initial training set!");
        }
        return index.calibrate(
            queries, svs::recall_convert(groundtruth), num_neighbors, target_recall
        );
    }

    template <svs::data::ImmutableMemoryDataset Queries, typename Groundtruth>
    static config_type calibrate_with_hint(
        index_type& index,
        const Queries& queries,
        const Groundtruth& groundtruth,
        size_t num_neighbors,
        double target_recall,
        svsbenchmark::CalibrateContext ctx,
        const config_type& preset,
        svsbenchmark::vamana::DynamicOptimizationLevel opt_level
    ) {
        using SearchBufferOptimization =
            svs::index::vamana::CalibrationParameters::SearchBufferOptimization;
        using enum vamana::DynamicOptimizationLevel;

        // Handle rejected cases.
        if (ctx == svsbenchmark::CalibrateContext::InitialTrainingSet) {
            throw ANNEXCEPTION("Invalid call to calibrate!");
        }

        // Abort early if configured with minimal optimizations.
        index.set_search_parameters(preset);
        if (opt_level == Minimal && ctx == CalibrateContext::TrainingSetTune) {
            return preset;
        }

        // Don't calibrate the prefetchers.
        auto p = svs::index::vamana::CalibrationParameters();
        p.train_prefetchers_ = false;

        if (ctx == CalibrateContext::TestSetTune) {
            p.search_buffer_optimization_ = SearchBufferOptimization::ROITuneUp;
        } else {
            p.search_buffer_optimization_ = SearchBufferOptimization::All;
        }

        // Configure the index to use the preset parameters and perform the partial
        // optimization.
        return index.calibrate(
            queries, svs::recall_convert(groundtruth), num_neighbors, target_recall, p
        );
    }
};

} // namespace svsbenchmark
