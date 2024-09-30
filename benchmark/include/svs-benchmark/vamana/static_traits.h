/*
 * Copyright (C) 2024 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */

#pragma once

// svs-benchmark
#include "svs-benchmark/index_traits.h"

// svs
#include "svs/orchestrators/vamana.h"

// stl
#include <functional>
#include <string>

namespace svsbenchmark {

template <> struct IndexTraits<svs::Vamana> {
    using config_type = svs::index::vamana::VamanaSearchParameters;
    using state_type = svsbenchmark::vamana::VamanaState;
    using calibration_constructor =
        std::function<svs::index::vamana::CalibrationParameters()>;

    // A callback yielding a `CalibrationParameters` instance for use in the regression
    // testing routines.
    //
    // This will do the full set of optimizations (prefetchers, split-buffer etc.).
    static calibration_constructor regression_optimization() {
        return calibration_constructor([]() {
            return svs::index::vamana::CalibrationParameters();
        });
    }

    // A callback yielding a `CalibrationParameters` instance for use in the test
    // generators.
    //
    // This will skip training prefetchers and use more conservative upper bounds.
    static calibration_constructor test_generation_optimization() {
        return calibration_constructor([]() {
            auto c = svs::index::vamana::CalibrationParameters();
            c.train_prefetchers_ = false;
            c.search_window_size_upper_ = 100;
            c.search_window_capacity_upper_ = 100;
            return c;
        });
    }

    static std::string name() { return "static vamana index (type erased)"; }

    // Configuration Space.
    static void apply_config(svs::Vamana& index, const config_type& config) {
        index.set_search_parameters(config);
    }

    template <svs::data::ImmutableMemoryDataset Queries>
    static auto search(
        svs::Vamana& index,
        const Queries& queries,
        size_t num_neighbors,
        const config_type& config
    ) {
        apply_config(index, config);
        return index.search(queries, num_neighbors);
    }

    static state_type report_state(const svs::Vamana& index) { return state_type(index); }

    // Calibrate from scratch
    template <svs::data::ImmutableMemoryDataset Queries, typename Groundtruth>
    static config_type calibrate(
        svs::Vamana& index,
        const Queries& queries,
        const Groundtruth& groundtruth,
        size_t num_neighbors,
        double target_recall,
        svsbenchmark::CalibrateContext ctx,
        const calibration_constructor& f
    ) {
        // This method may only be called in the initial training set context.
        if (ctx != CalibrateContext::InitialTrainingSet) {
            throw ANNEXCEPTION("Default static calibration may only be performed on the "
                               "initial training set!");
        }

        auto c = f();
        return index.experimental_calibrate(
            queries, svs::recall_convert(groundtruth), num_neighbors, target_recall, c
        );
    }

    // Calibrate with hint.
    template <svs::data::ImmutableMemoryDataset Queries, typename Groundtruth>
    static config_type calibrate_with_hint(
        svs::Vamana& index,
        const Queries& queries,
        const Groundtruth& groundtruth,
        size_t num_neighbors,
        double target_recall,
        svsbenchmark::CalibrateContext ctx,
        const config_type& preset,
        const calibration_constructor& f
    ) {
        using SearchBufferOptimization =
            svs::index::vamana::CalibrationParameters::SearchBufferOptimization;
        // This method may only be called in the initial training set context.
        if (ctx != CalibrateContext::TestSetTune) {
            throw ANNEXCEPTION("Calibration tune-up for the static index may only be "
                               "called to obtain the desired accuracy on the test queries."
            );
        }

        auto c = f();
        c.train_prefetchers_ = false;
        c.search_buffer_optimization_ = SearchBufferOptimization::ROITuneUp;

        fmt::print("Tuning up recall");
        index.set_search_parameters(preset);
        return index.experimental_calibrate(
            queries, svs::recall_convert(groundtruth), num_neighbors, target_recall, c
        );
    }
};

} // namespace svsbenchmark
