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

// svs-benchmark
#include "svs-benchmark/ivf/uncompressed.h"
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/datasets/uncompressed.h"
#include "svs-benchmark/ivf/build.h"
#include "svs-benchmark/ivf/common.h"
#include "svs-benchmark/ivf/static_traits.h"

// svs
#include "svs/core/distance.h"
#include "svs/lib/dispatcher.h"
#include "svs/third-party/toml.h"

// stl
#include <string_view>

namespace svsbenchmark::ivf {

namespace {

// Specializations
#define X(Q, T, D, N) f.template operator()<Q, T, D, N>()
template <typename F> void for_standard_specializations(F&& f) {
    if constexpr (!is_minimal) {
        X(float, svs::Float16, svs::distance::DistanceL2, 96);  // deep
        X(float, svs::Float16, svs::distance::DistanceL2, 100); // msturing
        X(float, svs::Float16, svs::distance::DistanceIP, 200); // text2image
        X(float, svs::Float16, svs::distance::DistanceIP, 512); // open-images, laion
        X(float, svs::Float16, svs::distance::DistanceIP, 768); // dpr, rqa
        // // Generic fallbacks
        // X(float, svs::Float16, svs::distance::DistanceL2, svs::Dynamic);
        // X(float, svs::Float16, svs::distance::DistanceIP, svs::Dynamic);
    }
}
#undef X

// Load and Search
template <typename Q, typename T, typename D, size_t N>
toml::table run_static_search(
    // dispatch arguments.
    svsbenchmark::TypedUncompressed<T> SVS_UNUSED(tag),
    DispatchType<Q> SVS_UNUSED(query_type),
    DispatchType<T> SVS_UNUSED(data_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const SearchJob& job
) {
    auto tic = svs::lib::now();
    auto index = svs::IVF::assemble_from_file<Q, svs::BFloat16>(
        job.config_,
        svs::data::SimpleData<T, N, svs::HugepageAllocator<T>>::load(job.data_),
        distance,
        job.num_threads_
    );

    double load_time = svs::lib::time_difference(tic);
    auto queries = svs::data::SimpleData<Q>::load(job.queries_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(job.groundtruth_);

    auto results = svsbenchmark::search::run_search(
        index,
        job,
        search::QuerySet<Q, uint32_t>(queries, groundtruth, job.queries_in_training_set_),
        svsbenchmark::LoadTime{load_time},
        svsbenchmark::Placeholder{}
    );
    return svs::lib::save_to_table(results);
}

// Static build and search
template <typename Q, typename T, typename D, size_t N>
toml::table run_static_uncompressed(
    // dispatch arguments.
    TypedUncompressed<T> SVS_UNUSED(tag),
    DispatchType<Q> SVS_UNUSED(query_type),
    DispatchType<T> SVS_UNUSED(data_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const BuildJob& job
) {
    auto data = svs::data::SimpleData<T, N>::load(job.data_);
    auto tic = svs::lib::now();
    auto clustering = svs::IVF::build_clustering<Q>(
        job.build_parameters_, data, distance, job.num_threads_
    );
    double build_time = svs::lib::time_difference(tic);
    auto index =
        svs::IVF::assemble_from_clustering<Q>(clustering, data, distance, job.num_threads_);

    // Save the index if requested by the caller.
    job.maybe_save_index(clustering);

    // Load and run queries.
    auto queries = svs::data::SimpleData<Q>::load(job.queries_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(job.groundtruth_);
    auto results = svsbenchmark::search::run_search(
        index,
        job,
        search::QuerySet<Q, uint32_t>(queries, groundtruth, job.queries_in_training_set_),
        svsbenchmark::BuildTime{build_time},
        svsbenchmark::Placeholder{}
    );
    return svs::lib::save_to_table(results);
}

template <typename Eltype, typename Distance>
svsbenchmark::TestFunctionReturn test_search(const IVFTest& job) {
    // Get the groundtruth for the distance.
    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);
    auto kind = svsbenchmark::Uncompressed(svs::datatype_v<Eltype>);

    // Construct a `SearchJob` for this operation.
    auto search_job = SearchJob{
        "IVF uncompressed reference search",
        kind,
        job.index_config_,
        job.graph_,
        job.data_f32_,
        job.queries_f32_,
        groundtruth_path,
        job.queries_in_training_set_,
        svs::datatype_v<Eltype>,
        svs::DataType::float32,
        svs::distance_type_v<Distance>,
        Extent(svs::Dynamic),
        job.num_threads_,
        test_search_parameters(),
        test_search_configs()};

    // Load the components for the test.
    auto tic = svs::lib::now();
    auto data_loader = svs::lib::Lazy([&]() {
        return svsbenchmark::convert_data(
            svs::lib::Type<Eltype>(), svs::data::SimpleData<float>::load(job.data_f32_)
        );
    });
    auto index = svs::IVF::assemble_from_file<float, svs::BFloat16>(
        job.index_config_, data_loader, Distance(), job.num_threads_
    );
    double load_time = svs::lib::time_difference(tic);
    auto queries = svs::data::SimpleData<float>::load(job.queries_f32_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(groundtruth_path);

    auto results = svsbenchmark::search::run_search(
        index,
        search_job,
        svsbenchmark::search::QuerySet{
            std::move(queries), std::move(groundtruth), job.queries_in_training_set_},
        svsbenchmark::LoadTime{load_time},
        svsbenchmark::Placeholder{}
    );

    return TestFunctionReturn{
        .key_ = "ivf_test_search",
        .results_ = svs::lib::save_to_table(ivf::ExpectedResult(std::move(kind), results))};
}

template <typename Eltype, typename Distance, bool Hierarchical = true>
svsbenchmark::TestFunctionReturn test_build(const IVFTest& job) {
    // Get the groundtruth for the distance.
    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);

    auto build_parameters =
        svs::index::ivf::IVFBuildParameters{128, 10000, 10, Hierarchical, 0.1};

    auto kind = svsbenchmark::Uncompressed(svs::datatype_v<Eltype>);

    // Construct a `SearchJob` for this operation.
    auto build_job = BuildJob{
        groundtruth_path,
        {{10, 1.0}, {50, 1.0}},
        test_search_parameters(),
        std::nullopt,
        "IVF uncompressed reference build",
        kind,
        job.data_f32_,
        job.queries_f32_,
        job.queries_in_training_set_,
        svs::datatype_v<Eltype>,
        svs::DataType::float32,
        svs::distance_type_v<Distance>,
        Extent(svs::Dynamic),
        build_parameters,
        job.num_threads_};

    // Load the components for the test.
    auto data = svsbenchmark::convert_data(
        svs::lib::Type<Eltype>(), svs::data::SimpleData<float>::load(job.data_f32_)
    );
    auto tic = svs::lib::now();
    auto clustering = svs::IVF::build_clustering<Eltype>(
        build_parameters, data, Distance(), job.num_threads_
    );
    double build_time = svs::lib::time_difference(tic);
    auto index = svs::IVF::assemble_from_clustering<float>(
        clustering, data, distance, job.num_threads_
    );

    auto queries = svs::data::SimpleData<float>::load(job.queries_f32_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(groundtruth_path);

    auto results = svsbenchmark::search::run_search(
        index,
        build_job,
        svsbenchmark::search::QuerySet{
            std::move(queries), std::move(groundtruth), job.queries_in_training_set_},
        svsbenchmark::BuildTime{build_time},
        svsbenchmark::Placeholder{}
    );

    return TestFunctionReturn{
        .key_ = "ivf_test_build",
        .results_ = svs::lib::save_to_table(ivf::ExpectedResult(std::move(kind), results))};
}

} // namespace

// target-registration.
void register_uncompressed_static_search(ivf::StaticSearchDispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, typename D, size_t N>() {
            auto method = &run_static_search<Q, T, D, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

void register_uncompressed_static_build(ivf::StaticBuildDispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, typename D, size_t N>() {
            auto method = &run_static_uncompressed<Q, T, D, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

std::vector<TestFunction> register_uncompressed_test_routines() {
    if constexpr (build_test_generators) {
        return std::vector<TestFunction>({
            // Searching
            &test_search<float, svs::DistanceL2>,
            &test_search<svs::Float16, svs::DistanceIP>,
            // Building
            &test_build<float, svs::DistanceL2>,
            &test_build<svs::Float16, svs::DistanceIP>,
            &test_build<svs::BFloat16, svs::DistanceL2>,
            &test_build<svs::BFloat16, svs::DistanceIP, false>,
        });
    } else {
        return std::vector<TestFunction>();
    }
}

} // namespace svsbenchmark::ivf
