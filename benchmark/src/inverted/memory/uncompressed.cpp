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

// svs-benchmark
#include "svs-benchmark/datasets/uncompressed.h"
#include "svs-benchmark/inverted/memory/traits.h"
#include "svs-benchmark/inverted/memory/uncompressed.h"

namespace svsbenchmark::inverted::memory {
namespace {

// Specializations
template <typename F> void for_standard_specializations(F&& f) {
#define X(Q, T, D, S, N) f.template operator()<Q, T, D, S, N>()
    using SparseStrategy = svs::index::inverted::SparseStrategy;
    using DenseStrategy = svs::index::inverted::DenseStrategy;

    if constexpr (!is_minimal) {
        // bigann
        X(uint8_t, uint8_t, svs::distance::DistanceL2, SparseStrategy, 128);
        X(uint8_t, uint8_t, svs::distance::DistanceL2, DenseStrategy, 128);

        X(int8_t, int8_t, svs::distance::DistanceL2, SparseStrategy, 100);
        X(int8_t, int8_t, svs::distance::DistanceL2, DenseStrategy, 100);

        X(float, svs::Float16, svs::distance::DistanceL2, SparseStrategy, 96); // deep
        X(float, svs::Float16, svs::distance::DistanceL2, DenseStrategy, 96);  // deep

        X(float, svs::Float16, svs::distance::DistanceL2, SparseStrategy, 100); // msturing
        X(float, svs::Float16, svs::distance::DistanceL2, DenseStrategy, 100);  // msturing

        X(float, svs::Float16, svs::distance::DistanceIP, SparseStrategy, 200);
        X(float, svs::Float16, svs::distance::DistanceIP, DenseStrategy, 200);

        X(float, svs::Float16, svs::distance::DistanceIP, SparseStrategy, 768);

        // X(float, svs::Float16, svs::distance::DistanceIP, SparseStrategy, svs::Dynamic);
    }
}

// Index Construction and Search
template <
    typename QueryType,
    typename SourceType,
    typename Distance,
    typename ClusterStrategy,
    size_t Extent>
toml::table build_uncompressed_memory(
    // Dispatch Arguments
    svsbenchmark::TypedUncompressed<SourceType> SVS_UNUSED(dispatch_type),
    DispatchType<QueryType> SVS_UNUSED(query_type),
    DispatchType<SourceType> SVS_UNUSED(data_type),
    Distance distance,
    ClusterStrategy clustering_strategy,
    svs::lib::ExtentTag<Extent> SVS_UNUSED(extent),
    // Feed-forward arguments
    const MemoryBuildJob& job
) {
    auto tic = svs::lib::now();

    // Custom build routine that can save partially completed results for down-stream
    // experimentation.
    auto index = svsbenchmark::inverted::memory::build<QueryType>(
        job,
        svs::VectorDataLoader<SourceType, Extent>(job.data_),
        distance,
        clustering_strategy
    );
    auto build_time = svs::lib::time_difference(tic);

    // Load queries and groundtruth.
    auto queries = svs::data::SimpleData<QueryType>::load(job.queries_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(job.groundtruth_);

    auto results = svsbenchmark::search::run_search(
        index,
        job,
        search::QuerySet<QueryType, uint32_t>(queries, groundtruth, queries.size() / 2),
        svsbenchmark::BuildTime{build_time},
        svsbenchmark::Placeholder{}
    );

    return svs::lib::save_to_table(results);
}

// Index Search
template <
    typename QueryType,
    typename SourceType,
    typename Distance,
    typename ClusterStrategy,
    size_t Extent>
toml::table search_uncompressed_memory(
    // Dispatch Arguments
    svsbenchmark::TypedUncompressed<SourceType> SVS_UNUSED(dispatch_type),
    DispatchType<QueryType> SVS_UNUSED(query_type),
    DispatchType<SourceType> SVS_UNUSED(data_type),
    Distance distance,
    ClusterStrategy clustering_strategy,
    svs::lib::ExtentTag<Extent> SVS_UNUSED(extent),
    // Feed-forward arguments
    const MemorySearchJob& job
) {
    auto tic = svs::lib::now();
    const auto& assembly = job.assembly_;
    auto index = svs::Inverted::assemble_from_clustering<QueryType>(
        assembly.clustering_,
        svs::VectorDataLoader<SourceType, Extent>(job.original_data_),
        distance,
        assembly.primary_index_config_,
        assembly.primary_index_graph_,
        job.num_threads_,
        clustering_strategy
    );
    auto load_time = svs::lib::time_difference(tic);

    // Load queries and groundtruth.
    auto queries = svs::data::SimpleData<QueryType>::load(job.queries_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(job.groundtruth_);

    auto results = svsbenchmark::search::run_search(
        index,
        job,
        search::QuerySet<QueryType, uint32_t>(queries, groundtruth, queries.size() / 2),
        svsbenchmark::LoadTime{load_time},
        svsbenchmark::Placeholder{}
    );

    return svs::lib::save_to_table(results);
}

///// Test Generation
SVS_BENCHMARK_FOR_TESTS_ONLY svs::index::vamana::VamanaBuildParameters
select_build_parameters(svs::DistanceType distance) {
    float alpha = (distance == svs::DistanceType::L2) ? 1.2F : 0.95F;
    return svs::index::vamana::VamanaBuildParameters{alpha, 64, 200, 1000, 60, true};
}

// Test Routines
SVS_BENCHMARK_FOR_TESTS_ONLY search::SearchParameters test_search_parameters() {
    return search::SearchParameters{10, {0.2, 0.5, 0.8, 0.9}};
}

// Result generation for Unit Tests.
template <typename Eltype, typename Distance>
svsbenchmark::TestFunctionReturn test_build(const InvertedTest& job) {
    using ClusteringParameters = svs::index::inverted::ClusteringParameters;
    auto num_threads = job.num_threads_;

    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);

    auto clustering_parameters = ClusteringParameters()
                                     .percent_centroids(svs::lib::Percent(0.12))
                                     .epsilon(1.0)
                                     .max_replicas(7)
                                     .max_cluster_size(60)
                                     .refinement_alpha(1.0)
                                     .num_intermediate_results(64);

    auto kind = svsbenchmark::Uncompressed(svs::datatype_v<Eltype>);

    auto build_job = memory::MemoryBuildJob(
        "inverted reference build",
        kind,
        job.data_f32_,
        job.queries_f32_,
        groundtruth_path,
        svs::datatype_v<Eltype>,
        svs::DataType::float32,
        distance,
        Extent(svs::Dynamic),
        select_build_parameters(distance),
        clustering_parameters,
        {},
        num_threads,
        memory::ClusterStrategy::Sparse,
        std::nullopt,
        // search-setup
        {},
        test_search_parameters()
    );

    // Load the data.
    auto tic = svs::lib::now();
    auto data_loader = svs::lib::Lazy([&]() {
        return svsbenchmark::convert_data(
            svs::lib::Type<Eltype>(),
            svs::data::SimpleData<float, svs::Dynamic, svs::HugepageAllocator<float>>::load(
                job.data_f32_
            )
        );
    });

    // Custom build routine that can save partially completed results for down-stream
    // experimentation.
    auto index = svsbenchmark::inverted::memory::build<float>(
        build_job, data_loader, Distance(), svs::index::inverted::SparseStrategy()
    );
    auto build_time = svs::lib::time_difference(tic);

    // Load queries and groundtruth.
    auto queries = svs::data::SimpleData<float>::load(job.queries_f32_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(groundtruth_path);

    auto results = svsbenchmark::search::run_search(
        index,
        build_job,
        search::QuerySet{
            std::move(queries), std::move(groundtruth), job.queries_in_training_set_},
        svsbenchmark::BuildTime{build_time},
        svsbenchmark::Placeholder{}
    );

    return svsbenchmark::TestFunctionReturn{
        .key_ = "inverted_test_build",
        .results_ =
            svs::lib::save_to_table(memory::ExpectedResult(std::move(kind), results))};
}

} // namespace

// Header-file hooks.
void register_uncompressed_memory_build(inverted::memory::MemoryBuildDispatcher& dispatcher
) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, typename D, typename S, size_t N>() {
            auto method = &build_uncompressed_memory<Q, T, D, S, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

void register_uncompressed_memory_search(memory::MemorySearchDispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, typename D, typename S, size_t N>() {
            auto method = &search_uncompressed_memory<Q, T, D, S, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

std::vector<TestFunction> register_uncompressed_test_routines() {
    if constexpr (build_test_generators) {
        return std::vector<TestFunction>({
            &test_build<float, svs::DistanceL2>,
            &test_build<float, svs::DistanceIP>,
        });
    } else {
        return std::vector<TestFunction>();
    }
}

} // namespace svsbenchmark::inverted::memory
