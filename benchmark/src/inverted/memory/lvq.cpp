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

// svs-benchmark
#include "svs-benchmark/datasets/lvq.h"
#include "svs-benchmark/inverted/memory/build.h"
#include "svs-benchmark/inverted/memory/lvq.h"
#include "svs-benchmark/inverted/memory/traits.h"

// svs
#include "svs/extensions/inverted/lvq.h"

namespace svsbenchmark::inverted::memory {
namespace {

using svs::quantization::lvq::Sequential;

// Specializations
template <typename F> void for_lvq_specializations(F&& f) {
#define X(P, R, Q, T, D, S, C, N) f.template operator()<P, R, Q, T, D, S, C, N>();
    using SparseStrategy = svs::index::inverted::SparseStrategy;
    using Distance = svs::distance::DistanceIP;
    const size_t Dims = 768;

    if constexpr (!is_minimal) {
        X(4, 0, float, svs::Float16, Distance, Sequential, SparseStrategy, Dims);
        X(7, 0, float, svs::Float16, Distance, Sequential, SparseStrategy, Dims);
        X(8, 0, float, svs::Float16, Distance, Sequential, SparseStrategy, Dims);
    }
#undef X
}

// Index Construction and Search
template <
    size_t Primary,
    size_t Residual,
    typename QueryType,
    typename SourceType,
    typename Distance,
    typename LVQStrategy,
    typename ClusterStrategy,
    size_t Extent>
toml::table build_lvq_memory(
    // Dispatch Arguments
    svsbenchmark::TypedLVQ<Primary, Residual, LVQStrategy> SVS_UNUSED(dispatch_type),
    DispatchType<QueryType> SVS_UNUSED(query_type),
    DispatchType<SourceType> SVS_UNUSED(data_type),
    Distance distance,
    ClusterStrategy clustering_strategy,
    svs::lib::ExtentTag<Extent> SVS_UNUSED(extent),
    // Feed-forward arguments
    const MemoryBuildJob& job
) {
    namespace lvq = svs::quantization::lvq;
    using A = svs::HugepageAllocator<std::byte>;
    using LVQType = lvq::LVQDataset<Primary, Residual, Extent, LVQStrategy, A>;

    auto tic = svs::lib::now();

    // Custom build routine that can save partially completed results for down-stream
    // experimentation.
    auto index = svsbenchmark::inverted::memory::build<QueryType>(
        job,
        svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
            auto data = svs::data::SimpleData<SourceType>::load(job.data_);
            return LVQType::compress(data, threadpool, 32);
        }),
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

template <
    size_t Primary,
    size_t Residual,
    typename QueryType,
    typename SourceType,
    typename Distance,
    typename LVQStrategy,
    typename ClusterStrategy,
    size_t Extent>
toml::table search_lvq_memory(
    // Dispatch Arguments
    svsbenchmark::TypedLVQ<Primary, Residual, LVQStrategy> SVS_UNUSED(dispatch_type),
    DispatchType<QueryType> SVS_UNUSED(query_type),
    DispatchType<SourceType> SVS_UNUSED(data_type),
    Distance distance,
    ClusterStrategy clustering_strategy,
    svs::lib::ExtentTag<Extent> SVS_UNUSED(extent),
    // Feed-forward arguments
    const MemorySearchJob& job
) {
    namespace lvq = svs::quantization::lvq;
    using A = svs::HugepageAllocator<std::byte>;
    using LVQType = lvq::LVQDataset<Primary, Residual, Extent, LVQStrategy, A>;

    auto tic = svs::lib::now();
    const auto& assembly = job.assembly_;
    auto index = svs::Inverted::assemble_from_clustering<QueryType>(
        assembly.clustering_,
        svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
            auto data = svs::data::SimpleData<SourceType>::load(job.original_data_);
            return LVQType::compress(data, threadpool, 32);
        }),
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
    return search::SearchParameters{10, {0.2, 0.5, 0.8}};
}

// Result generation for Unit Tests.
template <size_t Primary, typename Distance>
svsbenchmark::TestFunctionReturn test_build(const InvertedTest& job) {
    namespace lvq = svs::quantization::lvq;
    auto num_threads = job.num_threads_;

    using ClusteringParameters = svs::index::inverted::ClusteringParameters;

    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);

    auto clustering_parameters = ClusteringParameters()
                                     .percent_centroids(svs::lib::Percent(0.12))
                                     .epsilon(1.0)
                                     .max_replicas(7)
                                     .max_cluster_size(60)
                                     .refinement_alpha(1.0)
                                     .num_intermediate_results(64);

    auto kind = svsbenchmark::LVQ(Primary, 0, svsbenchmark::LVQPackingStrategy::Sequential);

    auto build_job = memory::MemoryBuildJob(
        "inverted lvq reference build",
        kind,
        job.data_f32_,
        job.queries_f32_,
        groundtruth_path,
        svs::datatype_v<float>,
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
    auto data_loader = svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::data::SimpleData<float>::load(job.data_f32_);
        return lvq::LVQDataset<Primary, 0>::compress(data, threadpool, 0);
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
void register_lvq_memory_build(inverted::memory::MemoryBuildDispatcher& dispatcher) {
    for_lvq_specializations([&dispatcher]<
                                size_t P,
                                size_t R,
                                typename Q,
                                typename T,
                                typename D,
                                typename S,
                                typename C,
                                size_t N>() {
        auto method = &build_lvq_memory<P, R, Q, T, D, S, C, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

void register_lvq_memory_search(inverted::memory::MemorySearchDispatcher& dispatcher) {
    for_lvq_specializations([&dispatcher]<
                                size_t P,
                                size_t R,
                                typename Q,
                                typename T,
                                typename D,
                                typename S,
                                typename C,
                                size_t N>() {
        auto method = &search_lvq_memory<P, R, Q, T, D, S, C, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

std::vector<TestFunction> register_lvq_test_routines() {
    if constexpr (build_test_generators) {
        return std::vector<TestFunction>({
            // Build L2
            &test_build<8, svs::DistanceL2>,
            &test_build<4, svs::DistanceL2>,
            // Build IP
            &test_build<8, svs::DistanceIP>,
            &test_build<4, svs::DistanceIP>,
        });
    } else {
        return std::vector<TestFunction>();
    }
}

} // namespace svsbenchmark::inverted::memory
