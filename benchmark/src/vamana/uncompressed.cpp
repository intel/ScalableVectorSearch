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
#include "svs-benchmark/vamana/uncompressed.h"
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/datasets/uncompressed.h"
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/vamana/common.h"
#include "svs-benchmark/vamana/dynamic_traits.h"
#include "svs-benchmark/vamana/iterator.h"
#include "svs-benchmark/vamana/static_traits.h"

// svs
#include "svs/core/distance.h"
#include "svs/lib/dispatcher.h"
#include "svs/third-party/toml.h"

// stl
#include <string_view>

namespace svsbenchmark::vamana {

namespace {

// Specializations
#define X(Q, T, D, N) f.template operator()<Q, T, D, N>()
template <typename F> void for_standard_specializations(F&& f) {
    if constexpr (!is_minimal) {
        X(uint8_t, uint8_t, svs::distance::DistanceL2, 128);    // bigann
        X(float, svs::Float16, svs::distance::DistanceL2, 96);  // deep
        X(float, svs::Float16, svs::distance::DistanceL2, 100); // msturing
        X(int8_t, int8_t, svs::distance::DistanceL2, 100);      // spacev
        X(float, svs::Float16, svs::distance::DistanceIP, 200); // text2image
        X(float, svs::Float16, svs::distance::DistanceIP, 768); // dpr
        // Generic fallbacks
        // X(float, float, svs::distance::DistanceL2, svs::Dynamic);
        X(float, svs::Float16, svs::distance::DistanceL2, svs::Dynamic);
    }
}

template <typename F> void for_standard_search_specializations(F&& f) {
    if constexpr (svsbenchmark::vamana_supersearch) {
        X(uint8_t, uint8_t, svs::distance::DistanceL2, 128);          // bigann
        X(uint8_t, uint8_t, svs::distance::DistanceL2, svs::Dynamic); // bigann

        X(float, svs::Float16, svs::distance::DistanceL2, 96);      // deep
        X(float, svs::Float16, svs::distance::DistanceL2, 100);     // msturing
        X(int8_t, int8_t, svs::distance::DistanceL2, 100);          // spacev
        X(int8_t, int8_t, svs::distance::DistanceL2, svs::Dynamic); // spacev

        X(float, svs::Float16, svs::distance::DistanceIP, 200); // text2image
        X(float, svs::Float16, svs::distance::DistanceIP, 768); // dpr/rqa
        X(float, svs::Float16, svs::distance::DistanceIP, 512); // laion
        X(float, svs::Float16, svs::distance::DistanceL2, 512); // open-images

        X(float, svs::Float16, svs::distance::DistanceIP, svs::Dynamic);
        X(float, svs::Float16, svs::distance::DistanceL2, svs::Dynamic);
    } else {
        for_standard_specializations(SVS_FWD(f));
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
    auto index = svs::Vamana::assemble<Q>(
        job.config_,
        svs::GraphLoader{job.graph_},
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
        IndexTraits<svs::Vamana>::regression_optimization()
    );
    return svs::lib::save_to_table(results);
}

// Iterator
template <typename Q, typename T, typename D, size_t N>
toml::table run_iterator_search(
    // dispatch arguments
    svsbenchmark::TypedUncompressed<T> SVS_UNUSED(dataset_type),
    svsbenchmark::DispatchType<Q> SVS_UNUSED(query_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const svsbenchmark::Checkpoint& checkpointer,
    const svsbenchmark::vamana::IteratorSearch& job
) {
    auto index = svs::Vamana::assemble<Q>(
        job.config_,
        svs::GraphLoader{job.graph_},
        svs::data::SimpleData<T, N, svs::HugepageAllocator<T>>::load(job.data_),
        distance,
        1
    );

    auto queries = svs::data::SimpleData<Q>::load(job.queries_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(job.groundtruth_);

    auto query_set = vamana::subsample(
        queries.cview(), groundtruth.cview(), job.parameters_.query_subsample_
    );
    return vamana::tune_and_search_iterator(index, job, query_set, checkpointer);
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
    auto tic = svs::lib::now();
    auto index = svs::Vamana::build<Q>(
        job.build_parameters_,
        svs::data::SimpleData<T, N, svs::HugepageAllocator<T>>::load(job.data_),
        distance,
        job.num_threads_,
        svs::HugepageAllocator<uint32_t>()
    );
    double build_time = svs::lib::time_difference(tic);

    // Save the index if requested by the caller.
    job.maybe_save_index(index);

    // Load and run queries.
    auto queries = svs::data::SimpleData<Q>::load(job.queries_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(job.groundtruth_);
    auto results = svsbenchmark::search::run_search(
        index,
        job,
        search::QuerySet<Q, uint32_t>(queries, groundtruth, job.queries_in_training_set_),
        svsbenchmark::BuildTime{build_time},
        IndexTraits<svs::Vamana>::regression_optimization()
    );
    return svs::lib::save_to_table(results);
}

// Dynamic build and search
template <typename Q, typename T, typename D, size_t N>
toml::table run_dynamic_uncompressed(
    // dispatch arguments.
    TypedUncompressed<T> SVS_UNUSED(tag),
    DispatchType<Q> SVS_UNUSED(query_type),
    DispatchType<T> SVS_UNUSED(data_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const DynamicBuildJob& job,
    const Checkpoint& checkpointer
) {
    auto bundle = svsbenchmark::build::initialize_dynamic<T, Q>(
        job.data_,
        job.queries_,
        distance,
        job.get_dynamic_schedule(),
        job.num_threads_,
        [&](const auto& points, const auto& ids) {
            using A = svs::HugepageAllocator<T>;
            auto data_mutable =
                svs::data::BlockedData<T, N, A>(points.size(), points.dimensions());
            svs::data::copy(points, data_mutable);
            return svs::index::vamana::MutableVamanaIndex(
                job.build_parameters_,
                std::move(data_mutable),
                ids,
                distance,
                job.num_threads_
            );
        }
    );

    return svsbenchmark::build::dynamic_test_loop(
        bundle,
        job,
        [&](const toml::table& table) {
            checkpointer.checkpoint(table, benchmark_name(DynamicBenchmark()));
        },
        job.dynamic_optimization_
    );
}

template <typename Eltype, typename Distance>
svsbenchmark::TestFunctionReturn test_search(const VamanaTest& job) {
    // Get the groundtruth for the distance.
    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);
    auto kind = svsbenchmark::Uncompressed(svs::datatype_v<Eltype>);

    // Construct a `SearchJob` for this operation.
    auto search_job = SearchJob{
        "uncompressed reference search",
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
        test_search_configs()
    };

    // Load the components for the test.
    auto tic = svs::lib::now();
    auto data_loader = svs::lib::Lazy([&]() {
        return svsbenchmark::convert_data(
            svs::lib::Type<Eltype>(), svs::data::SimpleData<float>::load(job.data_f32_)
        );
    });
    auto index = svs::Vamana::assemble<float>(
        job.index_config_,
        svs::GraphLoader{job.graph_},
        data_loader,
        Distance(),
        job.num_threads_
    );
    double load_time = svs::lib::time_difference(tic);
    auto queries = svs::data::SimpleData<float>::load(job.queries_f32_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(groundtruth_path);

    auto results = svsbenchmark::search::run_search(
        index,
        search_job,
        svsbenchmark::search::QuerySet{
            std::move(queries), std::move(groundtruth), job.queries_in_training_set_
        },
        svsbenchmark::LoadTime{load_time},
        IndexTraits<svs::Vamana>::test_generation_optimization()
    );

    return TestFunctionReturn{
        .key_ = "vamana_test_search",
        .results_ =
            svs::lib::save_to_table(vamana::ExpectedResult(std::move(kind), results))
    };
}

template <typename Eltype, typename Distance>
svsbenchmark::TestFunctionReturn test_build(const VamanaTest& job) {
    // Get the groundtruth for the distance.
    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);

    auto build_parameters = svs::index::vamana::VamanaBuildParameters{
        pick_alpha(distance), 32, 100, 250, 28, true
    };

    auto kind = svsbenchmark::Uncompressed(svs::datatype_v<Eltype>);

    // Construct a `SearchJob` for this operation.
    auto build_job = BuildJob{
        groundtruth_path,
        svsbenchmark::vamana::search_parameters_from_window_sizes({1, 2, 3, 4, 5, 10}),
        test_search_parameters(),
        std::nullopt,
        "uncompressed reference build",
        kind,
        job.data_f32_,
        job.queries_f32_,
        job.queries_in_training_set_,
        svs::datatype_v<Eltype>,
        svs::DataType::float32,
        svs::distance_type_v<Distance>,
        Extent(svs::Dynamic),
        build_parameters,
        job.num_threads_
    };

    // Load the components for the test.
    auto tic = svs::lib::now();
    auto data_loader = svs::lib::Lazy([&]() {
        return svsbenchmark::convert_data(
            svs::lib::Type<Eltype>(), svs::data::SimpleData<float>::load(job.data_f32_)
        );
    });
    auto index = svs::Vamana::build<float>(
        build_parameters, data_loader, Distance(), job.num_threads_
    );
    double build_time = svs::lib::time_difference(tic);
    auto queries = svs::data::SimpleData<float>::load(job.queries_f32_);
    auto groundtruth = svs::data::SimpleData<uint32_t>::load(groundtruth_path);

    auto results = svsbenchmark::search::run_search(
        index,
        build_job,
        svsbenchmark::search::QuerySet{
            std::move(queries), std::move(groundtruth), job.queries_in_training_set_
        },
        svsbenchmark::BuildTime{build_time},
        IndexTraits<svs::Vamana>::test_generation_optimization()
    );

    return TestFunctionReturn{
        .key_ = "vamana_test_build",
        .results_ =
            svs::lib::save_to_table(vamana::ExpectedResult(std::move(kind), results))
    };
}

} // namespace

// target-registration.
void register_uncompressed_static_search(vamana::StaticSearchDispatcher& dispatcher) {
    for_standard_search_specializations(
        [&dispatcher]<typename Q, typename T, typename D, size_t N>() {
            auto method = &run_static_search<Q, T, D, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

void register_uncompressed_iterator_search(vamana::IteratorDispatcher& dispatcher) {
    for_standard_search_specializations(
        [&dispatcher]<typename Q, typename T, typename D, size_t N>() {
            auto method = &run_iterator_search<Q, T, D, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

void register_uncompressed_static_build(vamana::StaticBuildDispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, typename D, size_t N>() {
            auto method = &run_static_uncompressed<Q, T, D, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}
void register_uncompressed_dynamic_build(vamana::DynamicBuildDispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, typename D, size_t N>() {
            auto method = &run_dynamic_uncompressed<Q, T, D, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

std::vector<TestFunction> register_uncompressed_test_routines() {
    if constexpr (build_test_generators) {
        return std::vector<TestFunction>({
            // Searching
            &test_search<float, svs::DistanceL2>,
            &test_search<float, svs::DistanceIP>,
            &test_search<float, svs::DistanceCosineSimilarity>,
            // &test_search<uint8_t, svs::DistanceL2>,
            // &test_search<int8_t, svs::DistanceL2>,
            // &test_search<svs::Float16, svs::DistanceL2>,
            // Building
            &test_build<float, svs::DistanceL2>,
            &test_build<float, svs::DistanceIP>,
            &test_build<float, svs::DistanceCosineSimilarity>,
            &test_build<svs::Float16, svs::DistanceL2>,
            &test_build<svs::Float16, svs::DistanceIP>,
            &test_build<svs::Float16, svs::DistanceCosineSimilarity>,
        });
    } else {
        return std::vector<TestFunction>();
    }
}

} // namespace svsbenchmark::vamana
