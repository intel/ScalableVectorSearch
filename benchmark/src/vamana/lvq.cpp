// svs-benchmark
#include "svs-benchmark/datasets/lvq.h"
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/vamana/common.h"
#include "svs-benchmark/vamana/dynamic_traits.h"
#include "svs-benchmark/vamana/lvq.h"
#include "svs-benchmark/vamana/static_traits.h"

// svs
#include "svs/extensions/vamana/lvq.h"

namespace svsbenchmark::vamana {

namespace {
using svs::quantization::lvq::Sequential;
using svs::quantization::lvq::Turbo;

// Specializations
template <typename F> void lvq_specializations(F&& f) {
#define X(P, R, Q, T, D, S, N) f.template operator()<P, R, Q, T, D, S, N>()
    if constexpr (!is_minimal) {
        using SrcType = svs::Float16;
        using Distance = svs::distance::DistanceIP;
        const size_t Dim = 768;

        // Sequential
        X(8, 0, float, SrcType, Distance, Sequential, Dim);
        X(4, 8, float, SrcType, Distance, Sequential, Dim);
        X(8, 8, float, SrcType, Distance, Sequential, Dim);

        // Turbo
        using Turbo16x8 = Turbo<16, 8>;
        using Turbo16x4 = Turbo<16, 4>;
        X(4, 8, float, SrcType, Distance, Turbo16x8, Dim);
        X(8, 0, float, SrcType, Distance, Turbo16x4, Dim);
    }
#undef X
}

// Load and Search
template <
    size_t Primary,
    size_t Residual,
    typename Q,
    typename T,
    typename D,
    typename S,
    size_t N>
toml::table run_static_search(
    // dispatch arguments
    svsbenchmark::TypedLVQ<Primary, Residual, S> SVS_UNUSED(dispatch_type),
    DispatchType<Q> SVS_UNUSED(query_type),
    DispatchType<T> SVS_UNUSED(data_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const SearchJob& job
) {
    namespace lvq = svs::quantization::lvq;
    using A = svs::HugepageAllocator<std::byte>;
    using LVQType = lvq::LVQDataset<Primary, Residual, N, S, A>;

    auto tic = svs::lib::now();
    auto lazy = svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::data::SimpleData<T, N>::load(job.data_);
        return LVQType::compress(data, threadpool, 32);
    });

    auto index = svs::Vamana::assemble<Q>(
        job.config_, svs::GraphLoader{job.graph_}, lazy, distance, job.num_threads_
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

// Static build and search
template <
    size_t Primary,
    size_t Residual,
    typename Q,
    typename T,
    typename D,
    typename S,
    size_t N>
toml::table run_static_lvq(
    // dispatch arguments
    svsbenchmark::TypedLVQ<Primary, Residual, S> SVS_UNUSED(dispatch_type),
    DispatchType<Q> SVS_UNUSED(query_type),
    DispatchType<T> SVS_UNUSED(data_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const BuildJob& job
) {
    namespace lvq = svs::quantization::lvq;
    using A = svs::HugepageAllocator<std::byte>;
    using LVQType = lvq::LVQDataset<Primary, Residual, N, S, A>;

    auto tic = svs::lib::now();
    auto lazy = svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::data::SimpleData<T, N>::load(job.data_);
        return LVQType::compress(data, threadpool, 32);
    });

    auto index =
        svs::Vamana::build<Q>(job.build_parameters_, lazy, distance, job.num_threads_);
    double build_time = svs::lib::time_difference(tic);
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
template <
    size_t Primary,
    size_t Residual,
    typename Q,
    typename T,
    typename D,
    typename S,
    size_t N>
toml::table run_dynamic_lvq(
    // dispatch arguments.
    svsbenchmark::TypedLVQ<Primary, Residual, S> SVS_UNUSED(dispatch_type),
    DispatchType<Q> SVS_UNUSED(query_type),
    DispatchType<T> SVS_UNUSED(data_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const DynamicBuildJob& job,
    const Checkpoint& checkpointer
) {
    namespace lvq = svs::quantization::lvq;
    using A = svs::data::Blocked<svs::HugepageAllocator<std::byte>>;
    using LVQType = lvq::LVQDataset<Primary, Residual, N, S, A>;

    auto bundle = svsbenchmark::build::initialize_dynamic<T, Q>(
        job.data_,
        job.queries_,
        distance,
        job.get_dynamic_schedule(),
        job.num_threads_,
        [&](const auto& points, const auto& ids) {
            return svs::index::vamana::MutableVamanaIndex(
                job.build_parameters_,
                LVQType::compress(points, job.num_threads_, 32),
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

template <size_t Primary, size_t Residual, typename Distance>
svsbenchmark::TestFunctionReturn test_search(const VamanaTest& job) {
    namespace lvq = svs::quantization::lvq;

    // Get the groundtruth for the distance.
    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);
    auto kind = svsbenchmark::LVQ(Primary, Residual, LVQPackingStrategy::Sequential);

    // Construct a `SearchJob` for this operation.
    auto search_job = SearchJob{
        "lvq reference search",
        kind,
        job.index_config_,
        job.graph_,
        job.data_f32_,
        job.queries_f32_,
        groundtruth_path,
        job.queries_in_training_set_,
        svs::DataType::float32,
        svs::DataType::float32,
        svs::distance_type_v<Distance>,
        Extent(svs::Dynamic),
        job.num_threads_,
        test_search_parameters(),
        test_search_configs()};

    // Load the components for the test.
    auto tic = svs::lib::now();
    auto data_loader = svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::data::SimpleData<float>::load(job.data_f32_);
        return lvq::LVQDataset<Primary, Residual>::compress(data, threadpool, 0);
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
            std::move(queries), std::move(groundtruth), job.queries_in_training_set_},
        svsbenchmark::LoadTime{load_time},
        IndexTraits<svs::Vamana>::test_generation_optimization()
    );

    return svsbenchmark::TestFunctionReturn{
        .key_ = "vamana_test_search",
        .results_ =
            svs::lib::save_to_table(vamana::ExpectedResult(std::move(kind), results))};
}

template <size_t Primary, size_t Residual, typename Distance>
TestFunctionReturn test_build(const VamanaTest& job) {
    namespace lvq = svs::quantization::lvq;

    // Get the groundtruth for the distance.
    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);

    auto build_parameters = svs::index::vamana::VamanaBuildParameters{
        pick_alpha(distance), 32, 100, 250, 28, true};

    auto kind =
        svsbenchmark::LVQ(Primary, Residual, svsbenchmark::LVQPackingStrategy::Sequential);

    // Construct a `SearchJob` for this operation.
    auto build_job = BuildJob{
        groundtruth_path,
        {1, 2, 3, 4, 5, 10},
        test_search_parameters(),
        "lvq reference build",
        kind,
        job.data_f32_,
        job.queries_f32_,
        job.queries_in_training_set_,
        svs::DataType::float32,
        svs::DataType::float32,
        svs::distance_type_v<Distance>,
        Extent(svs::Dynamic),
        build_parameters,
        job.num_threads_};

    // Load the components for the test.
    auto tic = svs::lib::now();
    auto data_loader = svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::data::SimpleData<float>::load(job.data_f32_);
        return lvq::LVQDataset<Primary, Residual>::compress(data, threadpool, 0);
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
            std::move(queries), std::move(groundtruth), job.queries_in_training_set_},
        svsbenchmark::BuildTime{build_time},
        IndexTraits<svs::Vamana>::test_generation_optimization()
    );

    return TestFunctionReturn{
        .key_ = "vamana_test_build",
        .results_ = svs::lib::save_to_table(vamana::ExpectedResult(kind, results))};
}

} // namespace

// target-registration.
void register_lvq_static_search(vamana::StaticSearchDispatcher& dispatcher) {
    lvq_specializations([&dispatcher]<
                            size_t P,
                            size_t R,
                            typename Q,
                            typename T,
                            typename D,
                            typename S,
                            size_t N>() {
        auto method = &run_static_search<P, R, Q, T, D, S, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

void register_lvq_static_build(vamana::StaticBuildDispatcher& dispatcher) {
    lvq_specializations([&dispatcher]<
                            size_t P,
                            size_t R,
                            typename Q,
                            typename T,
                            typename D,
                            typename S,
                            size_t N>() {
        auto method = &run_static_lvq<P, R, Q, T, D, S, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

void register_lvq_dynamic_build(vamana::DynamicBuildDispatcher& dispatcher) {
    lvq_specializations([&dispatcher]<
                            size_t P,
                            size_t R,
                            typename Q,
                            typename T,
                            typename D,
                            typename S,
                            size_t N>() {
        auto method = &run_dynamic_lvq<P, R, Q, T, D, S, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

std::vector<TestFunction> register_lvq_test_routines() {
    if constexpr (build_test_generators) {
        return std::vector<TestFunction>({
            // Searching
            &test_search<8, 0, svs::DistanceL2>,
            &test_search<4, 0, svs::DistanceL2>,
            &test_search<4, 4, svs::DistanceL2>,
            &test_search<4, 8, svs::DistanceL2>,
            &test_search<8, 8, svs::DistanceL2>,
            // Building
            &test_build<8, 0, svs::DistanceL2>,
            &test_build<8, 0, svs::DistanceIP>,
            &test_build<4, 8, svs::DistanceL2>,
            &test_build<4, 8, svs::DistanceIP>,
        });
    } else {
        return std::vector<TestFunction>();
    }
}

} // namespace svsbenchmark::vamana
