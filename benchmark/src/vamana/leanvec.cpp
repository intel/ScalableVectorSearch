// svs-benchmark
#include "svs-benchmark/datasets/leanvec.h"
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/vamana/common.h"
#include "svs-benchmark/vamana/leanvec.h"
#include "svs-benchmark/vamana/static_traits.h"

// svs
#include "svs/extensions/vamana/leanvec.h"

namespace svsbenchmark::vamana {

namespace {
namespace leanvec = svs::leanvec;

// Specializations
template <typename F> void leanvec_specializations(F&& f) {
#define X(P, S, Q, T, D, L, N) f.template operator()<P, S, Q, T, D, L, N>()
    if constexpr (!is_minimal) {
        using SrcType = svs::Float16;
        using Distance = svs::distance::DistanceIP;
        const size_t Dim = 768;
        const size_t LeanVecDim = 160;

        X(SrcType, SrcType, float, SrcType, Distance, LeanVecDim, Dim);
        X(leanvec::UsingLVQ<8>, SrcType, float, SrcType, Distance, LeanVecDim, Dim);
        X(leanvec::UsingLVQ<8>,
          leanvec::UsingLVQ<8>,
          float,
          SrcType,
          Distance,
          LeanVecDim,
          Dim);
        X(SrcType, leanvec::UsingLVQ<8>, float, SrcType, Distance, LeanVecDim, Dim);
    }
#undef X
}

// Load and Search
template <
    typename Primary,
    typename Secondary,
    typename Q,
    typename T,
    typename D,
    size_t L,
    size_t N>
toml::table run_static_search(
    // dispatch arguments
    svsbenchmark::TypedLeanVec<Primary, Secondary, L> dataset,
    DispatchType<Q> SVS_UNUSED(query_type),
    DispatchType<T> SVS_UNUSED(data_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const SearchJob& job
) {
    using A = svs::HugepageAllocator<std::byte>;
    using LeanVecType = leanvec::LeanDataset<Primary, Secondary, L, N, A>;

    auto tic = svs::lib::now();
    auto lazy = svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::data::SimpleData<T, N, svs::HugepageAllocator<T>>::load(job.data_);
        return LeanVecType::reduce(data, dataset.transformation_, threadpool, 32);
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
    typename Primary,
    typename Secondary,
    typename Q,
    typename T,
    typename D,
    size_t L,
    size_t N>
toml::table run_static_leanvec(
    // dispatch arguments
    svsbenchmark::TypedLeanVec<Primary, Secondary, L> dataset,
    DispatchType<Q> SVS_UNUSED(query_type),
    DispatchType<T> SVS_UNUSED(data_type),
    D distance,
    svs::lib::ExtentTag<N> SVS_UNUSED(extent),
    // feed-forward arguments
    const BuildJob& job
) {
    using A = svs::HugepageAllocator<std::byte>;
    using LeanVecType = leanvec::LeanDataset<Primary, Secondary, L, N, A>;

    auto tic = svs::lib::now();
    auto lazy = svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::data::SimpleData<T, N, svs::HugepageAllocator<T>>::load(job.data_);
        return LeanVecType::reduce(data, dataset.transformation_, threadpool, 32);
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

template <
    typename Primary,
    typename Secondary,
    size_t LeanVecDims,
    typename Distance,
    bool IsPCA = true>
TestFunctionReturn test_search(const VamanaTest& job) {
    // Get the groundtruth for the distance.
    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);

    auto kind = svsbenchmark::LeanVec{
        svsbenchmark::detail::LeanVecKindMap<Primary>::kind,
        svsbenchmark::detail::LeanVecKindMap<Secondary>::kind,
        LeanVecDims,
        IsPCA ? std::nullopt : std::optional(job.leanvec_data_matrix_),
        IsPCA ? std::nullopt : std::optional(job.leanvec_query_matrix_)};

    using LeanVecType = leanvec::LeanDataset<Primary, Secondary, LeanVecDims, svs::Dynamic>;
    using MatrixType = svs::data::SimpleData<float, LeanVecDims>;

    // Construct a `SearchJob` for this operation.
    auto search_job = SearchJob{
        "leanvec reference search",
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
    auto data_loader = svs::lib::Lazy([&]() {
        auto data = svs::data::SimpleData<float>::load(job.data_f32_);
        auto leanvec_matrices = std::optional<leanvec::LeanVecMatrices<LeanVecDims>>();
        if (kind.data_matrix_.has_value()) {
            leanvec_matrices.emplace(
                MatrixType::load(kind.data_matrix_.value()),
                MatrixType::load(kind.query_matrix_.value())
            );
        }
        return LeanVecType::reduce(data, leanvec_matrices);
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

    return TestFunctionReturn{
        .key_ = "vamana_test_search",
        .results_ =
            svs::lib::save_to_table(vamana::ExpectedResult(std::move(kind), results))};
}

template <
    typename Primary,
    typename Secondary,
    size_t LeanVecDims,
    typename Distance,
    bool IsPCA = true>
TestFunctionReturn test_build(const VamanaTest& job) {
    // Get the groundtruth for the distance.
    constexpr svs::DistanceType distance = svs::distance_type_v<Distance>;
    const auto& groundtruth_path = job.groundtruth_for(distance);

    auto build_parameters = svs::index::vamana::VamanaBuildParameters{
        pick_alpha(distance), 32, 100, 250, 28, true};

    auto kind = svsbenchmark::LeanVec{
        svsbenchmark::detail::LeanVecKindMap<Primary>::kind,
        svsbenchmark::detail::LeanVecKindMap<Secondary>::kind,
        LeanVecDims,
        IsPCA ? std::nullopt : std::optional(job.leanvec_data_matrix_),
        IsPCA ? std::nullopt : std::optional(job.leanvec_query_matrix_)};

    using LeanVecType = leanvec::LeanDataset<Primary, Secondary, LeanVecDims, svs::Dynamic>;
    using MatrixType = svs::data::SimpleData<float, LeanVecDims>;

    // Construct a `SearchJob` for this operation.
    auto build_job = BuildJob{
        groundtruth_path,
        svsbenchmark::vamana::search_parameters_from_window_sizes({1, 2, 3, 4, 5, 10}),
        test_search_parameters(),
        "leanvec reference build",
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
    auto data_loader = svs::lib::Lazy([&]() {
        auto data = svs::data::SimpleData<float>::load(job.data_f32_);
        auto leanvec_matrices = std::optional<leanvec::LeanVecMatrices<LeanVecDims>>();
        if (kind.data_matrix_.has_value()) {
            leanvec_matrices.emplace(
                MatrixType::load(kind.data_matrix_.value()),
                MatrixType::load(kind.query_matrix_.value())
            );
        }
        return LeanVecType::reduce(data, leanvec_matrices);
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
        .results_ =
            svs::lib::save_to_table(vamana::ExpectedResult(std::move(kind), results))};
}

} // namespace

// target-registration.
void register_leanvec_static_search(vamana::StaticSearchDispatcher& dispatcher) {
    leanvec_specializations([&dispatcher]<
                                typename P,
                                typename S,
                                typename Q,
                                typename T,
                                typename D,
                                size_t L,
                                size_t N>() {
        auto method = &run_static_search<P, S, Q, T, D, L, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

void register_leanvec_static_build(vamana::StaticBuildDispatcher& dispatcher) {
    leanvec_specializations([&dispatcher]<
                                typename P,
                                typename S,
                                typename Q,
                                typename T,
                                typename D,
                                size_t L,
                                size_t N>() {
        auto method = &run_static_leanvec<P, S, Q, T, D, L, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

std::vector<TestFunction> register_leanvec_test_routines() {
    if constexpr (build_test_generators) {
        return std::vector<TestFunction>({
            // Searching
            &test_search<float, float, 64, svs::DistanceL2>,
            &test_search<leanvec::UsingLVQ<4>, leanvec::UsingLVQ<4>, 64, svs::DistanceL2>,
            &test_search<leanvec::UsingLVQ<4>, leanvec::UsingLVQ<8>, 64, svs::DistanceL2>,
            &test_search<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<4>, 64, svs::DistanceL2>,
            &test_search<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, 64, svs::DistanceL2>,
            &test_search<float, float, 96, svs::DistanceL2>,
            &test_search<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, 96, svs::DistanceL2>,
            // LeanVec OOD
            &test_search<float, float, 64, svs::DistanceL2, false>,
            &test_search<
                leanvec::UsingLVQ<8>,
                leanvec::UsingLVQ<8>,
                64,
                svs::DistanceL2,
                false>,
            // Building
            &test_build<float, float, 64, svs::DistanceL2>,
            &test_build<float, float, 64, svs::DistanceIP>,
            &test_build<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, 64, svs::DistanceL2>,
            &test_build<leanvec::UsingLVQ<8>, leanvec::UsingLVQ<8>, 64, svs::DistanceIP>,
            // LeanVec OOD
            &test_build<
                leanvec::UsingLVQ<8>,
                leanvec::UsingLVQ<8>,
                64,
                svs::DistanceL2,
                false>,
            &test_build<
                leanvec::UsingLVQ<8>,
                leanvec::UsingLVQ<8>,
                64,
                svs::DistanceIP,
                false>,
        });
    } else {
        return std::vector<TestFunction>();
    }
}

} // namespace svsbenchmark::vamana
