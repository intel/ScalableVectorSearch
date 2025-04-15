/**
 *    Copyright (C) 2024 Intel Corporation
 *
 *    This software and the related documents are Intel copyrighted materials,
 *    and your use of them is governed by the express license under which they
 *    were provided to you ("License"). Unless the License provides otherwise,
 *    you may not use, modify, copy, publish, distribute, disclose or transmit
 *    this software or the related documents without Intel's prior written
 *    permission.
 *
 *    This software and the related documents are provided as is, with no
 *    express or implied warranties, other than those that are expressly stated
 *    in the License.
 */

// SVS
#include "svs/core/recall.h"
#include "svs/extensions/flat/leanvec.h"
#include "svs/extensions/flat/lvq.h"
#include "svs/extensions/vamana/leanvec.h"
#include "svs/extensions/vamana/lvq.h"
#include "svs/orchestrators/dynamic_vamana.h"
#include "svs/orchestrators/exhaustive.h"
#include "svs/orchestrators/vamana.h"

#include "utils.h"

// Alternative main definition
#include "svsmain.h"

// SVS setup and parameters
size_t num_threads = 4;
size_t search_window_size = 20;
size_t n_neighbors = 1;
std::string dfname = "data.vecs";
std::string dfname_f16 = "data_f16.vecs";
std::string qfname = "query.vecs";
std::string gtfname = "gt.vecs";

const std::filesystem::path& config_path = "./config";
const std::filesystem::path& graph_path = "./graph";
// const std::filesystem::path& data_path = "./data";
const std::filesystem::path& config_path_dynamic = "./config_dynamic";
const std::filesystem::path& graph_path_dynamic = "./graph_dynamic";

void svs_setup() {
    // convert to fp16
    auto reader = svs::io::vecs::VecsReader<float>{dfname};
    auto writer = svs::io::vecs::VecsWriter<svs::Float16>{dfname_f16, reader.ndims()};
    {
        for (auto i : reader) {
            writer << i;
        }
    }
    writer.flush();
}

template <size_t P, size_t R, size_t E, typename S, typename A>
auto create_blocked_lvq_data() {
    namespace lvq = svs::quantization::lvq;
    using blocked_type = svs::data::Blocked<A>;

    auto compressor = svs::lib::Lazy([=](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::VectorDataLoader<svs::Float16, E>(dfname_f16).load();
        return lvq::LVQDataset<P, R, E, S, blocked_type>::compress(data, threadpool, 32);
    });

    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto data = svs::detail::dispatch_load(compressor, threadpool);
    fmt::print("Create LVQ data with P={}, R={}, E={}\n", P, R, E);
    return data;
}

template <typename P, typename S, size_t L, size_t D, typename A>
auto create_blocked_leanvec_data(size_t lean_dim = 32) {
    namespace leanvec = svs::leanvec;
    assert(lean_dim <= D);
    size_t leanvec_dim = (L == svs::Dynamic) ? lean_dim : L;
    using blocked_type = svs::data::Blocked<A>;

    auto compressor = svs::lib::Lazy([=](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::VectorDataLoader<svs::Float16, D>(dfname_f16).load();
        return leanvec::LeanDataset<P, S, L, D, blocked_type>::reduce(
            data, std::nullopt, threadpool, 32, svs::lib::MaybeStatic<L>(leanvec_dim)
        );
    });

    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto data = svs::detail::dispatch_load(compressor, threadpool);
    fmt::print("Create Leanvec data with L={}, leanvec_dim={}, D={}\n", L, leanvec_dim, D);
    return data;
}

float get_alpha(svs::distance::DistanceL2 /*dist*/) { return 1.2; }

float get_alpha(svs::distance::DistanceIP /*dist*/) { return 0.9; }

template <typename Data, typename Distance>
void vamana_build(Data& data, Distance distance) {
    auto parameters = svs::index::vamana::VamanaBuildParameters{
        get_alpha(distance), // alpha
        64,                  // graph max degree
        128,                 // search window size
        750,                 // max candidate pool size
        60,                  // prune to degree
        true,                // full search history
    };

    auto tic = svs::lib::now();
    svs::Vamana index = svs::Vamana::build<float>(parameters, data, distance, num_threads);
    auto build_time = svs::lib::time_difference(tic);
    fmt::print(
        "Vamana index build time: {} Distance: {}\n",
        build_time,
        svs::name(svs::distance_type_v<Distance>)
    );
    index.save("config", "graph", "data");
}

template <typename Data, typename Distance>
void vamana_search(Data& data, Distance distance) {
    auto index = svs::Vamana::assemble<float>(
        config_path, svs::GraphLoader(graph_path), data, distance, num_threads
    );

    index.set_search_window_size(search_window_size);
    const auto query_data = svs::load_data<float>(qfname);
    const auto groundtruth = svs::load_data<int>(gtfname);

    auto tic = svs::lib::now();
    auto query_result = index.search(query_data, n_neighbors);
    auto search_time = svs::lib::time_difference(tic);

    std::vector<double> qps;
    for (int i = 0; i < 5; i++) {
        tic = svs::lib::now();
        query_result = index.search(query_data, n_neighbors);
        search_time = svs::lib::time_difference(tic);
        qps.push_back(query_data.size() / search_time);
    }

    auto recall = svs::k_recall_at_n(groundtruth, query_result, n_neighbors, n_neighbors);
    // fmt::print("Raw QPS: {:7.3f} \n", fmt::join(qps, ", "));
    fmt::print(
        "Vamana Distance: {}, sws: {}, Recall: {}, Max QPS: {:7.3f} \n",
        svs::name(svs::distance_type_v<Distance>),
        search_window_size,
        recall,
        *std::max_element(qps.begin(), qps.end())
    );
}

template <typename Data> void vamana_build_search(Data& data) {
    vamana_build(data, svs::distance::DistanceL2());
    vamana_search(data, svs::distance::DistanceL2());

    vamana_build(data, svs::distance::DistanceIP());
    vamana_search(data, svs::distance::DistanceIP());
}

template <typename Data, typename Distance>
void dynamic_vamana_build(Data& data, Distance distance) {
    auto parameters = svs::index::vamana::VamanaBuildParameters{
        get_alpha(distance), // alpha
        64,                  // graph max degree
        128,                 // search window size
        750,                 // max candidate pool size
        60,                  // prune to degree
        true,                // full search history
    };

    auto tic = svs::lib::now();
    std::vector<size_t> ids(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        ids[i] = i;
    }

    svs::DynamicVamana index = svs::DynamicVamana::build<float>(
        parameters, data, svs::lib::as_span(ids), distance, num_threads
    );
    auto build_time = svs::lib::time_difference(tic);
    fmt::print(
        "DynamicVamana index build time: {} Distance: {}\n",
        build_time,
        svs::name(svs::distance_type_v<Distance>)
    );
    index.save("config_dynamic", "graph_dynamic", "data_dynamic");
}

template <typename Data, typename Distance>
void dynamic_vamana_search(Data& data, Distance distance) {
    using Idx = uint32_t;
    auto index = svs::DynamicVamana::assemble<float>(
        config_path_dynamic,
        SVS_LAZY(svs::graphs::SimpleBlockedGraph<Idx>::load(graph_path_dynamic)),
        data,
        distance,
        num_threads
    );

    index.set_search_window_size(search_window_size);
    const auto query_data = svs::load_data<float>(qfname);
    const auto groundtruth = svs::load_data<int>(gtfname);

    auto tic = svs::lib::now();
    auto query_result = index.search(query_data, n_neighbors);
    auto search_time = svs::lib::time_difference(tic);

    std::vector<double> qps;
    for (int i = 0; i < 5; i++) {
        tic = svs::lib::now();
        query_result = index.search(query_data, n_neighbors);
        search_time = svs::lib::time_difference(tic);
        qps.push_back(query_data.size() / search_time);
    }

    auto recall = svs::k_recall_at_n(groundtruth, query_result, n_neighbors, n_neighbors);
    // fmt::print("Raw QPS: {:7.3f} \n", fmt::join(qps, ", "));
    fmt::print(
        "DynamicVamana Distance: {}, sws: {}, Recall: {}, Max QPS: {:7.3f} \n",
        svs::name(svs::distance_type_v<Distance>),
        search_window_size,
        recall,
        *std::max_element(qps.begin(), qps.end())
    );
}

template <typename Data> void dynamic_vamana_build_search(Data& data) {
    dynamic_vamana_build(data, svs::distance::DistanceL2());
    dynamic_vamana_search(data, svs::distance::DistanceL2());

    dynamic_vamana_build(data, svs::distance::DistanceIP());
    dynamic_vamana_search(data, svs::distance::DistanceIP());
}

template <typename Data, typename Distance>
void flat_search(Data& data, Distance distance) {
    svs::Flat index = svs::Flat::assemble<float>(data, distance, num_threads);

    const auto query_data = svs::load_data<float>(qfname);
    const auto groundtruth = svs::load_data<int>(gtfname);

    auto tic = svs::lib::now();
    auto query_result = index.search(query_data, n_neighbors);
    auto search_time = svs::lib::time_difference(tic);

    std::vector<double> qps;
    for (int i = 0; i < 5; i++) {
        tic = svs::lib::now();
        query_result = index.search(query_data, n_neighbors);
        search_time = svs::lib::time_difference(tic);
        qps.push_back(query_data.size() / search_time);
    }

    auto recall = svs::k_recall_at_n(groundtruth, query_result, n_neighbors, n_neighbors);
    // fmt::print("Raw QPS: {:7.3f} \n", fmt::join(qps, ", "));
    fmt::print(
        "Flat search Distance: {}, Recall: {}, Max QPS: {:7.3f} \n",
        svs::name(svs::distance_type_v<Distance>),
        recall,
        *std::max_element(qps.begin(), qps.end())
    );
}

template <typename Data> void flat_search(Data& data) {
    flat_search(data, svs::distance::DistanceL2());
    flat_search(data, svs::distance::DistanceIP());
}

template <size_t L, size_t D, typename A> void all_search() {
    using S = svs::quantization::lvq::Sequential;
    using S1 = svs::quantization::lvq::Turbo<16, 8>;

    // Flat Index
    {
        auto data = create_blocked_lvq_data<4, 0, D, S, A>();
        flat_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 4, D, S, A>();
        flat_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 8, D, S, A>();
        flat_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 0, D, S1, A>();
        flat_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 4, D, S1, A>();
        flat_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 8, D, S1, A>();
        flat_search(data);
    }

    // Vamana Index
    {
        auto data = create_blocked_lvq_data<4, 0, D, S, A>();
        vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 4, D, S, A>();
        vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 8, D, S, A>();
        vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 0, D, S1, A>();
        vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 4, D, S1, A>();
        vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 8, D, S1, A>();
        vamana_build_search(data);
    }

    {
        using P = svs::leanvec::UsingLVQ<8>;
        using S = svs::leanvec::UsingLVQ<8>;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        vamana_build_search(data);
    }

    {
        using P = svs::leanvec::UsingLVQ<4>;
        using S = svs::leanvec::UsingLVQ<8>;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        vamana_build_search(data);
    }

    {
        using P = svs::leanvec::UsingLVQ<8>;
        using S = svs::Float16;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        vamana_build_search(data);
    }

    {
        using P = svs::Float16;
        using S = svs::Float16;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        vamana_build_search(data);
    }

    {
        using P = float;
        using S = float;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        vamana_build_search(data);
    }

    // Dynamic vamana Index
    {
        auto data = create_blocked_lvq_data<4, 0, D, S, A>();
        dynamic_vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 4, D, S, A>();
        dynamic_vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 8, D, S, A>();
        dynamic_vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 0, D, S1, A>();
        dynamic_vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 4, D, S1, A>();
        dynamic_vamana_build_search(data);
    }

    {
        auto data = create_blocked_lvq_data<4, 8, D, S1, A>();
        dynamic_vamana_build_search(data);
    }

    {
        using P = svs::leanvec::UsingLVQ<8>;
        using S = svs::leanvec::UsingLVQ<8>;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        dynamic_vamana_build_search(data);
    }

    {
        using P = svs::leanvec::UsingLVQ<4>;
        using S = svs::leanvec::UsingLVQ<8>;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        dynamic_vamana_build_search(data);
    }

    {
        using P = svs::leanvec::UsingLVQ<8>;
        using S = svs::Float16;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        dynamic_vamana_build_search(data);
    }

    {
        using P = svs::Float16;
        using S = svs::Float16;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        dynamic_vamana_build_search(data);
    }

    {
        using P = float;
        using S = float;
        auto data = create_blocked_leanvec_data<P, S, L, D, A>();
        dynamic_vamana_build_search(data);
    }
}

// Alternative main definition
int svs_main(std::vector<std::string> SVS_UNUSED(args)) {
    using A = svs::lib::Allocator<std::byte>;
    const size_t D = 512;
    size_t dataset_size = 100;
    size_t query_size = 10;


    generate_random_data(D, dataset_size, query_size);
    svs_setup();
    all_search<svs::Dynamic, svs::Dynamic, A>();

    return 0;
}

// Special main providing some helpful utilties.
SVS_DEFINE_MAIN();
