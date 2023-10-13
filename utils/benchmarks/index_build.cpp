/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#include "svs/core/recall.h"
#include "svs/index/flat/flat.h"
#include "svs/index/vamana/index.h"
#include "svs/lib/timing.h"
#include "svs/third-party/fmt.h"

#include "svsmain.h"

// Compile-time Settings
using Eltype = svs::Float16;
using QueryEltype = float;
inline constexpr auto global_distance = svs::distance::DistanceL2();
const size_t Dims = 96;
const size_t NumNeighbors = 10;

namespace {

svs::VectorDataLoader<Eltype, Dims> make_data_loader(const std::filesystem::path& path) {
    return svs::VectorDataLoader<Eltype, Dims>(path);
}

svs::QueryResult<size_t> compute_groundtruth(
    const std::filesystem::path& data_path,
    const svs::data::SimplePolymorphicData<QueryEltype>& queries,
    svs::lib::Timer& timer,
    size_t num_threads
) {
    auto handle = timer.push_back("compute groundtruth");
    auto index = svs::index::flat::auto_assemble(
        make_data_loader(data_path), global_distance, num_threads
    );
    return index.search(queries, 100);
}

struct BenchmarkResult {
    BenchmarkResult(
        size_t search_window_size, size_t num_neighbors, double qps, double recall
    )
        : search_window_size{search_window_size}
        , num_neighbors{num_neighbors}
        , qps{qps}
        , recall{recall} {}

    size_t search_window_size;
    size_t num_neighbors;
    double qps;
    double recall;
};

struct BuildSetup {
    size_t construction_window_size;
    size_t max_degree;
    float alpha;
};

std::vector<BenchmarkResult> benchmark(
    const std::filesystem::path& data_path,
    const svs::data::SimplePolymorphicData<QueryEltype>& queries,
    const svs::QueryResult<size_t>& groundtruth,
    svs::lib::Timer& timer,
    const BuildSetup& build_setup,
    size_t num_threads
) {
    auto parameters = svs::index::vamana::VamanaBuildParameters{
        build_setup.alpha,
        build_setup.max_degree,
        build_setup.construction_window_size,
        1000,
        build_setup.max_degree,
        true};

    auto build_time = timer.push_back("index build");
    auto index = svs::index::vamana::auto_build(
        parameters,
        make_data_loader(data_path),
        global_distance,
        num_threads,
        svs::HugepageAllocator()
    );
    build_time.finish();

    auto search_window_sizes = std::vector<size_t>{10, 15, 20, 25, 30, 35, 40, 45, 50};
    auto benchmark_results = std::vector<BenchmarkResult>();

    for (auto sws : search_window_sizes) {
        auto query_result = svs::QueryResult<size_t>();
        const size_t nloops = 10;
        auto total_search_time = timer.push_back("search");
        auto label = fmt::format("search {}", sws);
        for (size_t i = 0; i < nloops; ++i) {
            auto search_time = timer.push_back(label);
            query_result = index.search(queries, sws);
        }
        double elapsed = svs::lib::as_seconds(total_search_time.finish());
        double qps = (nloops * queries.size()) / elapsed;
        double recall =
            svs::k_recall_at_n(groundtruth, query_result, NumNeighbors, NumNeighbors);
        benchmark_results.emplace_back(sws, sws, qps, recall);
    }
    return benchmark_results;
}

} // namespace

template <> struct fmt::formatter<BenchmarkResult> : svs::format_empty {
    auto format(const auto& x, auto& ctx) const {
        return fmt::format_to(
            ctx.out(),
            "{{ sws = {}, knn = {}, qps = {}, recall = {} }}",
            x.search_window_size,
            x.num_neighbors,
            x.qps,
            x.recall
        );
    }
};

int svs_main(std::vector<std::string> args) {
    size_t i = 1;
    const auto& data_path = args.at(i++);
    const auto& query_path = args.at(i++);
    auto num_threads = std::stoull(args.at(i++));

    auto timer = svs::lib::Timer();
    auto load_timer = timer.push_back("data loading");
    auto queries = svs::io::auto_load<QueryEltype>(query_path);
    load_timer.finish();

    auto groundtruth = compute_groundtruth(data_path, queries, timer, num_threads);

    auto construction_window_sizes = std::vector<size_t>{32, 64, 128};
    auto graph_degrees = std::vector<size_t>{32, 64, 128};

    auto result_strings = std::vector<std::string>();
    for (auto window_size : construction_window_sizes) {
        for (auto degree : graph_degrees) {
            auto label = fmt::format("build (sws = {}, gd = {})", window_size, degree);
            auto guard = timer.push_back(label);
            auto results = benchmark(
                data_path,
                queries,
                groundtruth,
                timer,
                BuildSetup{window_size, degree, 1.2},
                num_threads
            );
            result_strings.push_back(fmt::format("{}: {}", label, fmt::join(results, ", "))
            );
        }
    }

    fmt::print("RESULTS\n");
    for (const auto& str : result_strings) {
        fmt::print("{}\n", str);
    }
    fmt::print("TIMINGS\n");

    timer.print();
    return 0;
}

SVS_DEFINE_MAIN();
