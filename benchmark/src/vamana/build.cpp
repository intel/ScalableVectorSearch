// svs-benchmark
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/vamana/leanvec.h"
#include "svs-benchmark/vamana/lvq.h"
#include "svs-benchmark/vamana/uncompressed.h"

// svs
#include "svs/lib/saveload.h"
#include "svs/third-party/toml.h"

// third-party
#include "fmt/core.h"
#include "fmt/ranges.h"

// stl
#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

namespace svsbenchmark::vamana {
namespace {

template <typename BenchmarkType> struct BuildDispatcher;

template <typename BenchmarkType> auto get_dispatcher() {
    return BuildDispatcher<BenchmarkType>::dispatcher();
}

/////
///// Static Builds
/////

auto static_dispatcher() {
    auto dispatcher = vamana::StaticBuildDispatcher{};
    vamana::register_uncompressed_static_build(dispatcher);
    vamana::register_lvq_static_build(dispatcher);
    vamana::register_leanvec_static_build(dispatcher);
    return dispatcher;
}

/////
///// Dynamic Builds
/////

auto dynamic_dispatcher() {
    auto dispatcher = vamana::DynamicBuildDispatcher{};
    vamana::register_uncompressed_dynamic_build(dispatcher);
    vamana::register_lvq_dynamic_build(dispatcher);
    return dispatcher;
}

// Dispatcher for Uncompressed Data.
template <> struct BuildDispatcher<StaticBenchmark> {
    static auto dispatcher() { return static_dispatcher(); }
};

template <> struct BuildDispatcher<DynamicBenchmark> {
    static auto dispatcher() { return dynamic_dispatcher(); }
};

bool check_job(
    StaticBenchmark SVS_UNUSED(overload_tag), const associated_job_t<StaticBenchmark>& job
) {
    auto dispatcher = get_dispatcher<StaticBenchmark>();
    return dispatcher.has_match(
        job.dataset_, job.query_type_, job.data_type_, job.distance_, job.ndims_, job
    );
}

toml::table run_job(
    StaticBenchmark SVS_UNUSED(overload_tag),
    const associated_job_t<StaticBenchmark>& job,
    const Checkpoint& SVS_UNUSED(checkpointer)
) {
    auto dispatcher = get_dispatcher<StaticBenchmark>();
    return dispatcher.invoke(
        job.dataset_, job.query_type_, job.data_type_, job.distance_, job.ndims_, job
    );
}

bool check_job(
    DynamicBenchmark SVS_UNUSED(overload_tag), const associated_job_t<DynamicBenchmark>& job
) {
    auto dispatcher = get_dispatcher<DynamicBenchmark>();
    return dispatcher.has_match(
        job.dataset_,
        job.query_type_,
        job.data_type_,
        job.distance_,
        job.ndims_,
        job,
        svsbenchmark::Checkpoint()
    );
}

toml::table run_job(
    DynamicBenchmark SVS_UNUSED(overload_tag),
    const associated_job_t<DynamicBenchmark>& job,
    const Checkpoint& checkpointer
) {
    auto dispatcher = get_dispatcher<DynamicBenchmark>();
    return dispatcher.invoke(
        job.dataset_,
        job.query_type_,
        job.data_type_,
        job.distance_,
        job.ndims_,
        job,
        checkpointer
    );
}

// Parse the jobs to run.
template <typename BenchmarkType>
std::vector<associated_job_t<BenchmarkType>> parse_jobs(
    const std::filesystem::path& config_path,
    const std::optional<std::filesystem::path>& data_root
) {
    using job_type = associated_job_t<BenchmarkType>;
    // Parse the configuration file.
    toml::table configuration = toml::parse_file(std::string(config_path));
    return svs::lib::load_at<std::vector<job_type>>(
        configuration, benchmark_name(BenchmarkType()), data_root
    );
}

constexpr std::string_view HELP_TEMPLATE = R"(
Run a {} benchmark for the Vamana index.

Usage:
    (1) src-file.toml output-file.toml [basename]
    (2) --help
    (3) --example

1. Run all the benchmarks in the global `{}` array in `src-file.toml`.
   All elements in the array must be parseable as a `{}`.

   Results will be saved to `output-file.toml`.

   Optional third argument `basename` will be used as the root for all file paths parsed.

2. Print this help message.

3. Display an example input TOML file to `stdout`.

Backend specializations are dispatched on the following fields of the input TOML file:
* build_type: The dataset type to use.
* query_type: The element type of the query dataset.
* data_type: The input type of the source dataset.
* distance: The distance function to use.
* ndims: The compile-time dimensionality.

Compiled specializations are listed below:
{{ build_type, query_type, data_type, distance, ndims }}
)";

template <typename BenchmarkType> void print_help() {
    if constexpr (std::is_same_v<BenchmarkType, StaticBenchmark>) {
        fmt::print(
            HELP_TEMPLATE,
            "static build and search",
            benchmark_name(BenchmarkType()),
            "svsbenchmark::Vamana::BuildJob"
        );
    } else if constexpr (std::is_same_v<BenchmarkType, DynamicBenchmark>) {
        fmt::print(
            HELP_TEMPLATE,
            "dynamic build, modification, and search",
            benchmark_name(BenchmarkType()),
            "svsbenchmark::Vamana::DynamicBuildJob"
        );
    } else {
        throw ANNEXCEPTION("Unreachable");
    }
    auto dispatcher = get_dispatcher<BenchmarkType>();
    for (size_t i = 0; i < dispatcher.size(); ++i) {
        auto dispatch_strings = std::array<std::string, 5>{
            dispatcher.description(i, 0),
            dispatcher.description(i, 1),
            dispatcher.description(i, 2),
            dispatcher.description(i, 3),
            dispatcher.description(i, 4),
        };
        fmt::print("{{ {} }}\n", fmt::join(dispatch_strings, ", "));
    }
}

template <typename BenchmarkType> void print_example() {
    using job_type = associated_job_t<BenchmarkType>;
    auto results = toml::array();
    results.push_back(svs::lib::save_to_table(job_type::example()));
    std::cout << "The example provides a skeleton TOML file for static vamana index "
                 "building\n\n";

    auto table = toml::table({{benchmark_name(BenchmarkType()), std::move(results)}});
    std::cout << table << '\n';
}

template <typename BenchmarkType>
int run_build_benchmark(
    const std::filesystem::path& config_path,
    const std::filesystem::path& destination_path,
    const std::optional<std::filesystem::path>& data_root
) {
    auto jobs = parse_jobs<BenchmarkType>(config_path, data_root);

    // Check that appropriate specializations exist for all jobs.
    bool okay = true;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto& job = jobs.at(i);
        if (!check_job(BenchmarkType(), job)) {
            fmt::print("Unimplemented specialization for job number {}!\n", i);
            fmt::print("Contents of the job are given below:\n");
            std::cout << svs::lib::save_to_table(job) << "\n\n";
            okay = false;
        }
    }

    if (!okay) {
        print_help<BenchmarkType>();
        return 1;
    }

    auto results = toml::table({{"start_time", svs::date_time()}});
    for (const auto& job : jobs) {
        append_or_create(
            results,
            run_job(BenchmarkType(), job, Checkpoint(results, destination_path)),
            benchmark_name(BenchmarkType())
        );
        atomic_save(results, destination_path);
    }
    // Save a copy before post-processing just in case.
    results.emplace("stop_time", svs::date_time());
    atomic_save(results, destination_path);
    return 0;
}

template <typename BenchmarkType>
int run_build_benchmark(std::span<const std::string_view> args) {
    // Handle argument parsing.
    // Steps:
    // * Handle 0 argument cases.
    // * Deal with special arguments.
    // * Normal run.
    auto nargs = args.size();
    if (nargs == 0) {
        print_help<BenchmarkType>();
        return 0;
    }
    auto first_arg = args[0];
    if (first_arg == "help" || first_arg == "--help") {
        print_help<BenchmarkType>();
        return 0;
    }
    if (first_arg == "--example") {
        print_example<BenchmarkType>();
        return 0;
    }

    // Normal argument parsing.
    if (nargs < 2 || nargs > 3) {
        fmt::print("Expected 2 or 3 arguments. Instead, got {}.\n", nargs);
        print_help<BenchmarkType>();
        return 0;
    }

    // At this point, we have the correct number of arguments. Assume we're going to
    // proceed with a normal run.
    auto config_path = std::filesystem::path(first_arg);
    auto destination_path = std::filesystem::path(args[1]);
    auto data_root = std::optional<std::filesystem::path>();
    if (nargs == 3) {
        data_root = args[2];
    }
    return run_build_benchmark<BenchmarkType>(config_path, destination_path, data_root);
}

template <typename BenchmarkType> class BuildBenchmark : public Benchmark {
  public:
    BuildBenchmark() = default;

  protected:
    std::string do_name() const override {
        return std::string(benchmark_name(BenchmarkType()));
    }

    int do_run(std::span<const std::string_view> args) const override {
        return run_build_benchmark<BenchmarkType>(args);
    }
};
} // namespace

// Return an executor for this benchmark.
std::unique_ptr<Benchmark> static_workflow() {
    return std::make_unique<BuildBenchmark<StaticBenchmark>>();
}
std::unique_ptr<Benchmark> dynamic_workflow() {
    return std::make_unique<BuildBenchmark<DynamicBenchmark>>();
}
} // namespace svsbenchmark::vamana
