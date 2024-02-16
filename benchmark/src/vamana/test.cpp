// svs-benchmark
#include "svs-benchmark/vamana/test.h"
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
#include <iterator>
#include <memory>
#include <optional>

namespace svsbenchmark::vamana {
namespace {

std::vector<vamana::TestFunction> get_generators() {
    auto generator = std::vector<vamana::TestFunction>();
    svsbenchmark::append_to(generator, register_uncompressed_test_routines());
    svsbenchmark::append_to(generator, register_lvq_test_routines());
    svsbenchmark::append_to(generator, register_leanvec_test_routines());
    return generator;
}

const char* HELP = R"(
Generate reference results for the Vamana index.

Usage:
    (1) src-file.toml output-file.toml num_threads [basename]
    (2) --help
    (3) --example

1. Run the test generators using `src-file.toml` as the test driver input. (see (3))
   Store the post-processed results into `output-file.toml`.
   Third argument `num_threads` sets the number of worker threads to use for each job.
   Optional fourth argument `basename` will be used as the root for all file-paths parsed
   from `src-file.toml`.

   The output results will be saved to `output-file.toml` as a dictionary with the following
   structure:

   "vamana_test_search" : Array of serialized `svsbenchmark::vamana::ExpectedResult` for
     each search-only job registered. None of these entries should have the
     `build_parameters` field present.

   "vamana_test_build" : Array of serialized `svsbenchmark::vamana::ExpectedResult` for
     each build-job registered. All of these entries should have the `build_parameters`
     field present.

2. Print this message.

3. Display an example input TOML file to `stdout`.
)";

void print_help() { fmt::print("{}", HELP); }

void print_example() {
    using job_type = vamana::VamanaTest;
    std::cout << "The example provides a skeleton TOML file for static vamana index "
                 "building\n\n";

    auto table =
        toml::table({{test_benchmark_name(), svs::lib::save_to_table(job_type::example())}}
        );
    std::cout << table << '\n';
}

int run_benchmark(
    const std::filesystem::path& config_path,
    const std::filesystem::path& destination_path,
    const std::optional<std::filesystem::path>& data_root,
    size_t num_threads
) {
    auto table = toml::parse_file(config_path.native());
    auto job = svs::lib::load<vamana::VamanaTest>(
        svs::lib::node_view_at(table, test_benchmark_name()), data_root
    );
    auto generators = get_generators();

    // Check that appropriate specializations exist for all jobs.
    auto results = toml::table({{"start_time", svs::date_time()}});
    for (const auto& f : generators) {
        auto [key, job_results] = f(job, num_threads);
        append_or_create(results, std::move(job_results), key);
        atomic_save(results, destination_path);
    }
    // Save a copy before post-processing just in case.
    results.emplace("stop_time", svs::date_time());
    atomic_save(results, destination_path);
    return 0;
}

int run_benchmark(std::span<const std::string_view> args) {
    // Handle argument parsing.
    // Steps:
    // * Handle 0 argument cases.
    // * Deal with special arguments.
    // * Normal run.
    auto nargs = args.size();
    if (nargs == 0) {
        print_help();
        return 0;
    }
    auto first_arg = args[0];
    if (first_arg == "help" || first_arg == "--help") {
        print_help();
        return 0;
    }
    if (first_arg == "--example") {
        print_example();
        return 0;
    }

    // Normal argument parsing.
    if (nargs < 3 || nargs > 4) {
        fmt::print("Expected 3 or 4 arguments. Instead, got {}.\n", nargs);
        print_help();
        return 0;
    }

    // At this point, we have the correct number of arguments. Assume we're going to
    // proceed with a normal run.
    auto config_path = std::filesystem::path(first_arg);
    auto destination_path = std::filesystem::path(args[1]);
    auto num_threads = std::stoull(std::string{args[2]});
    auto data_root = std::optional<std::filesystem::path>();
    if (nargs == 4) {
        data_root = args[3];
    }
    return run_benchmark(config_path, destination_path, data_root, num_threads);
}

class TestGeneration : public Benchmark {
  public:
    TestGeneration() = default;

  protected:
    std::string do_name() const override { return std::string(test_benchmark_name()); }

    int do_run(std::span<const std::string_view> args) const override {
        return run_benchmark(args);
    }
};
} // namespace

// Return an executor for this benchmark.
std::unique_ptr<Benchmark> test_generator() { return std::make_unique<TestGeneration>(); }
} // namespace svsbenchmark::vamana
