// svs-benchmark
#include "svs-benchmark/vamana/test.h"
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/executable.h"
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

struct TestGenerator {
  public:
    // Type Aliases
    using job_type = vamana::VamanaTest;
    using test_type = std::vector<vamana::TestFunction>;

    constexpr TestGenerator() = default;

    static std::string_view name() { return vamana::test_benchmark_name(); }

    static test_type tests() {
        auto generator = test_type{};
        svsbenchmark::append_to(generator, register_uncompressed_test_routines());
        svsbenchmark::append_to(generator, register_lvq_test_routines());
        svsbenchmark::append_to(generator, register_leanvec_test_routines());
        return generator;
    }

    static job_type example() { return job_type::example(); }
    static void print_help() { fmt::print("{}", HELP); }

    template <typename F>
    static std::optional<job_type>
    parse_args_and_invoke(F&& f, std::span<const std::string_view> args) {
        // We should have 1 or 2 additional arguments to parse, corresponding to the
        // number of threads and an optional data root.
        auto nargs = args.size();
        bool nargs_okay = nargs == 1 || nargs == 2;
        if (!nargs_okay) {
            fmt::print("Received too few arguments for Inverted test generation!");
            print_help();
            return std::nullopt;
        }

        auto num_threads = std::stoull(std::string{args[0]});
        auto data_root = std::optional<std::filesystem::path>();
        if (nargs == 2) {
            data_root = args[1];
        }
        return f(num_threads, data_root);
    }
};
} // namespace

// Return an executor for this benchmark.
std::unique_ptr<Benchmark> test_generator() {
    return std::make_unique<TestBasedExecutable<TestGenerator>>();
}
} // namespace svsbenchmark::vamana
