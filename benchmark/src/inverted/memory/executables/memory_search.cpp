// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/executable.h"
#include "svs-benchmark/search.h"

#include "svs-benchmark/inverted/memory/lvq.h"
#include "svs-benchmark/inverted/memory/register.h"
#include "svs-benchmark/inverted/memory/search.h"
#include "svs-benchmark/inverted/memory/uncompressed.h"

// stl
#include <algorithm>
#include <string>
#include <vector>

// Application logic.
namespace svsbenchmark::inverted::memory {
namespace {

constexpr std::string_view MEMORY_SEARCH_HELP = R"(
Run a search-only benchmark for the Inverted index.

Usage:
    (1) src-file.toml (output-file.toml/--validate) [basename]
    (2) --help
    (3) --example

1. Run all the benchmarks in the global `search_inverted_static` array in `src-file.toml`.
   All elements in the array must be parseable as a
   ``svsbenchmark::inverted::memory::MemorySearchJob``.

   Results will be saved to `output-file.toml`.

   If `--validate` is given as the second argument, then all pre-run checks will be
   performed on the input file and arguments but not benchmark will actually be run.

   Optional third argument `basename` will be used as the root for all file paths parsed.

2. Print this help message.

3. Display an example input TOML file to `stdout`.

Backend specializations are dispatched on the following fields of the input TOML file:
* build_type: The dataset type to use.
* query_type: The element type of the query dataset.
* data_type: The input type of the source dataset.
* distance: The distance function to use.
* strategy: The technique used to store the clustered dataset.
* ndims: The compile-time dimensionality.

Compiled specializations are listed below:
{build_type, query_type, data_type, distance, strategy, ndims}
)";

struct MemorySearch {
  public:
    // Type Aliases
    using job_type = memory::MemorySearchJob;
    using dispatcher_type = memory::MemorySearchDispatcher;

    static dispatcher_type dispatcher() {
        auto dispatcher = dispatcher_type();
        svsbenchmark::inverted::memory::register_uncompressed_memory_search(dispatcher);
        svsbenchmark::inverted::memory::register_lvq_memory_search(dispatcher);
        return dispatcher;
    }

    static std::string_view name() { return static_search_name(); }

    static void print_help() {
        fmt::print("{}\n", MEMORY_SEARCH_HELP);
        auto f = dispatcher();
        for (size_t i = 0; i < f.size(); ++i) {
            auto dispatch_strings = std::array<std::string, 6>{
                f.description(i, 0),
                f.description(i, 1),
                f.description(i, 2),
                f.description(i, 3),
                f.description(i, 4),
                f.description(i, 5),
            };
            fmt::print("{{ {} }}\n", fmt::join(dispatch_strings, ", "));
        }
    }

    static job_type example() { return job_type::example(); }

    template <typename F>
    static std::optional<std::vector<job_type>>
    parse_args_and_invoke(F&& f, std::span<const std::string_view> args) {
        auto root = std::optional<std::filesystem::path>();
        if (args.size() == 1) {
            root = std::filesystem::path(args[0]);
        }
        return f(root);
    }
};
} // namespace

std::unique_ptr<Benchmark> static_search() {
    return std::make_unique<JobBasedExecutable<MemorySearch>>();
}
} // namespace svsbenchmark::inverted::memory
