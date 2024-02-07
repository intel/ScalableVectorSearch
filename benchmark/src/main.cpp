// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/datasets.h"
// vamana
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/vamana/search.h"
#include "svs-benchmark/vamana/test.h"
// inverted
#include "svs-benchmark/inverted/inverted.h"

// stl
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {
svsbenchmark::ExecutableDispatcher build_dispatcher() {
    auto dispatcher = svsbenchmark::ExecutableDispatcher();
    // vamana
    dispatcher.register_executable(svsbenchmark::vamana::search_static_workflow());
    dispatcher.register_executable(svsbenchmark::vamana::static_workflow());
    dispatcher.register_executable(svsbenchmark::vamana::dynamic_workflow());
    dispatcher.register_executable(svsbenchmark::vamana::test_generator());
    // inverted
    svsbenchmark::inverted::register_executables(dispatcher);
    // documentation
    svsbenchmark::register_dataset_documentation(dispatcher);
    return dispatcher;
}

} // namespace

std::span<const std::string_view>
get_executable_arguments(const std::vector<std::string_view>& arguments) {
    auto begin = arguments.begin() + 2;
    auto end = arguments.end();
    return {begin, end};
}

void print_help(
    const svsbenchmark::ExecutableDispatcher& dispatcher, std::string_view prefix = ""
) {
    // Set up the print-outs so `True` means we are compiling more things into the
    // final binary.
    //
    // Less difficult to visually parse this way.
    fmt::print("SVS Benchmarking Executable\n");
    fmt::print("        Benchmarks Built: {}\n", !svsbenchmark::is_minimal);
    fmt::print("   Test Generators Built: {}\n\n", svsbenchmark::build_test_generators);

    if (!prefix.empty()) {
        fmt::print("{}\n", prefix);
    }

    fmt::print("The following executables are registered with the benchmarking program:\n");
    auto names = dispatcher.executables();
    for (const auto& name : names) {
        fmt::print("    {}\n", name);
    }
}

int main_bootstrapped(const std::vector<std::string_view>& arguments) {
    auto dispatcher = build_dispatcher();

    // First level argument handling.
    auto nargs = arguments.size();
    if (nargs == 1) {
        print_help(dispatcher);
        return 1;
    }
    auto exe = std::string(arguments.at(1));
    if (exe == "help" || exe == "--help") {
        print_help(dispatcher);
        return 1;
    }

    // Warn if the library was compiled in minimal mode and we're calling an actual
    // executable.
    if constexpr (svsbenchmark::is_minimal) {
        fmt::print("WARNING! The benchmark executable was compiled in minimal mode!\n");
    }
    bool success = dispatcher.call(exe, get_executable_arguments(arguments));
    if (!success) {
        print_help(dispatcher, fmt::format("Could not find executable \"{}\".", exe));
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    // The very first thing we do is get the arguments into a more useful form.
    auto arguments = std::vector<std::string_view>();
    for (auto* i : std::span(argv, argc)) {
        arguments.emplace_back(i);
    }
    int return_code = 0;
    try {
        return_code = main_bootstrapped(arguments);
    } catch (const svs::ANNException& err) {
        std::cerr << "Application terminated with ANNException: " << err.what()
                  << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& err) {
        std::cerr << "Application terminated with unknown exception: " << err.what()
                  << std::endl;
        return EXIT_FAILURE;
    }
    return return_code;
}
