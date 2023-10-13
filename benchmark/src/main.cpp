// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/vamana/build.h"

// stl
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

class ExecutableDispatcher {
  public:
    ///// Type aliases
    using map_type =
        std::unordered_map<std::string, std::unique_ptr<svsbenchmark::Benchmark>>;

  private:
    ///// Members
    map_type executables_ = {};

  public:
    ///// Constructor
    ExecutableDispatcher() = default;

    ///// API
    void register_executable(std::unique_ptr<svsbenchmark::Benchmark> exe) {
        auto name = exe->name();

        // Check if this executable is already registered.
        if (lookup(name) != nullptr) {
            throw ANNEXCEPTION(
                "An executable with the name ", name, " is already registered!"
            );
        }
        executables_[std::move(name)] = std::move(exe);
    }

    std::vector<std::string> executables() const {
        auto names = std::vector<std::string>();
        for (const auto& kv : executables_) {
            names.push_back(kv.first);
        }
        std::sort(names.begin(), names.end());
        return names;
    }

    bool call(const std::string& name, std::span<const std::string_view> args) const {
        auto* f = lookup(name);
        if (f) {
            f->run(args);
            return true;
        }
        return false;
    }

  private:
    svsbenchmark::Benchmark* lookup(const std::string& name) const {
        if (auto itr = executables_.find(name); itr != executables_.end()) {
            return itr->second.get();
        }
        return nullptr;
    }
};

ExecutableDispatcher build_dispatcher() {
    auto dispatcher = ExecutableDispatcher();
    dispatcher.register_executable(svsbenchmark::vamana::static_workflow());
    dispatcher.register_executable(svsbenchmark::vamana::dynamic_workflow());
    return dispatcher;
}
} // namespace

std::span<const std::string_view>
get_executable_arguments(const std::vector<std::string_view>& arguments) {
    auto begin = arguments.begin() + 2;
    auto end = arguments.end();
    return {begin, end};
}

void print_help(const ExecutableDispatcher& dispatcher, std::string_view prefix = "") {
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

    bool success = dispatcher.call(exe, get_executable_arguments(arguments));
    if (!success) {
        print_help(dispatcher, fmt::format("Could not find executable \"{}\".", exe));
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if constexpr (svsbenchmark::is_minimal) {
        fmt::print("WARNING! The benchmark executable was compiled in minimal mode!\n");
    }

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
