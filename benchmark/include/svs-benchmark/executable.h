#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"

// svs
#include "svs/lib/preprocessor.h"
#include "svs/lib/saveload.h"
#include "svs/third-party/fmt.h"

#include <concepts>
#include <span>

namespace svsbenchmark {

namespace detail {

struct IsHelp {
    inline constexpr bool operator()(std::string_view str) const {
        return str == "-h" || str == "help" || str == "--help";
    }
};
inline constexpr IsHelp is_help{};

struct IsExample {
    inline constexpr bool operator()(std::string_view str) const {
        return str == "--example";
    }
};
inline constexpr IsExample is_example{};

} // namespace detail

///
/// @brief An executable for operations that consist of multiple homogeneous jobs.
///
/// Requirements for `Implementation`:
///
/// @code{cpp}
/// class Implementation {
///   public:
///     using job_type = /*implementation-defined*/;
///     using dispatcher_type = /*implementation-defined*/;
///
///     // Return a dispatcher for the executable.
///     dispatcher_type dispatcher() const;
///
///     // Return an example job to serve as a prototype.
///     job_type example() const;
///
///     // The name assoicated with the job. This will be used to both pull jobs from
///     // the input TOML file as well as be the key under which results will be collected
///     // in output TOML file.
///     std::convertible_to<std::string> name() const;
///
///     // Print the help message.
///     void print_help() const;
///
///     // Parse any command-line arguments needed specifically by this job and invoke
///     // the callable.
///     //
///     // `args` will contain the remaining commandline arguments after:
///     //    - svs_benchmark
///     //    - executable_name
///     //    - job_source_file.toml
///     //    - job_destination_file.toml
///     // `f` is a partially appliec function. Any arguments passed to `f` will be
///     //   forwarded to `svs::load` for `std::vector<job_type>`.
///     //
///     // Returns a empty optional if argument parsing failed. The callee must supply
///     // a diagnostic print-out on failure.
///     template<typename F>
///     std::optional<std::vector<job_type>>
///     parse_args_and_invoke(F&& f, std::span<const std::string_view> args) const;
/// }
/// @endcode
///
template <typename Implementation>
class JobBasedExecutable : private Implementation, public Benchmark {
    // Type Alises
  public:
    using job_type = typename Implementation::job_type;
    using dispatcher_type = typename Implementation::dispatcher_type;

  public:
    template <typename... Args>
    JobBasedExecutable(Args&&... args)
        : Implementation(SVS_FWD(args)...)
        , Benchmark() {}

    bool check_jobs(std::span<const job_type> jobs) const {
        auto f = Implementation::dispatcher();
        auto check = [&](auto&&... args) { return f.has_match(SVS_FWD(args)...); };

        size_t i = 0;
        auto tmp = Checkpoint();
        for (auto&& job : jobs) {
            if (!job.invoke(check, tmp)) {
                fmt::print("Unimplemented specialization for job number {}!\n", i);
                fmt::print("Contents of the job are given below:\n");
                fmt::print("{}\n\n", fmt::streamed(svs::lib::save_to_table(job)));
                return false;
            }
            ++i;
        }
        return true;
    }

    void print_example() const {
        auto arr = toml::array();
        arr.push_back(svs::lib::save_to_table(Implementation::example()));
        fmt::print("An example skeleton TOML file is provided below.\n\n");
        fmt::print(
            "{}\n", fmt::streamed(toml::table({{Implementation::name(), std::move(arr)}}))
        );
    }

    toml::table run_job(
        const dispatcher_type& dispatcher,
        const job_type& job,
        const svsbenchmark::Checkpoint& checkpoint
    ) const {
        return job.invoke(
            [&](auto&&... args) { return dispatcher.invoke(SVS_FWD(args)...); }, checkpoint
        );
    }

    int run_jobs(const std::filesystem::path& results_path, std::span<const job_type> jobs)
        const {
        // Now that we've finished parsing the jobs - we can check that we have ewverything
        // we need to complete the run.
        if (!check_jobs(jobs)) {
            return 1;
        }

        auto results = toml::table({{"start_time", svs::date_time()}});
        auto f = Implementation::dispatcher();

        for (auto&& job : jobs) {
            svsbenchmark::append_or_create(
                results,
                run_job(f, job, svsbenchmark::Checkpoint(results, results_path)),
                Implementation::name()
            );
            svsbenchmark::atomic_save(results, results_path);
        }

        // Save final results.
        results.emplace("stop_time", svs::date_time());
        atomic_save(results, results_path);
        return 0;
    }

    // Top level run-routine.
    int run(std::span<const std::string_view> args) const {
        auto nargs = args.size();
        if (nargs == 0) {
            Implementation::print_help();
            return 0;
        }

        if (std::any_of(args.begin(), args.end(), detail::is_help)) {
            Implementation::print_help();
            return 0;
        }
        if (std::any_of(args.begin(), args.end(), detail::is_example)) {
            print_example();
            return 0;
        }

        if (nargs < 2) {
            fmt::print("Expected at least two arguments. Instead, got {}.\n", nargs);
            Implementation::print_help();
            return 0;
        }

        // Done with error checking out-side of that needed by the lower-level
        // implementation.
        auto config_file = args[0];
        auto results_path = args[1];

        // Let the implementation parse the remainder of the arguments and then load the
        // requested job from the configuration file.
        auto jobs = Implementation::parse_args_and_invoke(
            [&](auto&&... args) {
                auto configuration = toml::parse_file(std::string(config_file));
                return svs::lib::load_at<std::vector<job_type>>(
                    configuration, Implementation::name(), SVS_FWD(args)...
                );
            },
            args.last(nargs - 2)
        );

        if (!jobs.has_value()) {
            return 1;
        }

        return run_jobs(results_path, svs::lib::as_const_span(jobs.value()));
    }

    ///// Benchmark API
    std::string do_name() const override { return std::string(Implementation::name()); }
    int do_run(std::span<const std::string_view> args) const override { return run(args); }
};

///
/// @brief An executable consisting of a single job passed to multiple routines.
///
/// Requirements for `Implementation`:
///
/// @code{cpp}
/// class Implementation {
///   public:
///     using job_type = /*implementation-defined*/;
///
///     // Return a dispatcher for the executable.
///     // Each test must take `const job_type&` as an argument and return a destructurable
///     // pair consiting of a string key and a `toml::table` of results.
///     std::vector</*implementation-defined*/> tests() const;
///
///     // Return an example job to serve as a prototype.
///     job_type example() const;
///
///     // The name assoicated with the job. This will be used to both pull jobs from
///     // the input TOML file as well as be the key under which results will be collected
///  T   // in output TOML file.
///     std::convertible_to<std::string> name() const;
///
///     // Print the help message.
///     void print_help() const;
///
///     // Parse any command-line arguments needed specifically by this job and invoke
///     // the callable.
///     //
///     // `args` will contain the remaining commandline arguments after:
///     //    - svs_benchmark
///     //    - executable_name
///     //    - job_source_file.toml
///     //    - job_destination_file.toml
///     // `f` is a partially appliec function. Any arguments passed to `f` will be
///     //   forwarded to `svs::load` for `job_type`.
///     //
///     // Returns a empty optional if argument parsing failed. The callee must supply
///     // a diagnostic print-out on failure.
///     template<typename F>
///     std::optional<job_type>
///     parse_args_and_invoke(F&& f, std::span<const std::string_view> args) const;
/// }
/// @endcode
///
template <typename Implementation>
class TestBasedExecutable : private Implementation, public Benchmark {
    // Type Alises
  public:
    using job_type = typename Implementation::job_type;

  public:
    template <typename... Args>
    TestBasedExecutable(Args&&... args)
        : Implementation(SVS_FWD(args)...)
        , Benchmark() {}

    void print_example() const {
        auto name = Implementation::name();
        auto example = svs::lib::save_to_table(Implementation::example());

        fmt::print("An example skeleton TOML file is provided below.\n\n");
        fmt::print("{}\n", fmt::streamed(toml::table({{name, example}})));
    }

    void run_jobs(const std::filesystem::path& results_path, const job_type& job) const {
        auto results = toml::table({{"start_time", svs::date_time()}});
        auto tests = Implementation::tests();

        for (auto&& test : tests) {
            auto [key, job_results] = test(job);
            svsbenchmark::append_or_create(results, std::move(job_results), key);
            svsbenchmark::atomic_save(results, results_path);
        }

        // Save final results.
        results.emplace("stop_time", svs::date_time());
        atomic_save(results, results_path);
    }

    // Top level run-routine.
    int run(std::span<const std::string_view> args) const {
        auto nargs = args.size();
        if (nargs == 0) {
            Implementation::print_help();
            return 0;
        }

        if (std::any_of(args.begin(), args.end(), detail::is_help)) {
            Implementation::print_help();
            return 0;
        }
        if (std::any_of(args.begin(), args.end(), detail::is_example)) {
            print_example();
            return 0;
        }

        if (nargs < 2) {
            fmt::print("Expected at least two arguments. Instead, got {}.\n", nargs);
            Implementation::print_help();
            return 0;
        }

        // Done with error checking out-side of that needed by the lower-level
        // implementation.
        auto config_file = args[0];
        auto results_path = args[1];

        // Let the implementation parse the remainder of the arguments and then load the
        // requested job from the configuration file.
        auto job = Implementation::parse_args_and_invoke(
            [&](auto&&... args) {
                auto configuration = toml::parse_file(std::string(config_file));
                return svs::lib::load_at<job_type>(
                    configuration, Implementation::name(), SVS_FWD(args)...
                );
            },
            args.last(nargs - 2)
        );

        // Parsing failed.
        if (!job.has_value()) {
            return 1;
        }

        // Parsing was successful - run the benchmarks.
        run_jobs(results_path, job.value());
        return 0;
    }

    ///// Benchmark API
    std::string do_name() const override { return std::string(Implementation::name()); }
    int do_run(std::span<const std::string_view> args) const override { return run(args); }
};

} // namespace svsbenchmark
