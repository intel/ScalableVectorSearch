// svs-benchmar
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/benchmark.h"

// svs
#include "svs/core/recall.h"
#include "svs/extensions/vamana/lvq.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/saveload.h"
#include "svs/misc/dynamic_helper.h"
#include "svs/orchestrators/dynamic_vamana.h"
#include "svs/orchestrators/vamana.h"

// third-party
#include "fmt/core.h"
#include "fmt/ranges.h"

// stl
#include <algorithm>
#include <filesystem>
#include <functional>
#include <optional>

namespace svsbenchmark::vamana {
namespace {

template <typename F> void standard_specializations(F&& f) {
#define X(Q, T, D, N) \
    f(svs::meta::Type<Q>(), svs::meta::Type<T>(), D(), svs::meta::Val<N>())
    if constexpr (!is_minimal) {
        X(uint8_t, uint8_t, svs::distance::DistanceL2, 128);    // bigann
        X(float, svs::Float16, svs::distance::DistanceL2, 96);  // deep
        X(float, svs::Float16, svs::distance::DistanceL2, 100); // msturing
        X(int8_t, int8_t, svs::distance::DistanceL2, 100);      // spacev
        X(float, svs::Float16, svs::distance::DistanceIP, 200); // text2image
        X(float, svs::Float16, svs::distance::DistanceIP, 768); // dpr
        // X(float, svs::Float16, svs::distance::DistanceIP, 768); // dpr
        // Generic fallbacks
        X(float, float, svs::distance::DistanceL2, svs::Dynamic);
        X(float, svs::Float16, svs::distance::DistanceL2, svs::Dynamic);
    }
#undef X
}

template <typename F> void lvq_specializations(F&& f) {
#define X(Q, T, D, N) \
    f(svs::meta::Type<Q>(), svs::meta::Type<T>(), D(), svs::meta::Val<N>())
    if constexpr (!is_minimal) {
        // X(float, svs::Float16, svs::distance::DistanceL2, 96);
        X(float, svs::Float16, svs::distance::DistanceIP, 768);
    }
#undef X
}

template <typename BenchmarkType, typename T> struct BuildDispatcher;

template <typename BenchmarkType, typename T>
using build_dispatch_t = typename BuildDispatcher<BenchmarkType, T>::type;

struct StaticUncompressed {
    // Dispatch Parameters:
    // (1) Query Type.
    // (2) Uncompressed Dataset Element Type.
    // (3) Distance Type.
    // (4) Dimensionality (svs::Dynamic if dynamic)
    using key_type = std::tuple<svs::DataType, svs::DataType, svs::DistanceType, size_t>;
    using mapped_type = std::function<toml::table(const BuildJob&, const Checkpoint&)>;

    // Specialize index building for run-time values.
    // * Query Type: The element types of each component of the query vectors.
    // * Data Type: The element types of each component of the query vectors.
    // * Distance Type: The distance functor to use.
    // * Data Dimensionality: The number of elements in each vector.
    template <typename Q, typename T, typename D, size_t N>
    static std::pair<key_type, mapped_type> specialize(
        svs::meta::Type<Q> query_type,
        svs::meta::Type<T> data_type,
        D distance,
        svs::meta::Val<N> ndims
    ) {
        key_type key = svs::lib::meta::make_key(query_type, data_type, distance, ndims);
        mapped_type value = [=](const BuildJob& job, const Checkpoint& checkpointer) {
            auto tic = svs::lib::now();
            auto index = svs::Vamana::build<Q>(
                job.build_parameters_,
                svs::data::SimpleData<T, N, svs::HugepageAllocator<T>>::load(job.data_),
                distance,
                job.num_threads_,
                svs::HugepageAllocator<uint32_t>()
            );

            double build_time = svs::lib::time_difference(tic);
            auto queries = svs::data::SimpleData<Q>::load(job.queries_);
            auto groundtruth = svs::data::SimpleData<uint32_t>::load(job.groundtruth_);
            return svsbenchmark::build::static_test(
                index,
                job,
                queries,
                groundtruth,
                build_time,
                [&](const toml::table& data) {
                    checkpointer.checkpoint(data, benchmark_name(StaticBenchmark()));
                }
            );
        };
        return std::make_pair(key, std::move(value));
    }

    template <typename F> static void fill(F&& f) {
        standard_specializations([&](auto... xs) {
            f(StaticUncompressed::specialize(xs...));
        });
    }
};

struct DynamicUncompressed {
    // Dispatch Parameters:
    // (1) Query Type.
    // (2) Uncompressed Dataset Element Type.
    // (3) Distance Type.
    // (4) Dimensionality (svs::Dynamic if dynamic)
    using key_type = std::tuple<svs::DataType, svs::DataType, svs::DistanceType, size_t>;
    using mapped_type =
        std::function<toml::table(const DynamicBuildJob&, const Checkpoint&)>;

    // Specialize index building for run-time values.
    // * Query Type: The element types of each component of the query vectors.
    // * Data Type: The element types of each component of the query vectors.
    // * Distance Type: The distance functor to use.
    // * Data Dimensionality: The number of elements in each vector.
    template <typename Q, typename T, typename D, size_t N>
    static std::pair<key_type, mapped_type> specialize(
        svs::meta::Type<Q> query_type,
        svs::meta::Type<T> data_type,
        D distance,
        svs::meta::Val<N> ndims
    ) {
        key_type key = svs::lib::meta::make_key(query_type, data_type, distance, ndims);
        mapped_type value = [=](const DynamicBuildJob& job,
                                const Checkpoint& checkpointer) {
            auto bundle = svsbenchmark::build::initialize_dynamic<T, Q>(
                job.data_,
                job.queries_,
                distance,
                job.get_dynamic_schedule(),
                job.num_threads_,
                [&](const auto& points, const auto& ids) {
                    using A = svs::HugepageAllocator<T>;
                    auto data_mutable =
                        svs::data::BlockedData<T, N, A>(points.size(), points.dimensions());
                    svs::data::copy(points, data_mutable);
                    return svs::index::vamana::MutableVamanaIndex(
                        job.build_parameters_,
                        std::move(data_mutable),
                        ids,
                        distance,
                        job.num_threads_
                    );
                }
            );

            return svsbenchmark::build::dynamic_test_loop(
                bundle,
                job,
                [&](const toml::table& table) {
                    checkpointer.checkpoint(table, benchmark_name(DynamicBenchmark()));
                }
            );
        };
        return std::make_pair(key, std::move(value));
    }

    template <typename F> static void fill(F&& f) {
        standard_specializations([&](auto... xs) {
            f(DynamicUncompressed::specialize(xs...));
        });
    }
};

template <size_t Primary, size_t Residual> struct StaticLVQ {
    // Dispatch Parameters:
    // (1) Query Type.
    // (2) Uncompressed Dataset Element Type.
    // (3) Distance Type.
    // (4) Dimensionality (svs::Dynamic if dynamic)
    using key_type = std::tuple<svs::DataType, svs::DataType, svs::DistanceType, size_t>;
    using mapped_type = std::function<toml::table(const BuildJob& job, const Checkpoint&)>;

    // Specialize index building for run-time values.
    // * Query Type: The element types of each component of the query vectors.
    // * Data Type: The element types of each component of the query vectors.
    // * Distance Type: The distance functor to use.
    // * Data Dimensionality: The number of elements in each vector.
    template <typename Q, typename T, typename D, size_t N>
    static std::pair<key_type, mapped_type> specialize(
        svs::meta::Type<Q> query_type,
        svs::meta::Type<T> data_type,
        D distance,
        svs::meta::Val<N> ndims
    ) {
        key_type key = svs::lib::meta::make_key(query_type, data_type, distance, ndims);
        mapped_type value = [=](const BuildJob& job, const Checkpoint& checkpointer) {
            namespace lvq = svs::quantization::lvq;
            using A = svs::HugepageAllocator<std::byte>;
            using LVQType = lvq::LVQDataset<Primary, Residual, N, A>;

            auto tic = svs::lib::now();
            auto lazy = svs::lib::Lazy([&](svs::threads::ThreadPool auto& threadpool) {
                auto data = svs::data::SimpleData<T, N>::load(job.data_);
                return LVQType::compress(data, threadpool, 32);
            });

            auto index = svs::Vamana::build<Q>(
                job.build_parameters_, lazy, distance, job.num_threads_
            );
            double build_time = svs::lib::time_difference(tic);
            auto queries = svs::data::SimpleData<Q>::load(job.queries_);
            auto groundtruth = svs::data::SimpleData<uint32_t>::load(job.groundtruth_);
            return svsbenchmark::build::static_test(
                index,
                job,
                queries,
                groundtruth,
                build_time,
                [&](const toml::table& data) {
                    checkpointer.checkpoint(data, benchmark_name(StaticBenchmark()));
                }
            );
        };
        return std::make_pair(key, std::move(value));
    }

    template <typename F> static void fill(F&& f) {
        lvq_specializations([&](auto... xs) { f(StaticLVQ::specialize(xs...)); });
    }
};

template <size_t Primary, size_t Residual> struct DynamicLVQ {
    // Dispatch Parameters:
    // (1) Query Type.
    // (2) Uncompressed Dataset Element Type.
    // (3) Distance Type.
    // (4) Dimensionality (svs::Dynamic if dynamic)
    using key_type = std::tuple<svs::DataType, svs::DataType, svs::DistanceType, size_t>;
    using mapped_type =
        std::function<toml::table(const DynamicBuildJob&, const Checkpoint&)>;

    // Specialize index building for run-time values.
    // * Query Type: The element types of each component of the query vectors.
    // * Data Type: The element types of each component of the query vectors.
    // * Distance Type: The distance functor to use.
    // * Data Dimensionality: The number of elements in each vector.
    template <typename Q, typename T, typename D, size_t N>
    static std::pair<key_type, mapped_type> specialize(
        svs::meta::Type<Q> query_type,
        svs::meta::Type<T> data_type,
        D distance,
        svs::meta::Val<N> ndims
    ) {
        key_type key = svs::lib::meta::make_key(query_type, data_type, distance, ndims);
        mapped_type value = [=](const DynamicBuildJob& job,
                                const Checkpoint& checkpointer) {
            namespace lvq = svs::quantization::lvq;
            using A = svs::data::Blocked<svs::HugepageAllocator<std::byte>>;
            using LVQType = lvq::LVQDataset<Primary, Residual, N, A>;

            auto bundle = svsbenchmark::build::initialize_dynamic<T, Q>(
                job.data_,
                job.queries_,
                distance,
                job.get_dynamic_schedule(),
                job.num_threads_,
                [&](const auto& points, const auto& ids) {
                    return svs::index::vamana::MutableVamanaIndex(
                        job.build_parameters_,
                        LVQType::compress(points, job.num_threads_, 32),
                        ids,
                        distance,
                        job.num_threads_
                    );
                }
            );

            return svsbenchmark::build::dynamic_test_loop(
                bundle,
                job,
                [&](const toml::table& table) {
                    checkpointer.checkpoint(table, benchmark_name(DynamicBenchmark()));
                }
            );
        };
        return std::make_pair(key, std::move(value));
    }

    template <typename F> static void fill(F&& f) {
        lvq_specializations([&](auto... xs) { f(DynamicLVQ::specialize(xs...)); });
    }
};

// Dispatcher for Uncompressed Data.
template <> struct BuildDispatcher<StaticBenchmark, Uncompressed> {
    using type = svs::lib::Dispatcher<StaticUncompressed>;
};

template <> struct BuildDispatcher<DynamicBenchmark, Uncompressed> {
    using type = svs::lib::Dispatcher<DynamicUncompressed>;
};

template <size_t Primary, size_t Residual>
struct BuildDispatcher<StaticBenchmark, LVQ<Primary, Residual>> {
    using type = svs::lib::Dispatcher<StaticLVQ<Primary, Residual>>;
};

template <size_t Primary, size_t Residual>
struct BuildDispatcher<DynamicBenchmark, LVQ<Primary, Residual>> {
    using type = svs::lib::Dispatcher<DynamicLVQ<Primary, Residual>>;
};

// Types used for dispatch purposes.
template <typename BenchmarkType> struct BuildTypes;

template <> struct BuildTypes<StaticBenchmark> {
    static constexpr auto types =
        svs::lib::meta::Types<Uncompressed, LVQ<8>, LVQ<4, 8>, LVQ<8, 8>>();
};

template <> struct BuildTypes<DynamicBenchmark> {
    static constexpr auto types =
        svs::lib::meta::Types<Uncompressed, LVQ<8>, LVQ<4, 8>, LVQ<8, 8>>();
};

template <typename T> inline constexpr auto build_types = BuildTypes<T>::types;

template <typename BenchmarkType>
bool check_job(BenchmarkType, const associated_job_t<BenchmarkType>& job) {
    return parse_dispatch(build_types<BenchmarkType>, job.build_type_, [&](auto&& tag) {
        using tag_type = std::decay_t<decltype(tag)>;
        using Dispatcher = build_dispatch_t<BenchmarkType, tag_type>;
        return Dispatcher::contains(
            false, job.ndims_, job.query_type_, job.data_type_, job.distance_
        );
    });
}

template <typename BenchmarkType>
toml::table run_job(
    BenchmarkType,
    const associated_job_t<BenchmarkType>& job,
    const Checkpoint& checkpointer
) {
    return parse_dispatch(build_types<BenchmarkType>, job.build_type_, [&](auto&& tag) {
        using Dispatcher = build_dispatch_t<BenchmarkType, std::decay_t<decltype(tag)>>;
        const auto& f = Dispatcher::lookup(
            false, job.ndims_, job.query_type_, job.data_type_, job.distance_
        );
        return f(job, checkpointer);
    });
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

template <typename BenchmarkType> void print_help() {
    bool first = true;
    svs::lib::meta::for_each_type(
        build_types<BenchmarkType>,
        [&]<typename T>(svs::lib::meta::Type<T>) {
            auto keys = build_dispatch_t<BenchmarkType, T>::keys();
            if (!first) {
                fmt::print("\n");
            }
            first = false;
            fmt::print("Compiled specializations for {} data:\n", T::name());
            fmt::print("{{ query_type, data_type, distance, dimensionality }}\n");
            for (const auto& key : keys) {
                fmt::print("{{ {} }}\n", fmt::join(key, ", "));
            }
        }
    );
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
    } else if (first_arg == "--example") {
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
