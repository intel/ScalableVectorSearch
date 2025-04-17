/*
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// svs
#include "svs/core/data.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/file.h"
#include "svs/lib/saveload.h"
#include "svs/third-party/toml.h"

// third-party
#include "fmt/core.h"

// stl
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace svsbenchmark {

// Trait to determine if we're in a minimal build environment.
SVS_VALIDATE_BOOL_ENV(SVS_BENCHMARK_MINIMAL)
#if SVS_BENCHMARK_MINIMAL
inline constexpr bool is_minimal = true;
#else
inline constexpr bool is_minimal = false;
#endif

SVS_VALIDATE_BOOL_ENV(SVS_BENCHMARK_BUILD_TEST_GENERATORS)
#if SVS_BENCHMARK_BUILD_TEST_GENERATORS
inline constexpr bool build_test_generators = true;
// The macro "SVS_BENCHMARK_FOR_TESTS_ONLY" allows internally defined helper functions to
// not be emitted in the final binary if they are not used.
//
// By marking these translation-unit defined functions as "inline", clang won't complain
// with `-Wunneeded-internal-declaration`.
#define SVS_BENCHMARK_FOR_TESTS_ONLY
#else
#define SVS_BENCHMARK_FOR_TESTS_ONLY [[maybe_unused]]
inline constexpr bool build_test_generators = false;
#endif

SVS_VALIDATE_BOOL_ENV(SVS_BENCHMARK_VAMANA_SUPERSEARCH)
#if SVS_BENCHMARK_VAMANA_SUPERSEARCH
inline constexpr bool vamana_supersearch = true;
#else
inline constexpr bool vamana_supersearch = false;
#endif

// Serialize the TOML table to a file in a way that either succeeds in overwriting an
// existing file at the path `path` or completely fails.
//
// Torn writes will be avoided.
void atomic_save(const toml::table& table, const std::filesystem::path& path);

// Mutate `table` by appending `data` to an array stored at `table[key]`.
// Create such an array if `table[key]` does not exist.
//
// Throws an `svs::ANNException` if the node at `table[key]` is not a `toml::array`.
void append_or_create(toml::table& table, const toml::table& data, std::string_view key);

// Extract a file path from the given TOML table with an optional root to prepend.
// Checks if the file exists or not.
//
// If the file does not exist, throw an ANNException with a descriptive error message.
//
// If the filepath is not a relative path, the optional root is not prepended but the
// existence of the file is still checked.
std::filesystem::path extract_filename(
    const svs::lib::ContextFreeLoadTable& table,
    std::string_view key,
    const std::optional<std::filesystem::path>& root
);

// A utility class that will check the uniqueness and validity of a save directory.
class SaveDirectoryChecker {
  public:
    SaveDirectoryChecker() = default;

    // Extract the entry in `table[key]` as a `std::filesystem::path`.
    // If the resulting path is empty, return an empty optional.
    //
    // If the path is not empty, ensure that the given path has not been previously
    // registered with the checker (uniqueness) and that the parent of the path exists and
    // is as directory.
    std::optional<std::filesystem::path>
    extract(const toml::table& table, std::string_view key);

    // Make the copy constructor and move assignment operators private to avoid
    // accidentally copying when we meant to take by reference.
  private:
    SaveDirectoryChecker(const SaveDirectoryChecker&) = default;
    SaveDirectoryChecker& operator=(const SaveDirectoryChecker&) = default;

  public:
    SaveDirectoryChecker(SaveDirectoryChecker&&) = default;
    SaveDirectoryChecker& operator=(SaveDirectoryChecker&&) = default;
    ~SaveDirectoryChecker() = default;

  private:
    #if defined(__APPLE__)
        // Custom hash for macOS
        struct PathHash {
            std::size_t operator()(const std::filesystem::path& p) const {
               return std::hash<std::string>{}(p.string());
            }
        };
        using PathSet = std::unordered_set<std::filesystem::path, PathHash>;
    #else
        using PathSet = std::unordered_set<std::filesystem::path>;
    #endif  // __APPLE__

    PathSet directories_{};
};

// Place-holder to indicate no extra arguments need forwarding to inner calls.
struct Placeholder {};
inline constexpr Placeholder placeholder{};

// Context for search operations.
// Can be used to selectively optimize various tuning parameters.
enum class CalibrateContext {
    // Initial calibration on the training set of queries.
    InitialTrainingSet,
    // Tune-up calibration on the training set of queries.
    TrainingSetTune,
    // Tune-up calibration on the test set of queries.
    // Any tuning should *not* measure *performance* of knobs turned to achieve the
    // desired recall - only performance agnostic for accuracy tuning is allowed.
    TestSetTune
};

// Unified polymorphic type for running benchmarks.
class Benchmark {
  public:
    Benchmark() = default;

    Benchmark(const Benchmark&) = delete;
    Benchmark(Benchmark&&) = delete;

    Benchmark& operator=(const Benchmark&) = delete;
    Benchmark& operator=(Benchmark&&) = delete;

    std::string name() const { return do_name(); }
    int run(std::span<const std::string_view> args) const { return do_run(args); }

    virtual ~Benchmark() = default;

  protected:
    // Note for implementers: The name passed by `do_name()` will be used by the main
    // executable to dispatch to the backend benchmark. It should be unique and not
    // contain spaces.
    virtual std::string do_name() const = 0;

    // The arguments given will be all the command-line arguments minus the first two:
    // Argument 0 is the executable name and not needed.
    // Argument 1 is used to dispatch to the appropriate backend.
    //
    // All the test are forwarded unaltered.
    virtual int do_run(std::span<const std::string_view>) const = 0;
};

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
                "An executable with the name \"{}\" is already registered!", name
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

// In general, index builds can take a long time and it may be beneficial to two things:
//
// (1) Regularly save checkpoints of results as they are generated so that if the
// application fails, we do not lose all of our data.
// (2) Provide results in as near real-time as we can so we can monitor currently running
//     processes to determine as early as possible if something has gone wrong.
//
// The Checkpoint class keeps a record of the current results generated so far, appends
// new results as they become available and regularly saves results.
//
// It *does* involve many copies of the underlying TOML data, but I believe the tradoff in
// data safety greatly outweights any extra time spent moving around TOML data.
class Checkpoint {
  private:
    toml::table data_{};
    std::optional<std::filesystem::path> filename_{std::nullopt};

  public:
    Checkpoint() = default;
    Checkpoint(const toml::table& data, const std::filesystem::path& filename)
        : data_{data}
        , filename_{filename} {}

    void checkpoint(const toml::table& new_data, std::string_view key) const {
        if (!filename_.has_value()) {
            return;
        }

        // Make a copy of our current checkpointed data and try to append the new data
        // to the list.
        //
        // Make sure to handle the case where this is the first data being registered
        // with the given key.
        //
        // This is not the most efficient implemenation because we make unnecessary copies,
        // but the complexitry required to correctly applying incremental data does not
        // seem to be worth it.
        auto data_copy = data_;
        append_or_create(data_copy, new_data, key);
        atomic_save(data_copy, filename_.value());
    }
};

/// Helper types to describe "extent"
struct Extent {
  public:
    size_t value_;

  public:
    explicit Extent(size_t value)
        : value_{value} {}
    operator size_t() const { return value_; }
};

/// Tag type for Dispatch Conversion.
template <typename T> struct DispatchType {};

/// Class shared by multiple indexes to record index construction time in the result
/// toml file.
struct BuildTime {
  public:
    double build_time_;

  public:
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_build_time";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema, save_version, {SVS_LIST_SAVE_(build_time)}
        );
    }
};

/// Class shared to record the time taken to load an index into a useable form.
struct LoadTime {
  public:
    double load_time_;

  public:
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "benchmark_load_time";
    svs::lib::SaveTable save() const {
        return svs::lib::SaveTable(
            serialization_schema, save_version, {SVS_LIST_SAVE_(load_time)}
        );
    }
};

/////
///// Algorithms
/////

template <typename T, typename Alloc>
void append_to(std::vector<T, Alloc>& dst, std::vector<T, Alloc>&& src) {
    dst.insert(
        dst.end(), std::make_move_iterator(src.begin()), std::make_move_iterator(src.end())
    );
}

} // namespace svsbenchmark

/////
///// Dispatching
/////

template <size_t N>
struct svs::lib::DispatchConverter<svsbenchmark::Extent, svs::lib::ExtentTag<N>> {
    static bool match(svsbenchmark::Extent dim) {
        // For the benchmarking framework - we ensure exact matches.
        return dim == N;
    }
    static svs::lib::ExtentTag<N> convert(svsbenchmark::Extent) {
        return svs::lib::ExtentTag<N>();
    }

    static std::string description() {
        if constexpr (N == svs::Dynamic) {
            return "dynamic";
        } else {
            return fmt::format("{}", N);
        }
    }
};

template <typename T>
struct svs::lib::DispatchConverter<svs::DataType, svsbenchmark::DispatchType<T>> {
    static constexpr bool match(svs::DataType type) { return type == svs::datatype_v<T>; }

    static constexpr svsbenchmark::DispatchType<T>
    convert([[maybe_unused]] svs::DataType type) {
        assert(match(type));
        return svsbenchmark::DispatchType<T>{};
    }
    static std::string description() { return fmt::format("{}", svs::datatype_v<T>); }
};

template <> struct svs::lib::Saver<svsbenchmark::Extent> {
    static SaveNode save(svsbenchmark::Extent x) {
        if (x.value_ == Dynamic) {
            return SaveNode("dynamic");
        } else {
            return SaveNode(narrow<int64_t>(x.value_));
        }
    }
};

template <> struct svs::lib::Loader<svsbenchmark::Extent> {
    using toml_type = toml::node;
    static svsbenchmark::Extent load(svs::lib::ContextFreeNodeView<toml_type> view) {
        const auto& node = view.unwrap();
        if (const auto* v = node.as<std::string>(); v != nullptr) {
            const std::string& str = v->get();
            if (str == "dynamic") {
                return svsbenchmark::Extent(Dynamic);
            }
            throw ANNEXCEPTION(
                "Unrecognized string {} when trying to load extent from {}!",
                str,
                fmt::streamed(node.source())
            );
        }

        // Try to get as an integer and fail hard if that doesn't work.
        auto u = toml_helper::get_as<int64_t>(node);
        return svsbenchmark::Extent(u == -1 ? Dynamic : narrow<size_t>(u));
    }
};
