#pragma once

// svs
#include "svs/core/data.h"
#include "svs/lib/file.h"
#include "svs/lib/saveload.h"
#include "svs/quantization/lvq/lvq.h"
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
#if defined(SVS_BENCHMARK_MINIMAL)
inline constexpr bool is_minimal = true;
#else
inline constexpr bool is_minimal = false;
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

// Extract a file path from the given TOML table with an optional root to append.
// Checks if the file exists or not.
//
// If the file does not exist, throw an ANNException with a descriptive error message.
std::filesystem::path extract_filename(
    const toml::table& table,
    std::string_view key,
    const std::optional<std::filesystem::path>& root
);

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

// In general, index builds can take a long time and it may be beneficial to two things:
//
// (1) Regularly save checkpoints of results as they are generated so that if the
// application
//     fails, we do not lose all of our data.
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
    toml::table data_;
    std::filesystem::path filename_;

  public:
    Checkpoint(const toml::table& data, const std::filesystem::path& filename)
        : data_{data}
        , filename_{filename} {}

    void checkpoint(const toml::table& new_data, std::string_view key) const {
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
        atomic_save(data_copy, filename_);
    }
};

/// Regular old uncompressed data.
struct Uncompressed {
    // Sadly, we can't have computed constexpr string names yet :(
    constexpr static std::string_view name() { return "uncompressed"; }
};

/// LVQ compression.
/// Setting `Residual = 0` implies one-level LVQ.
template <size_t Primary, size_t Residual = 0> struct LVQ {
    static std::string name() {
        if constexpr (Residual == 0) {
            return fmt::format("lvq{}", Primary);
        } else {
            return fmt::format("lvq{}x{}", Primary, Residual);
        }
    }
};

///
/// Take a collection of dispatch tag types. Requires the following of each type:
///
/// (1) Types are default constructible.
/// (2) Types have a static `name()` method returing a `std::string` or `std::string_view`.
///
/// Iterates through the list of types trying to match the `name` argument to the types
/// static `name()` result. If a match is found, call the callable `f` with a default
/// constructed instance of the matching type.
///
/// Throws ANNException if no match is found.
///
template <typename F, typename T, typename... Ts>
auto parse_dispatch(svs::lib::meta::Types<T, Ts...>, std::string_view name, F&& f) {
    if (name == T::name()) {
        return f(T());
    }

    if constexpr (sizeof...(Ts) == 0) {
        throw ANNEXCEPTION("No dispatch type matching name {}", name);
    } else {
        return parse_dispatch(svs::lib::meta::Types<Ts...>(), name, std::forward<F>(f));
    }
}

///
/// Helper types to describe "extent"
///
struct Extent {
  public:
    size_t value_;

  public:
    explicit Extent(size_t value)
        : value_{value} {}
    operator size_t() const { return value_; }
};

} // namespace svsbenchmark

namespace svs::lib {
template <> struct Saver<svsbenchmark::Extent> {
    static SaveNode save(svsbenchmark::Extent x) {
        if (x.value_ == Dynamic) {
            return SaveNode("dynamic");
        } else {
            return SaveNode(narrow<int64_t>(x.value_));
        }
    }
};

template <> struct Loader<svsbenchmark::Extent> {
    using toml_type = toml::node;
    static constexpr bool is_version_free = true;
    static svsbenchmark::Extent load(const toml_type& node) {
        if (auto* v = node.as<std::string>(); v != nullptr) {
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
} // namespace svs::lib
