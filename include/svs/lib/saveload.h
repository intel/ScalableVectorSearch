/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// Utilities for saving and loading objects.
#pragma once

// file handling
#include "svs/lib/exception.h"
#include "svs/lib/file.h"

// third-party
#include "svs/third-party/fmt.h"
#include "svs/third-party/toml.h" // Careful not to introduce a circular header dependency

// stl
#include <atomic>
#include <charconv>
#include <concepts>
#include <filesystem>
#include <string_view>

namespace svs {
namespace lib {

///
/// @brief Parse the provided view as a base-10 integer.
///
/// @tparam T The integer type to parse.
///
/// @param view The object to parse.
///
/// @returns The parsed integer.
///
/// @throws svs::ANNException if anything goes wrong during parsing.
///
template <std::integral T> [[nodiscard]] T parse_int(std::string_view view) {
    auto value = T{0};
    auto result = std::from_chars(view.begin(), view.end(), value);
    if (result.ec != std::errc()) {
        throw ANNEXCEPTION("Something went wrong with number parsing!");
    }
    return value;
}

///
/// @brief A representation of the typical three-numbered version identifier.
///
/// The version numbers are expected to roughly follow semantic versioning.
///
/// * MAJOR versions are incremented when incompatible API changes are made.
/// * MINOR versions are incremented when functionality is added in a backward compatible
///     manner.
/// * PATCH versions are for backwards compatible bug fixes.
///
/// In general, no guarentees are made with a version number "v0.0.x".
/// Such items are experimental and should not be relied upon.
///
/// Version numbers "v0.x.y" represent actively changing APIs and should be used with care.
///
struct Version {
    /// @brief Return the formatted version as "vMAJOR.MINOR.PATCH".
    std::string str() const { return fmt::format("v{}.{}.{}", major, minor, patch); }

    /// @brief Construct a new Version class.
    constexpr explicit Version(size_t major, size_t minor, size_t patch)
        : major{major}
        , minor{minor}
        , patch{patch} {}

    ///
    /// @brief Construct a new Version class by parsing a formatted string.
    ///
    /// @throws svs::ANNException if the string is malformed.
    ///
    /// The string to be formatted should be *exactly* in the form "vMAJOR.MINOR.PATCH"
    /// where each of MAJOR, MINOR, and PATCH is a positive base-10 integer.
    ///
    explicit Version(std::string_view v) {
        auto npos = v.npos;
        if (!v.starts_with('v')) {
            throw ANNEXCEPTION("Formatted version string doesn't begin with a 'v'!");
        }

        auto mallformed = []() { throw ANNEXCEPTION("Malformed version!"); };

        size_t start = 1;
        size_t stop = v.find('.', start);
        if (stop == npos) {
            mallformed();
        }
        major = parse_int<size_t>(v.substr(start, stop - start));

        // Parse minor
        start = stop + 1;
        stop = v.find('.', start);
        if (stop == npos) {
            mallformed();
        }
        minor = parse_int<size_t>(v.substr(start, stop - start));
        // parse to the end of the string.
        patch = parse_int<size_t>(v.substr(stop + 1));
    }

    ///// Members
    size_t major;
    size_t minor;
    size_t patch;
};

///
/// @brief Compare two versions for equality.
///
/// Two versions are equal if all fields compare equal.
///
inline constexpr bool operator==(const Version& x, const Version& y) {
    return (x.major == y.major) && (x.minor == y.minor) && (x.patch == y.patch);
}

/// @brief Compare two versions for a "less than" relationship.
inline constexpr bool operator<(const Version& x, const Version& y) {
    // Compare major.
    if (x.major < y.major) {
        return true;
    } else if (x.major > y.major) {
        return false;
    }

    // Major is equal -- compare minor.
    if (x.minor < y.minor) {
        return true;
    } else if (x.minor > y.minor) {
        return false;
    }

    // Minor is equal -- compare patch.
    return x.patch < y.patch;
}

} // namespace lib

///
/// @brief Get the version from a TOML table.
///
inline lib::Version get_version(const toml::table& table, std::string_view key) {
    return lib::Version(get(table, key).value());
}

///
/// @brief Prepare a version for TOML serialization.
///
inline std::string prepare(const lib::Version& version) { return version.str(); }

inline void emplace(toml::table& table, std::string_view key, const lib::Version& version) {
    emplace(table, key, prepare(version));
}

namespace lib {

///
/// @brief Context used when saving aggregate objects.
///
constexpr Version CURRENT_SAVE_VERSION = Version(0, 0, 1);
class SaveContext {
  public:
    ///
    /// @brief Construct a new SaveContext in the current directory.
    ///
    /// @param directory The directory where the data structure will be saved.
    /// @param version The saving version (leave at the default).
    ///
    SaveContext(
        std::filesystem::path directory, const Version& version = CURRENT_SAVE_VERSION
    )
        : directory_{std::move(directory)}
        , version_{version} {}

    /// @brief Return the current directory where intermediate files will be saved.
    const std::filesystem::path& get_directory() const { return directory_; }
    /// @brief Generate a unique suffix for the directory.
    std::filesystem::path
    generate_name(std::string_view prefix, std::string_view extension = "svs") const {
        // Generate the relative file path first.
        auto fullpath = get_directory() / prefix;
        auto count = count_.fetch_add(1);
        return std::filesystem::path(
            fmt::format("{}_{}.{}", fullpath.c_str(), count, extension)
        );
    }

    const Version& version() const { return version_; }

    // Delete the special members to avoid the context getting lost or accidentally copied.
    SaveContext(const SaveContext&) = delete;
    SaveContext operator=(const SaveContext&) = delete;
    SaveContext(SaveContext&&) = delete;
    SaveContext operator=(SaveContext&&) = delete;

  private:
    /// The working directory for saving and loading.
    std::filesystem::path directory_;
    /// The current save version.
    Version version_;
    /// An atomic variable used to generate unique suffixes for file names.
    mutable std::atomic<size_t> count_ = 0;
};

///
/// @brief Context used when loading aggregate objects.
///
class LoadContext {
  public:
    LoadContext(std::filesystem::path directory, const Version& version)
        : directory_{std::move(directory)}
        , version_{version} {}

    /// @brief Return the current directory where intermediate files will be saved.
    const std::filesystem::path& get_directory() const { return directory_; }

    ///
    /// @brief Return the current global loading version scheme.
    ///
    /// Saving and loading should prefer to implement their own versioning instead of
    /// relying on the global version.
    ///
    const Version& version() const { return version_; }

  private:
    /// The working directory for saving and loading.
    std::filesystem::path directory_;
    Version version_;
};

/// @brief Expected type to be returned from `.save()` methods.
using SaveType = std::tuple<toml::table, Version>;

template <typename T>
concept Saveable = requires(T& x, const SaveContext& ctx) {
    { x.save(ctx) } -> std::same_as<SaveType>;
};

///
/// @brief Trait to determine if an object is save-able.
///
template <typename T> inline constexpr bool is_saveable = false;
template <Saveable T> inline constexpr bool is_saveable<T> = true;

// Loading may require a helper type for loading.
template <typename T>
concept IsSelfLoader =
    requires(const toml::table& table, const LoadContext& ctx, const Version& version) {
        { T::load(table, ctx, version) } -> std::convertible_to<T>;
    };

template <typename T, typename U>
concept IsLoadHelperFor = requires(
    const T& loader,
    const toml::table& table,
    const LoadContext& ctx,
    const Version& version
) {
    { loader.load(table, ctx, version) } -> std::same_as<U>;
};

template <IsSelfLoader T> struct LoaderFor {
    T load(const toml::table& table, const LoadContext& ctx, const Version& version) const {
        return T::load(table, ctx, version);
    }
};

namespace detail {
template <typename T> struct AsLoader {
    using type = T;
};
template <IsSelfLoader T> struct AsLoader<T> {
    using type = LoaderFor<T>;
};
} // namespace detail

template <typename T> using as_loader = typename detail::AsLoader<T>::type;

///
/// @brief Tag type to indicate that a file path should be inferred (if possible).
///
struct InferPath {
    explicit constexpr InferPath() = default;
};

/////
///// Loading and Saving entry points.
/////

inline constexpr std::string_view config_file_name = "svs_config.toml";
inline constexpr std::string_view config_version_key = "__version__";
inline constexpr std::string_view config_object_key = "object";

///
/// @brief Recursively save a class.
///
/// @param x The class to save.
/// @param ctx The current save context.
///
/// @returns A post processed table representation any data and metadata associated with
///     saving ``x``.
///
/// When saving member classes, use ``recursive_save`` rather than directly invoking the
/// member ``save`` method.
///
template <typename T> toml::table recursive_save(const T& x, const SaveContext& ctx) {
    SaveType val = x.save(ctx);
    emplace(std::get<0>(val), config_version_key, std::get<1>(val));
    return std::get<0>(std::move(val));
}

///
/// @brief Recursively load a member class using the specified loader.
///
/// @param loader The loader to use.
/// @param table The subtable that the original class returned when it was saved.
/// @param ctx The current loading context.
///
/// Call this method to reload a member class as part of loading a class.
///
template <typename Loader>
auto recursive_load(
    const Loader& loader, const toml::table& table, const LoadContext& ctx
) {
    auto version = get_version(table, config_version_key);
    return loader.load(table, ctx, version);
}

///
/// @brief Recursively load a SelfLoading member class.
///
/// @param table The subtable that the class returned when it was saved.
/// @param ctx The current loading context.
///
/// Call this method to reload a member class as part of loading a class when that member
/// class is self loading.
///
template <typename SelfLoader>
auto recursive_load(const toml::table& table, const LoadContext& ctx) {
    return recursive_load(LoaderFor<SelfLoader>(), table, ctx);
}

///
/// @brief Save the object into the given directory.
///
/// @param x The object to save to disk.
/// @param dir The directory where the object should be saved.
///
/// As part of saving the object, multiple auxiliary files may be created in the
/// directory. It is the caller's responsibility to ensure that no existing data in the
/// given directory will be destroyed.
///
/// On the other hand, if during the saving of the object, any files are generated
/// *outside* of this directory, that should be considered a bug. Please report such
/// instances to the project maintainer.
///
template <typename T> void save(T& x, const std::filesystem::path& dir) {
    // Create the directory.
    // Per the documented API, if `dir` already exists, then there is no error.
    // Otherwise, the parent directory must exist.
    std::filesystem::create_directory(dir);

    // Assume the saving context is in the same directory as the path.
    auto ctx = SaveContext(dir);
    toml::table table = recursive_save(x, ctx);
    auto top_table = toml::table(
        {{config_version_key, prepare(ctx.version())},
         {config_object_key, std::move(table)}}
    );
    auto file = open_write(dir / config_file_name, std::ios_base::out);
    file << top_table;
}

///
/// Load Overrider
///
template <typename F> class LoadOverride {
  public:
    LoadOverride(F&& f)
        : f_{std::move(f)} {}

    auto load(
        const toml::table& table, const lib::LoadContext& ctx, const lib::Version& version
    ) const {
        return f_(table, ctx, version);
    }

  private:
    F f_;
};

///
/// @brief Top level object reloading function.
///
/// @sa load(const Loader&, const std::filesystem::path&)
///
template <typename T> auto load(const std::filesystem::path& path) {
    return load(LoaderFor<T>(), path);
}

template <typename Loader> auto load(const Loader& loader, std::filesystem::path path) {
    // Detect if we have a path to a file or directory.
    // If it's a directory, then assume we should append the canonical config file.
    if (std::filesystem::is_directory(path)) {
        path /= config_file_name;
    }
    auto table = toml::parse_file(path.c_str());
    auto version = get_version(table, config_version_key);
    auto ctx = LoadContext(path.parent_path(), version);
    return recursive_load(loader, subtable(table, config_object_key), ctx);
}

template <typename T> bool test_self_save_load(T& x, const std::filesystem::path& dir) {
    save(x, dir);
    T y = load<T>(dir);
    return x == y;
}

} // namespace lib
} // namespace svs
