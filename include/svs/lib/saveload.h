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
#include "svs/lib/version.h"

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
/// VERSION HISTORY
///
/// v0.0.0 - Original version.
/// v0.0.1 - Unknown change.
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
    template <typename... Args>
    T load(
        const toml::table& table,
        const LoadContext& ctx,
        const Version& version,
        Args&&... args
    ) const {
        return T::load(table, ctx, version, std::forward<Args>(args)...);
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

inline toml::table emplace_version(SaveType val) {
    emplace(std::get<0>(val), config_version_key, std::get<1>(val));
    return std::get<0>(std::move(val));
}

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
    return emplace_version(x.save(ctx));
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
template <typename Loader, typename... Args>
auto recursive_load(
    const Loader& loader, const toml::table& table, const LoadContext& ctx, Args&&... args
) {
    auto version = get_version(table, config_version_key);
    return loader.load(table, ctx, version, std::forward<Args>(args)...);
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
template <typename SelfLoader, typename... Args>
auto recursive_load(const toml::table& table, const LoadContext& ctx, Args&&... args) {
    return recursive_load(LoaderFor<SelfLoader>(), table, ctx, std::forward<Args>(args)...);
}

///
/// @brief Save the object into the given directory.
///
/// @param dir The directory where the object should be saved.
/// @param f A callable object that takes a ``const svs::lib::SaveContext&` and returns
///     a ``toml::table``.
///
/// As part of saving the object, multiple auxiliary files may be created in the
/// directory. It is the caller's responsibility to ensure that no existing data in the
/// given directory will be destroyed.
///
/// On the other hand, if during the saving of the object, any files are generated
/// *outside* of this directory, that should be considered a bug. Please report such
/// instances to the project maintainer.
///
template <typename F> void save_callable(const std::filesystem::path& dir, F&& f) {
    // Create the directory.
    // Per the documented API, if `dir` already exists, then there is no error.
    // Otherwise, the parent directory must exist.
    std::filesystem::create_directory(dir);

    // Assume the saving context is in the same directory as the path.
    auto ctx = SaveContext(dir);
    toml::table table = emplace_version(f(ctx));
    auto top_table = toml::table(
        {{config_version_key, prepare(ctx.version())},
         {config_object_key, std::move(table)}}
    );
    auto file = open_write(dir / config_file_name, std::ios_base::out);
    file << top_table;
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
    save_callable(dir, [&](const lib::SaveContext& ctx) { return x.save(ctx); });
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
