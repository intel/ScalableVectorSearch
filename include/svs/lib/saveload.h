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
#include "svs/lib/datatype.h"
#include "svs/lib/exception.h"
#include "svs/lib/file.h"
#include "svs/lib/misc.h"
#include "svs/lib/readwrite.h"
#include "svs/lib/uuid.h"
#include "svs/lib/version.h"

// third-party
#include "fmt/std.h"
#include "svs/third-party/fmt.h"
#include "svs/third-party/toml.h" // Careful not to introduce a circular header dependency

// stl
#include <atomic>
#include <charconv>
#include <concepts>
#include <filesystem>
#include <optional>
#include <string_view>

namespace svs {

inline bool config_file_by_extension(const std::filesystem::path& path) {
    return path.extension() == ".toml";
}

namespace lib {

// Reserved keyword for version strings in toml tables.
inline constexpr std::string_view config_version_key = "__version__";
inline constexpr std::string_view config_file_name = "svs_config.toml";
inline constexpr std::string_view config_object_key = "object";

inline lib::Version get_version(const toml::table& table, std::string_view key) {
    return lib::Version(toml_helper::get_as<toml::value<std::string>>(table, key).get());
}

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
    explicit SaveContext(
        std::filesystem::path directory, const Version& version = CURRENT_SAVE_VERSION
    )
        : directory_{std::move(directory)}
        , version_{version} {}

    /// @brief Return the current directory where intermediate files will be saved.
    const std::filesystem::path& get_directory() const { return directory_; }

    ///
    /// @brief Generate a unique filename in the saving directory.
    ///
    /// @param prefix An identifiable prefix for the file.
    /// @param extension The desired file extension.
    ///
    /// Note that the returned ``std::filesystem::path`` is an absolute path to the saving
    /// directory and as such, should not be stored directly in any configuration table
    /// in order for the resulting saved object to be relocatable.
    ///
    /// Instead, use the `.filepath()` member function to obtain a relative path to the
    /// saving directory.
    ///
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
    explicit LoadContext(std::filesystem::path directory, const Version& version)
        : directory_{std::move(directory)}
        , version_{version} {}

    /// @brief Return the current directory where intermediate files will be saved.
    const std::filesystem::path& get_directory() const { return directory_; }

    /// @brief Return the given relative path as a full path in the loading directory.
    std::filesystem::path resolve(const std::filesystem::path& relative) const {
        return get_directory() / relative;
    }

    /// @brief Return the relative path in `table` at position `key` as a full path.
    std::filesystem::path resolve(const toml::table& table, std::string_view key) const;

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

// Unversioned internal saving.
// This is meant for use of internal types and shouldn't be used for general data
// structures.
struct SaveNode {
    std::unique_ptr<toml::node> node_;

    // NOTE: This relies on patched behavior in the TOML library to allow `make_node` to
    // properly pass through a already-created `unique_ptr<toml::node>`.
    template <typename T>
    SaveNode(T&& val)
        : node_{toml::impl::make_node(std::forward<T>(val))} {}

    // Swipe the contents.
    const std::unique_ptr<toml::node>& get() const& { return node_; }
    std::unique_ptr<toml::node>&& get() && { return std::move(node_); }
};

///
/// @brief Versioned table use when saving classes.
///
class SaveTable {
  private:
    toml::table table_{};

    void insert_version(const Version& version) {
        table_.insert(config_version_key, version.str());
    }

  public:
    /// @brief Construct an empty table with the given version.
    explicit SaveTable(const Version& version) { insert_version(version); }

    ///
    /// @brief Construct a table using an initializer list of key-value pairs.
    ///
    /// Generally, values of the key-value pairs should be the return values from further
    /// calls to ``svs::lib::save()``.
    ///
    explicit SaveTable(
        const Version& version, std::initializer_list<toml::impl::table_init_pair> init
    )
        : table_{init} {
        insert_version(version);
    }

    ///
    /// @brief Insert a new value into the table with the provided key.
    ///
    /// The argument ``value`` should generally be obtained directly from a call to
    /// ``svs::lib::save()``.
    ///
    template <typename T> void insert(std::string_view key, T&& value) {
        table_.insert(key, std::forward<T>(value));
    }

    void insert(std::string_view key, const std::unique_ptr<toml::node>& node) {
        node->visit([&](const auto& inner) { table_.insert(key, inner); });
    }

    void insert(std::string_view key, std::unique_ptr<toml::node>&& node) {
        // Hack to get around TOML's inability to directly use unique pointers to nodes.
        // We basically need to pull out the concrete leaf type and insert that.
        std::move(node)->visit([&](auto inner) { table_.insert(key, std::move(inner)); });
    }

    /// @brief Checks if the container contains an element with the specified key.
    bool contains(std::string_view key) const { return table_.contains(key); }

    // Extract the underlying table.
    const toml::table& get() const& { return table_; }
    toml::table&& get() && { return std::move(table_); }
};

// clang-format off
namespace detail {
template <typename T, typename To>
concept HasZeroArgSaveTo = requires(const T& x) {
    { x.save() } -> std::same_as<To>;
};
}

template<typename T, typename... Args>
concept Loadable = requires(
    const T& x, const toml::table& table, Args&&... args
) {
    x.load(table, std::forward<Args>(args)...);
};

template<typename T, typename... Args>
concept StaticLoadable = requires(
    const toml::table& table, Args&&... args
) {
    T::load(table, std::forward<Args>(args)...);
};
// clang-format on

///
/// @brief Proxy object for an object ``x`` of type ``T``.
///
/// Specializations are expected to implement one of the following static functions.
///
/// \code{.cpp}
/// static RetType Saver<T>::save(const T&);
/// static RetType Saver<T>::save(const T&, const svs::lib::SaveContext&);
/// \endcode
///
/// If the first method has higher priority and will be called if both methods are
/// available.
///
/// The expected return type is either ``svs::lib::SaveTable`` or ``svs::lib::SaveNode``.
///
/// This class is automically defined for classes ``T`` with appropriate ``save()`` methods.
///
template <typename T> struct Saver {
    static SaveTable save(const T& x)
        requires detail::HasZeroArgSaveTo<T, SaveTable>
    {
        return x.save();
    }
    static SaveTable save(const T& x, const SaveContext& ctx) { return x.save(ctx); }
};

template <typename T>
concept SaveableContextFree = requires(const T& x) { Saver<T>::save(x); };

///
/// @brief Loader proxy-class for objects of type ``T``.
///
/// The following must be well formed as pre-requisites.
///
/// * ``Loader::toml_type`` must be defined and be one of the TOML built-in types:
///     - ``toml::table`` (default for classes using member ``save()`` and ``load()``
///       methods.)
///     - ``toml::node``
///     - ``toml::array``
///     - ``toml::value<T>`` or ``T`` where ``T`` is one of ``int64_t``, ``double``,
///       ``bool``, ``std::string``.
/// * ``Loader::is_version_free`` must be defined and constexpr-convertible to bool.
///   For classes using ``save()`` and ``load()`` member functions, this is ``false``.
///   If ``Loader::is_version_free`` evaluates to ``false``, then
///   ``std::is_same_v<Loader::toml_type, toml::table>`` must evaluate to ``true``.
///
/// Loading logic will attempt to call the following method with the following priority:
///
/// \code{.cpp}
/// // Requires Loader::is_version_free == true
/// T Loader<T>::load(const toml_type&, const svs::lib::Version&, Args&&...);
/// T Loader<T>::load(
///    const toml_type&, const svs::lib::LoadContext&, const svs::lib::Version&, Args&&...);
/// );
///
/// // Requires Loader::is_version_free == false
/// T Loader<T>::load(const toml_type&, Args&&...);
/// T Loader<T>::load(const toml_type&, const svs::lib::LoadContext&, Args&&...);
/// \endcode
///
template <typename T> struct Loader {
    using toml_type = toml::table;
    static constexpr bool is_version_free = false;

    // Context free path.
    // This is the preferred path in the load dispatching logic so should be chosen
    // if available.
    //
    // Use SFINAE to selectively disable if the type does not support context-free loading.
    template <typename... Args>
        requires StaticLoadable<T, const Version&, Args...>
    T load(const toml_type& table, const Version& version, Args&&... args) const {
        return T::load(table, version, std::forward<Args>(args)...);
    }

    template <typename... Args>
    T load(
        const toml_type& table,
        const LoadContext& ctx,
        const Version& version,
        Args&&... args
    ) const {
        return T::load(table, ctx, version, std::forward<Args>(args)...);
    }
};

// This is a hack for now as long as we have legacy loaders.
// Eventually, I'd like to move away from having loader objects be separate entities, but
// that is the job of another PR.
namespace detail {
template <typename T> struct ValueTypeDetector;
template <typename T> struct ValueTypeDetector<Loader<T>> {
    using type = T;
};

template <typename T>
using deduce_loader_value_type = typename detail::ValueTypeDetector<T>::type;

inline toml::table exit_hook(SaveTable val) { return std::move(val).get(); }
inline std::unique_ptr<toml::node> exit_hook(SaveNode val) { return std::move(val).get(); }

} // namespace detail

/// @defgroup save_group Object Saving

///
/// @ingroup save_group
/// @brief Save a class.
///
/// @param x The class to save.
/// @param ctx The current save context.
///
/// @returns A post processed table representation any data and metadata associated with
///     saving ``x``
///
/// When saving member classes, use ``svs::lib::save`` rather than directly invoking the
/// member ``save`` method.
///
/// Furthermore, the results should generally not be relied on directly. Rather, they should
/// be forwarded directly as the values of the initialer list when constructing a
/// ``svs::lib::SaveTable`` or when using ``svs::lib::SaveTable::insert()``.
///
/// If a ``toml::table`` is needed, use ``svs::lib::save_to_table()``.
///
template <typename T> auto save(const T& x, const SaveContext& ctx) {
    if constexpr (SaveableContextFree<T>) {
        return detail::exit_hook(Saver<T>::save(x));
    } else {
        return detail::exit_hook(Saver<T>::save(x, ctx));
    }
}

///
/// @ingroup save_group
/// @brief Save a class.
///
/// @param x The class to save.
///
/// @returns A post processed table representation any data and metadata associated with
///     saving ``x``
///
/// When saving member classes, use ``svs::lib::save`` rather than directly invoking the
/// member ``save`` method.
///
/// Furthermore, the results should generally not be relied on directly. Rather, they should
/// be forwarded directly as the values of the initialer list when constructing a
/// ``svs::lib::SaveTable`` or when using ``svs::lib::SaveTable::insert()``.
///
/// If a ``toml::table`` is needed, use ``svs::lib::save_to_table()``.
///
template <typename T> auto save(const T& x) { return detail::exit_hook(Saver<T>::save(x)); }

namespace detail {

// Context free loading
template <typename Loader, typename... Args>
    requires(!lib::first_is<LoadContext, Args...>())
auto load_impl(const Loader& loader, const toml::node& node, Args&&... args) {
    using To = typename Loader::toml_type;
    const To& converted = toml_helper::get_as<To>(node);
    if constexpr (Loader::is_version_free) {
        return loader.load(converted, std::forward<Args>(args)...);
    } else {
        auto version = get_version(converted, config_version_key);
        return loader.load(converted, version, std::forward<Args>(args)...);
    }
}

template <typename Loader, typename... Args>
auto load_impl(
    const Loader& loader, const toml::node& node, const LoadContext& ctx, Args&&... args
) {
    // What are we going to try to convert to.
    using To = typename Loader::toml_type;

    // See if the context-free path has a chance of succeeding. If so, drop `ctx`.
    if constexpr (Loadable<Loader, const Version&, Args...> || Loadable<Loader, Args...>) {
        return detail::load_impl(loader, node, std::forward<Args>(args)...);
    } else {
        // Contextual path.
        const To& converted = toml_helper::get_as<To>(node);
        if constexpr (Loader::is_version_free) {
            return loader.load(converted, ctx, std::forward<Args>(args)...);
        } else {
            static_assert(
                std::is_same_v<To, toml::table>,
                "Standard loadable objects must use a table for construction!"
            );

            auto version = get_version(converted, config_version_key);
            return loader.load(converted, ctx, version, std::forward<Args>(args)...);
        }
    }
}
} // namespace detail

/// @defgroup load_group Object Loading

///
/// @ingroup load_group
/// @brief Invoke the loader's ``.load()`` method with converted contents of ``node``.
///
/// See the documentation for ``svs::lib::Loader<T>`` for the priority of which member
/// function will be called.
///
template <typename Loader, typename... Args>
auto load(const Loader& loader, const toml::node& node, Args&&... args) {
    return detail::load_impl(loader, node, std::forward<Args>(args)...);
}

///
/// @ingroup load_group
/// @brief Like ``svs::lib::load()`` but try accessing the requested key.
///
/// Using this method can provide better runtime error messages.
///
/// @sa ``svs::lib::load(const Loader&, const toml::node&, Args&&)``.
///
template <typename Loader, typename... Args>
auto load_at(
    const Loader& loader, const toml::table& table, std::string_view key, Args&&... args
) {
    return detail::load_impl(
        loader, toml_helper::get_as<toml::node>(table, key), std::forward<Args>(args)...
    );
}

///
/// @ingroup load_group
/// @brief Like ``svs::lib::load()`` but try accessing the requested key.
///
/// Returns an empty ``std::optional`` if the requested key does not exist.
///
/// @sa ``svs::lib::load(const Loader&, const toml::node&, Args&&)``.
///
template <typename Loader, typename... Args>
std::optional<detail::deduce_loader_value_type<Loader>> try_load_at(
    const Loader& loader, const toml::table& table, std::string_view key, Args&&... args
) {
    using T = detail::deduce_loader_value_type<Loader>;
    auto view = table[key];
    if (!view) {
        return std::optional<T>();
    }
    return std::optional<T>{
        detail::load_impl(loader, *view.node(), std::forward<Args>(args)...)};
}

///
/// @ingroup load_group
/// @brief Invoke ``SelfLoader`` static  ``load()`` method.
///
/// See the documentation for ``svs::lib::Loader<T>`` for the priority of which member
/// function will be called.
///
template <typename SelfLoader, typename... Args>
auto load(const toml::node& node, Args&&... args) {
    return detail::load_impl(Loader<SelfLoader>(), node, std::forward<Args>(args)...);
}

template <typename SelfLoader, typename... Args>
auto load(const std::unique_ptr<toml::node>& node, Args&&... args) {
    return detail::load_impl(Loader<SelfLoader>(), *node, std::forward<Args>(args)...);
}

///
/// @ingroup load_group
/// @brief Like ``svs::lib::load()`` but try accessing the requested key.
///
/// Using this method can provide better runtime error messages.
///
/// @sa ``svs::lib::load(const toml::node&, Args&&)``.
///
template <typename SelfLoader, typename... Args>
auto load_at(const toml::table& table, std::string_view key, Args&&... args) {
    return load_at(Loader<SelfLoader>(), table, key, std::forward<Args>(args)...);
}

///
/// @ingroup load_group
/// @brief Like ``svs::lib::load()`` but try accessing the requested key.
///
/// Returns an empty ``std::optional`` if the requested key does not exist.
///
/// @sa ``svs::lib::load(const toml::node&, Args&&)``.
///
template <typename SelfLoader, typename... Args>
std::optional<detail::deduce_loader_value_type<Loader<SelfLoader>>>
try_load_at(const toml::table& table, std::string_view key, Args&&... args) {
    return try_load_at(Loader<SelfLoader>(), table, key, std::forward<Args>(args)...);
}

///
/// @brief Save a class to a `toml::table`.
///
/// @param x The class to save.
///
/// Requires the class to implment context-free saving.
/// Ensures that the return type is a ``toml::table``.
///
template <typename T> toml::table save_to_table(const T& x) {
    static_assert(
        std::is_same_v<decltype(svs::lib::save(x)), toml::table>,
        "Save to Table is only enabled for classes returing TOML tables."
    );
    return lib::save(x);
}

/// @brief Class allowing a lamba to be used for ad-hoc saving.
template <typename F> class SaveOverride {
  public:
    /// @brief Construct a new ``SaveOverride`` around the callable ``f``.
    SaveOverride(F&& f)
        : f_{std::move(f)} {}

    // Context free saving.
    // clang-format off
    SaveTable save() const
        requires requires(const F& f) { { f() } -> std::same_as<SaveTable>;}
    {
        return f_();
    }
    // clang-format on

    // Contextual saving.
    SaveTable save(const SaveContext& ctx) const { return f_(ctx); }

  private:
    F f_;
};

namespace detail {
template <typename Nodelike>
void save_node_to_file(
    Nodelike&& node,
    const std::filesystem::path& path,
    const lib::Version& version = CURRENT_SAVE_VERSION
) {
    auto top_table = toml::table(
        {{config_version_key, version.str()}, {config_object_key, SVS_FWD(node)}}
    );
    auto file = lib::open_write(path, std::ios_base::out);
    file << top_table << "\n";
}
} // namespace detail

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
/// @see svs::lib::save_to_file
///
template <typename T> void save_to_disk(const T& x, const std::filesystem::path& dir) {
    // Create the directory.
    // Per the documented API, if `dir` already exists, then there is no error.
    // Otherwise, the parent directory must exist.
    std::filesystem::create_directory(dir);

    // Assume the saving context is in the same directory as the path.
    auto ctx = SaveContext(dir);
    detail::save_node_to_file(lib::save(x, ctx), dir / config_file_name, ctx.version());
}

///
/// @brief Save the object into the given file.
///
/// @param x The object to save to disk.
/// @param path The file where the object will be saved.
///
/// This method requires that the class `x` implements context-free saving. That is, the
/// serialized representation of `x` does not need auxiliary files. If `x` does not
/// implement context-free saving, a compile-time error will ge emitted.
///
/// @see svs::lib::save_to_disk
///
template <typename T> void save_to_file(const T& x, const std::filesystem::path& path) {
    static_assert(SaveableContextFree<T>, "save_to_file requires context-free saving!");
    detail::save_node_to_file(lib::save(x), path);
}

/// @brief Class to enable a lambda to be used for ad-hoc loading.
template <typename F> class LoadOverride {
  public:
    using toml_type = toml::table;
    static constexpr bool is_version_free = false;

    /// @brief Construct a new ``LoadOverride`` around the callable ``f``.
    LoadOverride(F&& f)
        : f_{std::move(f)} {}

    // Context free loading.
    template <typename... Args>
    auto load(const toml::table& table, const lib::Version& version) const
        requires requires(const F& f) { f(table, version); }
    {
        return f_(table, version);
    }

    auto load(
        const toml::table& table, const lib::LoadContext& ctx, const lib::Version& version
    ) const {
        return f_(table, ctx, version);
    }

  private:
    F f_;
};

/// @defgroup load_from_disk_group Loading from Disk

///
/// @ingroup load_from_disk_group
/// @brief Load an object from a previously saved object directory.
///
/// Path can point to one of the following:
///
/// * A directory. In which case, the directory will be inspected for an `svs_config.toml`
///   file, which will then be used for object loading.
/// * A full path to a config TOML file. In this case, that file itself will be used
///   for object loading.
///
/// @param loader The loader object with a ``load`` method.
/// @param path The path to the object directory.
/// @param args Any arguments to forward to the final load method.
///
template <typename Loader, typename... Args>
auto load_from_disk(const Loader& loader, std::filesystem::path path, Args&&... args) {
    // Detect if we have a path to a file or directory.
    // If it's a directory, then assume we should append the canonical config file.
    if (std::filesystem::is_directory(path)) {
        path /= config_file_name;
    }
    auto table = toml::parse_file(path.c_str());
    auto version = get_version(table, config_version_key);
    auto ctx = LoadContext(path.parent_path(), version);
    return lib::load_at(loader, table, config_object_key, ctx, std::forward<Args>(args)...);
}

///
/// @ingroup load_from_disk_group
/// @brief Top level object reloading function.
///
/// Path can point to one of the following:
///
/// * A directory. In which case, the directory will be inspected for an `svs_config.toml`
///   file, which will then be used for object loading.
/// * A full path to a config TOML file. In this case, that file itself will be used
///   for object loading.
///
/// @tparam The class to be loaded using a static ``load`` method.
///
/// @param path The path to the object directory.
/// @param args Any arguments to forward to the final load method.
///
template <typename T, typename... Args>
auto load_from_disk(const std::filesystem::path& path, Args&&... args) {
    return lib::load_from_disk(Loader<T>(), path, std::forward<Args>(args)...);
}

///
/// @ingroup load_from_disk_group
/// @brief Load an object from a previously saved object file.
///
/// Path must point to a config TOML file and the loader *must* implement context-free
/// loading.
///
/// @param loader The loader object with a ``load`` method.
/// @param path The path to the object serialization file.
/// @param args Any arguments to forward to the final load method.
///
template <typename Loader, typename... Args>
auto load_from_file(const Loader& loader, std::filesystem::path path, Args&&... args) {
    auto table = toml::parse_file(path.c_str());
    return lib::load_at(loader, table, config_object_key, std::forward<Args>(args)...);
}

///
/// @ingroup load_from_disk_group
/// @brief Load an object from a previously saved object file.
///
/// Path must point to a config TOML file and the loader *must* implement context-free
/// loading.
///
/// @tparam The class to be loaded using a static ``load`` method.
///
/// @param path The path to the object serialization file.
/// @param args Any arguments to forward to the final load method.
///
template <typename T, typename... Args>
auto load_from_file(const std::filesystem::path& path, Args&&... args) {
    return lib::load_from_file(Loader<T>(), path, std::forward<Args>(args)...);
}

template <typename T> bool test_self_save_load(T& x, const std::filesystem::path& dir) {
    lib::save_to_disk(x, dir);
    T y = lib::load_from_disk<T>(dir);
    return x == y;
}

template <typename T> bool test_self_save_load_context_free(const T& x) {
    return x == svs::lib::load<T>(svs::lib::save(x));
}

/////
///// Built-in Types
/////

// Integers
template <std::integral I> struct Saver<I> {
    static SaveNode save(I x) { return SaveNode(lib::narrow<int64_t>(x)); }
};

template <std::integral I> struct Loader<I> {
    using toml_type = I;
    static constexpr bool is_version_free = true;
    static I load(I value) { return value; }
};

// Bool
template <> struct Saver<bool> {
    static SaveNode save(bool x) { return SaveNode(x); }
};

template <> struct Loader<bool> {
    using toml_type = bool;
    static constexpr bool is_version_free = true;
    static bool load(bool value) { return value; }
};

// Floating Point
template <std::floating_point F> struct Saver<F> {
    static SaveNode save(F x) { return SaveNode(lib::narrow<double>(x)); }
};

template <std::floating_point F> struct Loader<F> {
    // Intenionally leave this as ``double``, otherwise we can get narrowing errors for
    // values that entered in by hand in the TOML files (such as 1.2).
    using toml_type = double;
    static constexpr bool is_version_free = true;
    static F load(F value) { return value; }
};

// String-like
template <> struct Saver<std::string> {
    static SaveNode save(const std::string& x) { return SaveNode(x); }
};

template <> struct Saver<std::string_view> {
    static SaveNode save(std::string_view x) { return SaveNode(x); }
};

template <> struct Loader<std::string> {
    using toml_type = toml::value<std::string>;
    static constexpr bool is_version_free = true;
    static std::string load(const toml_type& value) { return value.get(); }
};

// Filesystem
template <> struct Saver<std::filesystem::path> {
    static SaveNode save(const std::filesystem::path x) {
        return SaveNode(std::string_view(x.native()));
    }
};

template <> struct Loader<std::filesystem::path> {
    using toml_type = toml::value<std::string>;
    static constexpr bool is_version_free = true;
    static std::filesystem::path load(const toml_type& value) { return value.get(); }
};

///// Now - we can finally implement this member function.
inline std::filesystem::path
LoadContext::resolve(const toml::table& table, std::string_view key) const {
    return this->resolve(load_at<std::filesystem::path>(table, key));
}

// Timepoint.
template <> struct Saver<std::chrono::time_point<std::chrono::system_clock>> {
    static SaveNode save(std::chrono::time_point<std::chrono::system_clock> x) {
        auto today = std::chrono::floor<std::chrono::days>(x);
        auto ymd = std::chrono::year_month_day(today);
        auto date = toml::date(
            static_cast<int>(ymd.year()),
            static_cast<unsigned>(ymd.month()),
            static_cast<unsigned>(ymd.day())
        );
        auto hh_mm_ss = std::chrono::hh_mm_ss(x - today);
        auto time = toml::time{
            lib::narrow_cast<uint8_t>(hh_mm_ss.hours().count()),
            lib::narrow_cast<uint8_t>(hh_mm_ss.minutes().count()),
            lib::narrow_cast<uint8_t>(hh_mm_ss.seconds().count()),
        };
        return SaveNode(toml::date_time(date, time));
    }
};

// Vectors
template <typename T, typename Alloc> struct Saver<std::vector<T, Alloc>> {
    static SaveNode save(const std::vector<T, Alloc>& v)
        requires SaveableContextFree<T>
    {
        auto array = toml::array();
        for (const auto& i : v) {
            array.push_back(lib::save(i));
        }
        return SaveNode(std::move(array));
    }

    static SaveNode save(const std::vector<T, Alloc>& v, const SaveContext& ctx) {
        auto array = toml::array();
        for (const auto& i : v) {
            array.push_back(lib::save(i, ctx));
        }
        return SaveNode(std::move(array));
    }
};

template <typename T, typename Alloc> struct Loader<std::vector<T, Alloc>> {
    using toml_type = toml::array;
    static constexpr bool is_version_free = true;

    template <typename... Args>
    static void do_load(std::vector<T, Alloc>& v, const toml_type& array, Args&&... args) {
        for (const toml::node& node : array) {
            v.push_back(lib::load<T>(node, args...));
        }
    }

    // Context free path without an explicit allocator argument.
    template <typename... Args>
        requires(!lib::first_is<LoadContext, Args...>() && !lib::first_is<Alloc, Args...>())
    static std::vector<T, Alloc> load(const toml_type& array, Args&&... args) {
        auto v = std::vector<T, Alloc>();
        do_load(v, array, std::forward<Args>(args)...);
        return v;
    }

    // Context free path with an explicit allocator argument.
    template <typename... Args>
    static std::vector<T, Alloc>
    load(const toml_type& array, const Alloc& alloc, Args&&... args) {
        auto v = std::vector<T, Alloc>(alloc);
        do_load(v, array, std::forward<Args>(args)...);
        return v;
    }

    // Contextual path without an explicit allocator argument.
    template <typename... Args>
        requires(!lib::first_is<Alloc, Args...>())
    static std::vector<T, Alloc> load(
        const toml_type& array, const LoadContext& ctx, Args&&... args
    ) {
        auto v = std::vector<T, Alloc>();
        do_load(v, array, ctx, std::forward<Args>(args)...);
        return v;
    }

    // Contextual path with an explicit allocator argument.
    template <typename... Args>
    static std::vector<T, Alloc> load(
        const toml_type& array, const LoadContext& ctx, const Alloc& alloc, Args&&... args
    ) {
        auto v = std::vector<T, Alloc>(alloc);
        do_load(v, array, ctx, std::forward<Args>(args)...);
        return v;
    }
};

/////
///// DataType
/////

// This needs to go here because the DataType is needed during bootstrapping.
template <> struct Saver<DataType> {
    static SaveNode save(DataType x) { return name(x); }
};

template <> struct Loader<DataType> {
    using toml_type = toml::value<std::string>;
    static constexpr bool is_version_free = true;
    static DataType load(const toml_type& val) { return parse_datatype(val.get()); }
};

/////
///// UUID
/////

template <> struct Saver<UUID> {
    static SaveNode save(UUID x) { return x.str(); }
};

template <> struct Loader<UUID> {
    using toml_type = toml::value<std::string>;
    static constexpr bool is_version_free = true;
    static UUID load(const toml_type& val) { return UUID(val.get()); }
};

/////
///// Percent
/////

template <> struct Saver<Percent> {
    static SaveNode save(Percent x) { return SaveNode(x.value()); }
};

template <> struct Loader<Percent> {
    using toml_type = double;
    static constexpr bool is_version_free = true;
    static Percent load(toml_type value) { return Percent(value); }
};

/////
///// Save a full 64-bit unsigned integer
/////

struct FullUnsigned {
  public:
    explicit FullUnsigned(uint64_t value)
        : value_{value} {}
    operator uint64_t() const { return value_; }
    uint64_t value() const { return value_; }

  public:
    uint64_t value_;
};

template <> struct Saver<FullUnsigned> {
    static SaveNode save(FullUnsigned x) {
        return SaveNode(std::bit_cast<int64_t>(x.value()));
    }
};

template <> struct Loader<FullUnsigned> {
    using toml_type = int64_t;
    static constexpr bool is_version_free = true;
    static FullUnsigned load(toml_type value) {
        return FullUnsigned(std::bit_cast<uint64_t>(value));
    }
};

/////
///// BinaryBlob
/////

struct BinaryBlobSerializer {
    static constexpr lib::Version save_version{0, 0, 0};
};

template <typename T>
    requires(std::is_trivially_copyable_v<T>)
class BinaryBlobSaver : private BinaryBlobSerializer {
  public:
    explicit BinaryBlobSaver(std::span<const T> data)
        : BinaryBlobSerializer()
        , data_{data} {}

    template <typename Alloc>
    explicit BinaryBlobSaver(const std::vector<T, Alloc>& data)
        : BinaryBlobSaver(lib::as_const_span(data)) {}

    SaveTable save(const SaveContext& ctx) const {
        auto path = ctx.generate_name("binary_blob", "bin");
        size_t bytes_written = 0;
        {
            auto ostream = lib::open_write(path);
            bytes_written += lib::write_binary(ostream, data_);
        }

        return SaveTable(
            BinaryBlobSerializer::save_version,
            {{"filename", lib::save(path.filename())},
             {"element_size", lib::save(sizeof(T))},
             {"element_type", lib::save(datatype_v<T>)},
             {"num_elements", lib::save(data_.size())}}
        );
    }

  private:
    std::span<const T> data_;
};

template <typename T, typename Alloc = std::allocator<T>>
    requires(std::is_trivially_copyable_v<T>)
class BinaryBlobLoader : private BinaryBlobSerializer {
  public:
    explicit BinaryBlobLoader(size_t num_elements, const Alloc& allocator)
        : BinaryBlobSerializer()
        , data_(num_elements, allocator) {}

    // Implicit conversion to `std::vector`
    operator std::vector<T, Alloc>() && { return std::move(data_); }

    static BinaryBlobLoader load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const Alloc& allocator = {}
    ) {
        if (version != BinaryBlobSerializer::save_version) {
            throw ANNEXCEPTION("Version mismatch!");
        }

        auto element_type = lib::load_at<DataType>(table, "element_type");
        constexpr auto expected_element_type = datatype_v<T>;
        if (element_type != expected_element_type) {
            throw ANNEXCEPTION(
                "Element type mismatch! Expected {}, got {}.",
                element_type,
                expected_element_type
            );
        }

        // If this is an unknown data type - the best we can try to do is verify that the
        // element sizes are correct.
        if (element_type == DataType::undef) {
            auto element_size = lib::load_at<size_t>(table, "element_size");
            if (element_size != sizeof(T)) {
                throw ANNEXCEPTION(
                    "Size mismatch for unknown element types. Expected {}, got {}.",
                    element_size,
                    sizeof(T)
                );
            }
        }

        auto num_elements = lib::load_at<size_t>(table, "num_elements");
        auto filename = ctx.resolve(table, "filename");
        auto loader = BinaryBlobLoader(num_elements, allocator);
        {
            auto istream = lib::open_read(filename);
            lib::read_binary(istream, loader.data_);
        }
        return loader;
    }

  private:
    std::vector<T, Alloc> data_{};
};

/////
///// Utility Macros
/////

// Expected Transformation:
// SVS_LIST_SAVE_(x, args...) -> {"x", svs::lib::save(x_, args...)}
#define SVS_LIST_SAVE_(name, ...)                     \
    {                                                 \
#name, svs::lib::save(name##_, ##__VA_ARGS__) \
    }

// Expected Transformation:
// SVS_INSERT_SAVE_(table, x, args...)
//  -> table.insert("x", svs::lib::save(x_, args...))
#define SVS_INSERT_SAVE_(table, name, ...) \
    table.insert(#name, svs::lib::save(name##_, ##__VA_ARGS__))

// Expected Transformation:
// SVS_LOAD_MEMBER_AT(table, x, args...)
//  -> svs::lib::load_at<std::decay_t<decltype(x_)>>(table, x_, args...)
#define SVS_LOAD_MEMBER_AT_(table, name, ...) \
    svs::lib::load_at<std::decay_t<decltype(name##_)>>(table, #name, ##__VA_ARGS__)

// Non-underscored version

// Expected Transformation:
// SVS_LIST_SAVE_(x, args...) -> {"x", svs::lib::save(x, args...)}
#define SVS_LIST_SAVE(name, ...)                   \
    {                                              \
#name, svs::lib::save(name, ##__VA_ARGS__) \
    }

// Expected Transformation:
// SVS_INSERT_SAVE_(table, x, args...)
//  -> table.insert("x", svs::lib::save(x, args...))
#define SVS_INSERT_SAVE(table, name, ...) \
    table.insert(#name, svs::lib::save(name, ##__VA_ARGS__))

// Expected Transformation:
// SVS_LOAD_MEMBER_AT(table, x, args...)
//  -> svs::lib::load_at<std::decay_t<decltype(x)>>(table, x_, args...)
#define SVS_LOAD_MEMBER_AT(table, name, ...) \
    svs::lib::load_at<std::decay_t<decltype(name)>>(table, #name, ##__VA_ARGS__)

} // namespace lib
} // namespace svs
