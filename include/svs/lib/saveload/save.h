/*
 * Copyright 2024 Intel Corporation
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

// saveload
#include "svs/lib/saveload/core.h"

// svs
#include "svs/lib/file.h"
#include "svs/lib/readwrite.h"
#include "svs/lib/version.h"

// stl
#include <filesystem>

namespace svs::lib {

///
/// @brief Context used when saving aggregate objects.
///
/// VERSION HISTORY
///
/// v0.0.0 - Original version.
/// v0.0.1 - Unknown change.
/// v0.0.2 - Added support for optional named `schemas` to be associated with serialized
///     objects.
///
constexpr Version CURRENT_SAVE_VERSION = Version(0, 0, 2);
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

// Unversioned internal saving.
// This is meant for use of internal types and shouldn't be used for general data
// structures.
struct SaveNode {
    std::unique_ptr<toml::node> node_;

    // NOTE: This relies on patched behavior in the TOML library to allow `make_node` to
    // properly pass through a already-created `unique_ptr<toml::node>`.
    template <typename T>
        requires(!std::is_same_v<std::remove_cvref_t<T>, SaveNode>)
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
    toml::table table_;

    void insert_metadata(std::string_view schema, const Version& version) {
        table_.insert(config_schema_key, schema);
        table_.insert(config_version_key, version.str());
    }

  public:
    /// @brief Construct an empty table with the given version.
    explicit SaveTable(std::string_view schema, const Version& version) {
        insert_metadata(schema, version);
    }

    ///
    /// @brief Construct a table using an initializer list of key-value pairs.
    ///
    /// Generally, values of the key-value pairs should be the return values from further
    /// calls to ``svs::lib::save()``.
    ///
    explicit SaveTable(
        std::string_view key,
        const Version& version,
        std::initializer_list<toml::impl::table_init_pair> init
    )
        : table_{init} {
        insert_metadata(key, version);
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

namespace detail {
template <typename T, typename To>
concept HasZeroArgSaveTo = requires(const T& x) {
                               { x.save() } -> std::same_as<To>;
                           };
}

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

namespace detail {
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
    auto file = svs::lib::open_write(path, std::ios_base::out);
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

} // namespace svs::lib
