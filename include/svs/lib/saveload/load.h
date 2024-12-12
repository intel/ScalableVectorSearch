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

// stl
#include <filesystem>

namespace svs::lib {

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

/// @brief A view into a serialized object tree without a directory context.
///
/// @tparam T The concrete TOML type this view points to.
///
/// This is a view class and thus cheap to take by value OR const reference.
///
/// However, this class *does not* extent the lifetime of the pointed to object.
/// The user must ensure that the node-tree being traversed outlives the lifetime of the
/// NodeView class.
template <toml_helper::TomlType T> class ContextFreeNodeView {
  public:
    /// The TOML type pointed to.
    using toml_type = T;
    /// Does this point to a ``toml::node`` base class.
    static constexpr bool is_node = std::is_same_v<T, toml::node>;
    /// Does this point to a ``toml::array`` base class.
    static constexpr bool is_array = std::is_same_v<T, toml::array>;
    /// Does this point to a ``toml::table`` specializaion.
    static constexpr bool is_table = false; // Defined by explicit specialization.
    /// Does this point to a ``toml::value`` specializaion.
    static constexpr bool is_value = toml_helper::TomlValue<T>;

    ContextFreeNodeView() = delete;

    /// @brief Construct a NodeView to the argument.
    explicit ContextFreeNodeView(const T& toml)
        : toml_{toml} {}

    /// @brief Perform a checked down-cast that throws on an invalid cast.
    template <toml_helper::TomlType U>
        requires(is_node)
    ContextFreeNodeView<U> cast() const {
        return ContextFreeNodeView<U>{cast_impl<U>()};
    }

    /// @brief Perform a checked down-cast that returns an empty optional on failure.
    template <toml_helper::TomlType U>
        requires(is_node)
    std::optional<ContextFreeNodeView<U>> try_cast() const {
        const auto* v = try_cast_impl<U>();
        if (v == nullptr) {
            return std::nullopt;
        }
        return std::optional<ContextFreeNodeView<U>>{std::in_place, *v};
    }

    /// @brief Allow implicit conversion to the underlying value.
    operator const T&() const
        requires(is_value)
    {
        return toml_;
    }

    /// @brief Visit each value in an array with the provided functor.
    template <typename F>
        requires(is_array)
    void visit(F&& f) const {
        for (const auto& i : toml_) {
            f(ContextFreeNodeView<toml::node>{i});
        }
    }

    /// @brief Return the source for the wrapped object.
    const toml::source_region& source() const { return toml_.source(); }

    /// @brief Return the underlying TOML object.
    const T& unwrap() const { return toml_; }

    // Delete the assignment operators, but allow the copy and move constructors.
    ContextFreeNodeView(const ContextFreeNodeView&) = default;
    ContextFreeNodeView(ContextFreeNodeView&&) = default;
    ContextFreeNodeView& operator=(const ContextFreeNodeView&) = delete;
    ContextFreeNodeView& operator=(ContextFreeNodeView&&) = delete;
    ~ContextFreeNodeView() = default;

  protected:
    // Protected internal caster to allow the contextual deserializer to implement
    // its own ``cast`` that propagates the LoadContext.
    template <toml_helper::TomlType U>
        requires(is_node)
    const U& cast_impl() const {
        return toml_helper::get_as<U>(toml_);
    }

    template <toml_helper::TomlType U>
        requires(is_node)
    const U* try_cast_impl() const {
        return toml_.template as<U>();
    }

  private:
    // Reference into the top-level table that we are deserializing from.
    const T& toml_;
};

/// @brief A view into a serialized object tree without a directory context.
///
/// @tparam T The concrete TOML type this view points to.
///
/// This is a view class and thus cheap to take by value OR const reference.
/// This class further specializes the ``ContextFreeNodeView`` by requiring the presence
/// of ``__schema__`` and ``__version__`` fields in the parsed table.
///
/// However, this class *does not* extent the lifetime of the pointed to object.
/// The user must ensure that the node-tree being traversed outlives the lifetime of the
/// NodeView class.
template <> class ContextFreeNodeView<toml::table> {
  public:
    /// The TOML type pointed to.
    using toml_type = toml::table;
    /// @copydoc ContextFreeNodeView<T>::is_node
    static constexpr bool is_node = false;
    /// @copydoc ContextFreeNodeView<T>::is_array
    static constexpr bool is_array = false;
    /// @copydoc ContextFreeNodeView<T>::is_table
    static constexpr bool is_table = true;
    /// @copydoc ContextFreeNodeView<T>::is_value
    static constexpr bool is_value = false;

    ContextFreeNodeView() = delete;

    /// @brief Construct a NodeView to the argument.
    ///
    /// Throws if the table does not contain keys:
    /// * ``__schema__``: With value type string
    /// * ``__version``: With value type string and parsable to ``svs::lib::Version``.
    explicit ContextFreeNodeView(const toml::table& table)
        : table_{table}
        , schema_{lib::get_schema(table)}
        , version_{lib::get_version(table)} {}

    /// @brief Return a ``ContextFreeNodeView`` at entry ``key``.
    ///
    /// Throws if this table does not contain ``key``.
    ContextFreeNodeView<toml::node> at(std::string_view key) const {
        return ContextFreeNodeView<toml::node>{at_impl(key)};
    }

    /// @brief Return a ``ContextFreeNodeView`` at entry ``key``.
    ///
    /// Return an empty optional if this table does not contain ``key``.
    std::optional<ContextFreeNodeView<toml::node>> try_at(std::string_view key) const {
        auto v = try_at_impl(key);
        if (!v) {
            return std::nullopt;
        }
        return std::optional<ContextFreeNodeView<toml::node>>{std::in_place, *v.node()};
    }

    /// @brief Return the schema for the underlying table.
    [[nodiscard]] std::string_view schema() const { return schema_; }

    /// @brief Return the version for the underlying table.
    [[nodiscard]] const lib::Version& version() const { return version_; }

    /// @brief Return whether or not this table contains an entry to ``key``.
    [[nodiscard]] bool contains(std::string_view key) const { return table_.contains(key); }

    /// @brief Return the source for the node for the given key.
    ///
    /// Throws if the key does not exist in the table.
    const toml::source_region& source_for(std::string_view key) const {
        return toml_helper::get_as<toml::node>(table_, key).source();
    }

    /// @brief Return the source for the wrapped table.
    const toml::source_region& source() const { return table_.source(); }

    /// @brief Return the underlying TOML table.
    const toml::table& unwrap() const { return table_; }

    // Delete the assignment operators and default the copy/move assignment operators.
    ContextFreeNodeView(const ContextFreeNodeView&) = default;
    ContextFreeNodeView(ContextFreeNodeView&&) = default;
    ContextFreeNodeView& operator=(const ContextFreeNodeView&) = delete;
    ContextFreeNodeView& operator=(ContextFreeNodeView&&) = delete;
    ~ContextFreeNodeView() = default;

  protected:
    const toml::node& at_impl(std::string_view key) const {
        return toml_helper::get_as<toml::node>(table_, key);
    }

    toml::node_view<const toml::node> try_at_impl(std::string_view key) const {
        return table_[key];
    }

  private:
    const toml::table& table_;
    std::string schema_;
    lib::Version version_;
};

/// @brief A refinement of ``ContextFreeNodeView<T>`` that contains a loading context.
///
/// A loading context provides the directory containing the object being deserialized,
/// providing a mechanism to resolve relative files paths to absolute paths.
///
/// Since it inherits from ``ContextFreeNodeView<T>``, implicit reference conversions are
/// allowed. This provides compatibility with classes that do not need a deserialization
/// context.
///
/// Note that this conversion *must* only be one way. It is okay to implicitly drop the
/// ``svs::lib::LoadContext`` if not needed, but not okay to dd it implicitly.
///
/// Furthermore, this class does not extent the lifetime of either the pointed to Node nor
/// the pointed to ``svs::lib::LoadContext``.
template <toml_helper::TomlType T> class NodeView : public ContextFreeNodeView<T> {
  public:
    using parent_type = ContextFreeNodeView<T>;

    NodeView() = delete;

    /// @brief Construct a NodeView pointing to the argument and provided context.
    explicit NodeView(const T& toml, const LoadContext& context)
        : ContextFreeNodeView<T>(toml)
        , context_{context} {}

    /// @copydoc ContextFreeNodeView<T>::cast()
    template <toml_helper::TomlType U>
        requires(parent_type::is_node)
    NodeView<U> cast() const {
        return NodeView<U>(parent_type::template cast_impl<U>(), context_);
    }

    /// @copydoc ContextFreeNodeView<T>::try_cast()
    template <toml_helper::TomlType U>
        requires(parent_type::is_node)
    std::optional<NodeView<U>> try_cast() const {
        const auto* v = parent_type::template try_cast_impl<U>();
        if (v == nullptr) {
            return std::nullopt;
        }
        return std::optional<NodeView<U>>(std::in_place, *v, context_);
    }

    /// @brief Return a ``NodeView`` at entry ``key``.
    ///
    /// Throws if this table does not contain ``key``.
    NodeView<toml::node> at(std::string_view key) const
        requires(parent_type::is_table)
    {
        return NodeView<toml::node>(parent_type::at_impl(key), context_);
    }

    /// @brief Return a ``NodeView`` at entry ``key``.
    ///
    /// Throws if this table does not contain ``key``.
    std::optional<NodeView<toml::node>> try_at(std::string_view key) const
        requires(parent_type::is_table)
    {
        auto v = parent_type::try_at_impl(key);
        if (!v) {
            return std::nullopt;
        }
        return std::optional<NodeView<toml::node>>{std::in_place, *v.node(), context_};
    }

    /// @copydoc ContextFreeNodeView<T>::visit()
    template <typename F>
        requires(parent_type::is_array)
    void visit(F&& f) const {
        const toml::array& underlying = parent_type::unwrap();
        for (const toml::node& i : underlying) {
            f(NodeView<toml::node>{i, context_});
        }
    }

    /// @brief Return the ``svs::lib::LoadContext`` associated with this node.
    const LoadContext& context() const { return context_; }

    /// @brief Return the resolved filepath for the relative file stored at ``key``.
    std::filesystem::path resolve_at(std::string_view key) const
        requires(parent_type::is_table)
    {
        const toml::value<std::string>& v =
            at(key).template cast<toml::value<std::string>>();
        return context_.resolve(v.get());
    }

    /// @brief Return the resolved relative filepath as an absolute path.
    std::filesystem::path resolve(const std::filesystem::path& relative) const {
        return context_.resolve(relative);
    }

  private:
    // The load context for deserializing.
    const LoadContext& context_;
};

/// @brief Return a context-free view of the argument.
inline ContextFreeNodeView<toml::node> node_view(const toml::node& node) {
    return ContextFreeNodeView<toml::node>{node};
}

/// @brief Return a contextual view of the argument.
inline NodeView<toml::node> node_view(const toml::node& node, const lib::LoadContext& ctx) {
    return NodeView<toml::node>{node, ctx};
}

/// @brief Return a context-free view of the argument.
inline ContextFreeNodeView<toml::table> node_view(const toml::table& table) {
    return ContextFreeNodeView<toml::table>{table};
}

/// @brief Return a contextual view of the argument.
inline NodeView<toml::table>
node_view(const toml::table& table, const lib::LoadContext& ctx) {
    return NodeView<toml::table>{table, ctx};
}

/// @brief Return a context-free view of the entry at ``key`` in the provided table.
///
/// Throws an unspecified exception if the key does not exist.
inline ContextFreeNodeView<toml::node>
node_view_at(const toml::table& table, std::string_view key) {
    const auto& v = toml_helper::get_as<toml::node>(table, key);
    return node_view(v);
}

// Compatibility with some return-paths for ``svs::lib::save()``.
inline ContextFreeNodeView<toml::node> node_view(const std::unique_ptr<toml::node>& ptr) {
    if (ptr == nullptr) {
        throw ANNEXCEPTION("Trying to dereference a null pointer!");
    }
    return node_view(*ptr);
}

/////
///// Top level - owning deserialization contexts.
/////

/// @brief An anonymous deserialized object.
///
/// Most of the deserialization logic uses view types while traversing the parsed table
/// for efficiency reasons. These view do now extent the lifetime of the table they are
/// working through.
///
/// This class provides a stable, owning base that can be used to launch deserialzation
/// attempts.
class ContextFreeSerializedObject {
  public:
    using toml_type = toml::node;

    ContextFreeSerializedObject() = delete;

    /// @brief Assume ownership of the parsed table.
    ///
    /// This table must conform to the SVS global serialization schema and contain a key
    /// ``object`` that contains the data to begin deserialization.
    ContextFreeSerializedObject(toml::table&& table)
        : node_{std::make_shared<toml::table>(std::move(table))} {}

    /// @brief Obtain the underlying serialized object.
    ///
    /// Throws an unspecified exception if no such object exists.
    ContextFreeNodeView<toml::node> object() const {
        return ContextFreeNodeView<toml::node>{object_impl()};
    }

    /// @brief Obtain the underlying serialized object.
    ///
    /// Return an empty optional is no such object exists.
    std::optional<ContextFreeNodeView<toml::node>> try_object() const {
        auto v = try_object_impl();
        if (!v) {
            return std::nullopt;
        }
        return std::optional<ContextFreeNodeView<toml::node>>{std::in_place, *v.node()};
    }

    /// @brief Cast the underlying object to a more refined TOML type.
    ///
    /// Throws an unspecified exception is the cast is invalid.
    template <toml_helper::TomlType U> ContextFreeNodeView<U> cast() const {
        return object().template cast<U>();
    }

    /// @brief Cast the underlying object to a more refined TOML type.
    ///
    /// Returns an empty optional is the cast is invalid.
    template <toml_helper::TomlType U>
    std::optional<ContextFreeNodeView<U>> try_cast() const {
        auto o = try_object();
        if (!o) {
            return std::nullopt;
        }
        return o->template try_cast<U>();
    }

  protected:
    const toml::node& object_impl() const {
        auto v = node_->operator[](lib::config_object_key);
        if (!v) {
            throw ANNEXCEPTION("Trying to access invalid key {}!", lib::config_object_key);
        }
        return *v.node();
    }

    toml::node_view<const toml::node> try_object_impl() const {
        return node_->operator[](lib::config_object_key);
    }

  private:
    // Use a ``shared_ptr`` for the cases where we need to expliciltly clone contexts.
    std::shared_ptr<const toml::table> node_;
};

/// @brief A refinement of ``svs::lib::ContextFreeSerializationObject`` that has a context.
///
/// The context can be used to resolve relative filepaths into absolute filepaths.
class SerializedObject : public ContextFreeSerializedObject {
  public:
    using toml_type = toml::node;

    SerializedObject() = delete;

    /// @brief Assuem ownership of the parsed table and context.
    ///
    /// This table must conform to the SVS global serialization schema and contain a key
    /// ``object`` that contains the data to begin deserialization.
    SerializedObject(toml::table&& table, lib::LoadContext ctx)
        : ContextFreeSerializedObject{std::move(table)}
        , ctx_{std::move(ctx)} {}

    /// @copydoc ContextFreeSerializedObject::object()
    NodeView<toml::node> object() const {
        return NodeView<toml::node>{ContextFreeSerializedObject::object_impl(), ctx_};
    }

    /// @copydoc ContextFreeSerializedObject::try_object()
    std::optional<NodeView<toml::node>> try_object() const {
        auto v = ContextFreeSerializedObject::try_object_impl();
        if (!v) {
            return std::nullopt;
        }
        return std::optional<NodeView<toml::node>>{std::in_place, *v.node(), ctx_};
    }

    /// @copydoc ContextFreeSerializedObject::cast()
    template <toml_helper::TomlType U> NodeView<U> cast() const {
        return object().template cast<U>();
    }

    /// @copydoc ContextFreeSerializedObject::try_cast()
    template <toml_helper::TomlType U> std::optional<NodeView<U>> try_cast() const {
        auto o = try_object();
        if (!o) {
            return std::nullopt;
        }
        return o->template try_cast<U>();
    }

    /// @brief Return the associated ``svs::lib::LoadContext``.
    const lib::LoadContext& context() const { return ctx_; }

    /// @brief Return a resolved filepath for the entry at ``key``.
    ///
    /// Assumes that:
    /// 1. The underlying object exists (i.e., there is a key ``object`` in the table).
    /// 2. The underlying object can be cast to a ``toml::table``.
    /// 3. The key exists in the table.
    /// 4. The value associated with this key is a string.
    ///
    /// If any of these assumptions fails, an unspecified exception is thrown.
    std::filesystem::path resolve_at(std::string_view key) const {
        const toml::value<std::string>& v = object()
                                                .template cast<toml::table>()
                                                .at(key)
                                                .template cast<toml::value<std::string>>();
        return ctx_.resolve(v.get());
    }

    /// @brief Resolve a relative path to an absolute path.
    std::filesystem::path resolve(const std::filesystem::path& relative) const {
        return ctx_.resolve(relative);
    }

  private:
    // The context for deserializing.
    lib::LoadContext ctx_;
};

namespace detail {

template <toml_helper::TomlType From, toml_helper::TomlType To>
constexpr bool downcast_needed() {
    static_assert(std::is_base_of_v<From, To>);
    return !std::is_same_v<To, From>;
}

/////
///// traits
/////

template <typename T> inline constexpr bool is_nodelike_v = false;
template <typename T> inline constexpr bool is_nodelike_v<ContextFreeNodeView<T>> = true;
template <typename T> inline constexpr bool is_nodelike_v<NodeView<T>> = true;
template <> inline constexpr bool is_nodelike_v<ContextFreeSerializedObject> = true;
template <> inline constexpr bool is_nodelike_v<SerializedObject> = true;

template <typename T>
concept NodeLike = is_nodelike_v<T>;

template <typename T> inline constexpr bool is_tablelike_v = false;
template <> inline constexpr bool is_tablelike_v<ContextFreeNodeView<toml::table>> = true;
template <> inline constexpr bool is_tablelike_v<NodeView<toml::table>> = true;

template <typename T>
concept TableLike = is_tablelike_v<T>;

template <typename T> inline constexpr bool is_arraylike_v = false;
template <> inline constexpr bool is_arraylike_v<ContextFreeNodeView<toml::array>> = true;
template <> inline constexpr bool is_arraylike_v<NodeView<toml::array>> = true;

template <typename T>
concept ArrayLike = is_arraylike_v<T>;

// clang-format off

// Does the class define a method that checks for direct loading.
//
// If so, it then we also require it to implement the direct load method.
template <typename T, typename... Args>
concept HasStaticDirectLoad =
    requires(const std::filesystem::path& path, const Args&... args) {
        { T::can_load_direct(path, args...) } -> std::convertible_to<bool>;
    };

template <typename Loader, typename... Args>
concept HasDirectLoad =
    requires(const Loader& loader, const std::filesystem::path& path, const Args&... args) {
        { loader.can_load_direct(path, args...) } -> std::convertible_to<bool>;
    };

template <typename T>
concept HasCompatibilityCheck = requires(std::string_view schema, lib::Version version) {
    { T::check_load_compatibility(schema, version) } -> std::convertible_to<bool>;
};
// clang-format on

} // namespace detail

/// A context-free deserialization table with a schema and a version.
using ContextFreeLoadTable = ContextFreeNodeView<toml::table>;

/// A contextual. deserialization table with a schema and a version.
using LoadTable = NodeView<toml::table>;

/// @brief The defauilt definition of a loader for a class ``T``.
template <typename T> struct Loader {
    using toml_type = toml::table;

    /// @brief Check compatibility of the table's schema and version.
    ///
    /// If ``T::check_load_compatibility(std::string_view, svs::lib::Version)`` exists,
    /// then that will be called. Otherwise, ``T::save_version`` and
    /// ``T::serialization_schema`` will be checked.
    bool check_compatibility(const ContextFreeLoadTable& table) const {
        if constexpr (detail::HasCompatibilityCheck<T>) {
            return T::check_load_compatibility(table.schema(), table.version());
        } else {
            return T::save_version == table.version() &&
                   T::serialization_schema == table.schema();
        }
    }

    /// @brief Invoke ``T::load(table, SVS_FWD(args)...)``.
    ///
    /// A compatibility check will first be performed and an exception thrown if the check
    /// fails.
    template <detail::TableLike Table, typename... Args>
    T load(const Table& table, Args&&... args) const {
        if (!check_compatibility(table)) {
            throw ANNEXCEPTION(
                "Trying to deserialize incompatible object ({}, {}) from file {}.",
                table.schema(),
                table.version(),
                fmt::streamed(table.source())
            );
        }
        return T::load(table, SVS_FWD(args)...);
    }

    /// @brief Invoke ``T::try_load(table, SVS_FWD(args)...)``.
    ///
    /// A compatibility check will first be performed and a ``lib::Unexpected`` will be
    /// returned if the check fails.
    template <detail::TableLike Table, typename... Args>
    lib::TryLoadResult<T> try_load(const Table& table, const Args&... args) const {
        if (!check_compatibility(table)) {
            return lib::Unexpected(TryLoadFailureReason::InvalidSchema);
        }
        return T::try_load(table, args...);
    }

    /// @brief Invoke ``T::can_load_direct(path, args...)``.
    ///
    /// Only applicable if such a static member is defined with results convertible to
    /// ``bool``.
    template <typename... Args>
        requires detail::HasStaticDirectLoad<T, Args...> bool
    can_load_direct(const std::filesystem::path& path, const Args&... args) const {
        return T::can_load_direct(path, args...);
    }

    /// @brief Invoke ``T::load_direct(path, args...)``
    ///
    /// This method is applicable if ``T::can_load_direct`` is defined and will only
    /// be called if ``T::can_load_direct`` returns ``true``.
    template <typename... Args>
        requires detail::HasStaticDirectLoad<T, Args...>
    T load_direct(const std::filesystem::path& path, const Args&... args) const {
        return T::load_direct(path, args...);
    }

    /// @brief Invoke ``T::try_load_direct(path, args...)``
    ///
    /// This method is applicable if ``T::can_load_direct`` is defined and will only
    /// be called if ``T::can_load_direct`` returns ``true``.
    template <typename... Args>
        requires detail::HasStaticDirectLoad<T, Args...>
    lib::TryLoadResult<T>
    try_load_direct(const std::filesystem::path& path, const Args&... args) const {
        return T::try_load_direct(path, args...);
    }
};

///// load

/// @brief Load an object of type ``T`` from ``node`` using ``loader.load()``.
///
/// @param loader - The loader specialization to use.
/// @param node   - The serialized source node to pull from.
/// @param args   - Arguments to forward to ``loader.load()``.
///
/// Attempts to refine ``node`` to ``typename Loader<T>::toml_type`` if needed, throwing an
/// unspecified exception if the down-cast fails.
template <typename T, detail::NodeLike Node, typename... Args>
T load(const Loader<T>& loader, const Node& node, Args&&... args) {
    using From = typename Node::toml_type;
    using To = typename Loader<T>::toml_type;
    if constexpr (detail::downcast_needed<From, To>()) {
        return loader.load(node.template cast<To>(), SVS_FWD(args)...);
    } else {
        return loader.load(node, SVS_FWD(args)...);
    }
}

/// @brief Load an object of type ``T`` from ``node`` using
/// ``svs::lib::Loader<T>().load()``.
///
/// @tparam T - The type of object to load.
///
/// @param node   - The serialized source node to pull from.
/// @param args   - Arguments to forward to ``loader.load()``.
///
/// @see svs::lib::load(Loader<T>,Node,Args...)
template <typename T, detail::NodeLike Node, typename... Args>
T load(const Node& node, Args&&... args) {
    return load(Loader<T>(), node, SVS_FWD(args)...);
}

///// load_at

/// @brief Load an object of type ``T`` from the selected value in the table.
///
/// @param loader - The loader specialization to use.
/// @param table  - The table to access.
/// @param key    - The key used to access the table.
/// @param args   - Arguments to forward to ``loader.load()``.
///
/// Throws an unspecified exception if ``key`` does not exist in ``table``.
///
/// @see svs::lib::load(Loader<T>,Node,args...)
template <typename T, detail::TableLike Table, typename... Args>
T load_at(
    const Loader<T>& loader, const Table& table, std::string_view key, Args&&... args
) {
    return load(loader, table.at(key), SVS_FWD(args)...);
}

/// @brief Load an object of type ``T`` from the selected value in the table.
///
/// @tparam T - The type of object to load.
///
/// @param table  - The table to access.
/// @param key    - The key used to access the table.
/// @param args   - Arguments to forward to ``loader.load()``.
///
/// Throws an unspecified exception if ``key`` does not exist in ``table``.
/// Uses a default constructed ``svs::lib::Loader<T>``.
///
/// @see svs::lib::load(Loader<T>,Node,args...)
template <typename T, detail::TableLike Table, typename... Args>
T load_at(const Table& table, std::string_view key, Args&&... args) {
    return load_at(Loader<T>(), table, key, SVS_FWD(args)...);
}

///// try_load

template <typename T, detail::NodeLike Node, typename... Args>
lib::TryLoadResult<T>
try_load(const Loader<T>& loader, const Node& node, const Args&... args) {
    using From = typename Node::toml_type;
    using To = typename Loader<T>::toml_type;
    if constexpr (detail::downcast_needed<From, To>()) {
        auto down = node.template try_cast<To>();
        if (!down) {
            return lib::Unexpected(TryLoadFailureReason::Other);
        }
        return loader.try_load(*down, args...);
    } else {
        return loader.try_load(node, args...);
    }
}

template <typename T, detail::NodeLike Node, typename... Args>
lib::TryLoadResult<T> try_load(const Node& node, const Args&... args) {
    return try_load(Loader<T>(), node, args...);
}

///// try_load_at

template <typename T, detail::TableLike Table, typename... Args>
lib::TryLoadResult<T> try_load_at(
    const Loader<T>& loader, const Table& table, std::string_view key, const Args&... args
) {
    auto ex = table.try_at(key);
    if (!ex) {
        return lib::Unexpected(lib::TryLoadFailureReason::MissingKey);
    }
    return try_load(loader, *ex, args...);
}

template <typename T, detail::TableLike Table, typename... Args>
lib::TryLoadResult<T>
try_load_at(const Table& table, std::string_view key, const Args&... args) {
    return try_load_at(Loader<T>(), table, key, args...);
}

/////
///// Top Level Functions
/////

namespace detail {

inline void
check_global_version(lib::Version version, const std::filesystem::path& source) {
    if (version == lib::Version{0, 0, 1}) {
        throw ANNEXCEPTION(
            "File {} is using serialization version {}. Please upgrade using "
            "`svs.upgrader.upgrade({})` and try again.",
            source,
            version,
            source
        );
    }
    if (version != lib::Version{0, 0, 2}) {
        throw ANNEXCEPTION(
            "Cannot handle file from the future with serialization version {}!", version
        );
    }
}

inline SerializedObject begin_deserialization(const std::filesystem::path& fullpath) {
    auto table = toml::parse_file(fullpath.c_str());
    auto version = get_version(table, config_version_key);
    svs::lib::detail::check_global_version(version, fullpath);
    return SerializedObject{
        std::move(table), lib::LoadContext{fullpath.parent_path(), version}};
}

} // namespace detail

inline SerializedObject begin_deserialization(const std::filesystem::path& path) {
    // If we were given a path that is not a file, assume it is a directory with the
    // canonical structure (containing an ``svs_config.toml`` file).
    if (std::filesystem::is_directory(path)) {
        return detail::begin_deserialization(path / config_file_name);
    }
    return detail::begin_deserialization(path);
}

inline ContextFreeSerializedObject
begin_context_free_deserialization(const std::filesystem::path& fullpath) {
    auto table = toml::parse_file(fullpath.c_str());
    auto version = get_version(table, config_version_key);
    svs::lib::detail::check_global_version(version, fullpath);
    return ContextFreeSerializedObject{std::move(table)};
}

///// load_from_disk

/// @brief Load an object of type ``T``
///
///
template <typename T, typename... Args>
T load_from_disk(const Loader<T>& loader, std::filesystem::path path, Args&&... args) {
    constexpr bool has_direct =
        detail::HasDirectLoad<Loader<T>, std::remove_cvref_t<Args>...>;

    // Check if the path we have is a file.
    // If it is a file and the provided loader allows for direct loading, give that loader
    // a chance to check and maybe perform direct loading.
    bool maybe_file = !std::filesystem::is_directory(path);
    if constexpr (has_direct) {
        if (maybe_file && loader.can_load_direct(path, args...)) {
            return loader.load_direct(path, SVS_FWD(args)...);
        }
    }

    // At this point, we will try the saving/loading framework to load the object.
    // Here we go!
    return lib::load(loader, begin_deserialization(path), SVS_FWD(args)...);
}

template <typename T, typename... Args>
T load_from_disk(const std::filesystem::path& path, Args&&... args) {
    return lib::load_from_disk(Loader<T>(), path, SVS_FWD(args)...);
}

///// load_from_file

template <typename T, typename... Args>
T load_from_file(const Loader<T>& loader, std::filesystem::path path, Args&&... args) {
    return lib::load(loader, begin_context_free_deserialization(path), SVS_FWD(args)...);
}

template <typename T, typename... Args>
T load_from_file(const std::filesystem::path& path, Args&&... args) {
    return lib::load_from_file(Loader<T>(), path, SVS_FWD(args)...);
}

///// try_load_from_disk

template <typename T, typename... Args>
lib::TryLoadResult<T> try_load_from_disk(
    const Loader<T>& loader, const std::filesystem::path& path, Args&&... args
) {
    constexpr bool has_direct =
        detail::HasDirectLoad<Loader<T>, std::remove_cvref_t<Args>...>;

    // Check if the path we have is a file.
    // If it is a file and the provided loader allows for direct loading, give that loader
    // a chance to check and maybe perform direct loading.
    bool maybe_file = !std::filesystem::is_directory(path);
    if constexpr (has_direct) {
        if (maybe_file && loader.can_load_direct(path, args...)) {
            return loader.try_load_direct(path, SVS_FWD(args)...);
        }
    }

    // At this point, we will try the saving/loading framework to load the object.
    // Here we go!
    return lib::try_load(loader, begin_deserialization(path), SVS_FWD(args)...);
}

template <typename T, typename... Args>
lib::TryLoadResult<T>
try_load_from_disk(const std::filesystem::path& path, Args&&... args) {
    return lib::try_load_from_disk(Loader<T>(), path, SVS_FWD(args)...);
}

} // namespace svs::lib
