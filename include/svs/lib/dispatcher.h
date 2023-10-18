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

// svs
#include "svs/lib/meta.h"
#include "svs/lib/tuples.h"

// stl
#include <optional>
#include <unordered_map>

namespace svs::lib {

///
/// Helper type to assist in building dispatch tables for various index implementations.
/// A dispatch table consists of keys used for the dispatch, and mapped values which
/// contain type erased `std::functions` that call the implementing function using the
/// types used to create the key.
///
/// Dependent on a type `T` which must have the following properties:
///
/// -- Type Aliases
/// * `T::key_type` - A tuple consisting of runtime values corresponding to the unique
///   types used to construct the inner function.
///
///   For example, if the implemented function depends on an arithmetic type `T`` and a set
///   dimensionality `N`, the inner function would be constructed using the types
///   `svs::meta::Type<T>()` and `svs::meta::Val<N>()` while the key tuple would consist
///   of `(svs::datatype_v<T>, N)`.
///
/// * `T::mapped_type` - The type of the `std::function` implementing the dispatch.
///
/// -- Static Methods
/// ```
/// template<typename F>
/// static void T::fill(F&& f);
/// ```
/// Fill in a dispatch table. For each specialization of interest, call
/// `f(std::pair<T::key_type, T::mapped_type>);` to register the dispatched function.
///
/// -- API
/// The main API used to interact with the dispatcher is the `lookup()` static method, which
/// will use the provided arguments as a key to find a registered implementation or fail
/// with ``svs::ANNException``.
///
/// **Implementation details**: The dispatch table is implemented using a magic static
/// variable. Thus, a new dispatch table is not constructed every time `lookup()` is called.
///
template <typename T> class Dispatcher {
  public:
    using key_type = typename T::key_type;
    using mapped_type = typename T::mapped_type;
    using map_type = std::unordered_map<key_type, mapped_type, svs::lib::TupleHash>;

    ///
    /// Perform the dynamic dispatch encoded in the `dispatcher`.
    ///
    /// @param try_generic Boolean flag to indicate that generic (Dynamic) dimensionality
    ///        can be used. If `false`, then the dimensionality argument `ndims` must be
    ///        used as given.
    /// @param ndims The dimensionality of the data involved. Usually passed to try and
    ///        match an implementation specialized for that dimension. If
    ///        `try_generic == true`, then the dispatch logic will try a fallback using
    ///        dynamically sized dimensionality as well.
    /// @param args Any other arguments required the the dispatch tuple.
    ///
    /// **Note**: The `ndims` argument will always be passed to the tail of the constructed
    /// dispatch tuple.
    ///
    template <typename... Args>
    static mapped_type lookup(bool try_generic, size_t ndims, Args&&... args) {
        auto result = lookup_impl(try_generic, ndims, std::forward<Args>(args)...);
        if (result) {
            return *result;
        }
        throw ANNEXCEPTION("Unimplemented specialization!");
    }

    ///
    /// @brief Return whether a specialization exists without throwing an exception.
    ///
    /// See the documentation for ``lookup`` for argument semantics.
    ///
    template <typename... Args>
    static bool contains(bool try_generic, size_t ndims, Args&&... args) {
        return lookup_impl(try_generic, ndims, std::forward<Args>(args)...).has_value();
    }

    ///
    /// @brief Return all registered keys with the dispatcher.
    ///
    static std::vector<key_type> keys() {
        auto keys = std::vector<key_type>();
        for (const auto& kv : get()) {
            keys.push_back(kv.first);
        }
        return keys;
    }

  private:
    static map_type populate() {
        auto map = map_type();
        T::fill([&map](std::pair<key_type, mapped_type> kv) { map.insert(std::move(kv)); });
        return map;
    }

    static const map_type& get() {
        static map_type dispatcher = populate();
        return dispatcher;
    }

    // Common path for higher level APIs.
    // We're returning a pointer, which is a little scary.
    template <typename... Args>
    static std::optional<mapped_type>
    lookup_impl(bool try_generic, size_t ndims, Args&&... args) {
        const auto& dispatcher = Dispatcher::get();
        auto key = std::make_tuple(std::forward<Args>(args)..., ndims);
        auto end = dispatcher.end();
        if (auto iter = dispatcher.find(key); iter != end) {
            return {iter->second};
        }

        // If the first attempt failed, try performing an generic lookup.
        if (try_generic && ndims != Dynamic) {
            key = std::make_tuple(args..., Dynamic);
            if (auto iter = dispatcher.find(key); iter != dispatcher.end()) {
                return {iter->second};
            }
        }
        return {std::nullopt};
    }
};
} // namespace svs::lib
