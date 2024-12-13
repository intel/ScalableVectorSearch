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
#include "svs/lib/datatype.h"
#include "svs/lib/meta.h"
#include "svs/lib/tuples.h"

// stl
#include <functional>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace svs::lib {

/////
///// Dispatcher V2
/////

/// The worst possible invalid match.
inline constexpr int64_t invalid_match = -1;
/// The best possible match.
inline constexpr int64_t perfect_match = 0;
/// The next best possible match.
inline constexpr int64_t imperfect_match = 1;

/// @brief Match was found using an implicit conversion.
///
/// This conversion is performed on arguments with the same type (or ref-compatible types).
/// Use a non-zero value to allow specializations to provide better matches than the type
/// identity.
inline constexpr int64_t implicit_match = 10000;

/// Return whether all entries in argument `match` are non-negative.
template <size_t N> constexpr bool is_valid_match(const std::array<int64_t, N>& match) {
    for (size_t i = 0; i < N; ++i) {
        if (match[i] < 0) {
            return false;
        }
    }
    return true;
}

/// Resolve ties between equally applicable match scores using lexicographical scoring.
struct LexicographicResolver {
    template <size_t N> using array_type = std::array<int64_t, N>;

    constexpr LexicographicResolver() = default;

    template <size_t N>
    constexpr bool operator()(const array_type<N>& x, const array_type<N>& y) {
        return x < y;
    }
};

/// @brief Customization point for defining dispatch conversion rules.
///
/// Expected API:
/// @code{cpp}
/// template<> struct DispatchConverter<From, To> {
///     // Return a score for matching arguments of type `From` to type `To`.
///     // * Negative values indicate an invalid match (cannot convert).
///     // * Non-negative values are scored with lower values given higher priority.
///     static int64_t match(const std::remove_cvref_t<From>&);
///
///     // Perform a dispatch conversion.
///     // This behavior of this function is undefined if `match` returns an invalid score.
///     static To convert(From);
///
///     // An optional method describing the acceptable value for this conversion.
///     static std::string description();
/// }
/// @endcode
///
/// Note that specialization requires full cv-ref qualification of the ``From`` and ``To``
/// types in order to be applicable.
template <typename From, typename To> struct DispatchConverter : std::false_type {};

namespace detail {

template <typename From, typename To>
inline constexpr bool is_same_remove_cvref_v =
    std::is_same_v<std::remove_cvref_t<From>, std::remove_cvref_t<To>>;

template <typename From, typename To> consteval bool implicitly_dispatch_convertible() {
    if (!is_same_remove_cvref_v<From, To> || !std::is_convertible_v<From, To>) {
        return false;
    }

    // Otherwise, we're convertible, but we don't necessarily want to convert a
    // `const lvalue-ref` to a value, which would require a copy.
    if (!std::is_lvalue_reference_v<To>) {
        return !std::is_lvalue_reference_v<From>;
    }
    return true;
}

} // namespace detail

/// Two types are considered dispatcher convertible if:
/// * Removing cv-ref qualifiers from ``From`` and ``To`` yields the same type.
/// * ``From`` and be forwarded to ``To`` without invoking a copy constructor.
///
/// For a type non-reference type ``T`` the following implicit conversion are allowed:
/// * ``T -> T`` (using move construction), ``T -> const T&``, ``T&& -> const T&``
/// * ``T& -> T&``, ``T& -> const T&``
/// * ``const T& -> const T&``
/// * ``T&& -> T``, ``T&& -> const T&``, ``T&& -> T&&``
template <typename From, typename To>
concept ImplicitlyDispatchConvertible = detail::implicitly_dispatch_convertible<From, To>();

/// @brief A specialization of ``DispatchConverter`` for implicit conversions.
template <typename From, typename To>
    requires ImplicitlyDispatchConvertible<From, To>
struct DispatchConverter<From, To> {
    using FromBase = std::remove_cvref_t<From>;
    static constexpr int64_t match(const FromBase& SVS_UNUSED(from)) {
        return implicit_match;
    }
    static constexpr To&& convert(From&& x) { return SVS_FWD(x); }
    static constexpr std::string_view description() { return "all values"; }
};

/// Concept indicating whether a specialization of ``DispatchConverter`` e
/// xists
/// for this combination of arguments - implying that dispatcher conversion is well-defined.
template <typename From, typename To>
concept DispatchConvertible =
    (!std::derived_from<DispatchConverter<From, To>, std::false_type>);

/// @brief Return the matching score of an instance of type ``From`` to the type ``To``.
///
/// @param x The value being dispatched on.
///
/// @returns An signed integer score. Scores less than 0 imply invalid match and the entire
///     method being considered should be discarded.
///
///     All non-negative results should be considered with a lower number having a higher
///     priority. So ``0`` has the highest priority, followed by ``1``, then ``2`` etc.
///
/// Internally, this method calls ``svs:lib::DispatchConverter<From, To>::match`` and
/// forward the result. However, if the ``DispatchConverter`` returns a ``bool``, then the
/// return value will be converted to the canonical ``int64_t`` representation
/// appropriately.
template <typename From, typename To>
constexpr int64_t dispatch_match(const std::remove_cvref_t<From>& x) {
    static_assert(DispatchConvertible<From, To>);
    auto ret = DispatchConverter<From, To>::match(x);
    // Allow customization to return boolean values and have those be interpreted
    // in a sensible manner.
    if constexpr (std::is_same_v<decltype(ret), bool>) {
        return ret ? 0 : -1;
    } else {
        return ret;
    }
}

/// @brief Use dispatch conversion to convert a value of type ``From`` to ``To``.
///
/// It is undefined behavior to call this method if
/// ``svs::lib::dispatch_match<From, To>(x)`` is invalid.
template <typename From, typename To> constexpr To dispatch_convert(From&& x) {
    return DispatchConverter<From, To>::convert(SVS_FWD(x));
}

namespace detail {

// clang-format off
template <typename From, typename To>
concept DocumentedDispatch = requires {
    { DispatchConverter<From, To>::description() } -> std::convertible_to<std::string>;
};
// clang-format on

} // namespace detail

/// @brief Return documentation for a dispatch conversion, if available.
///
/// If Such conversion is not available
/// (``svs::lib::DispatchConverter<From, To>::description()`` was not defined), then returns
/// the sentinel string "unknown".
template <typename From, typename To> auto dispatch_description() -> std::string {
    static_assert(DispatchConvertible<From, To>);
    // If the description overload exists - then recklessly call it and try to construct
    // a string from it.
    if constexpr (requires { DispatchConverter<From, To>::description(); }) {
        return std::string{DispatchConverter<From, To>::description()};
    } else {
        return "unknown";
    }
}

namespace variant {

enum DispatchCategory { Value, Ref, ConstRef };
template <DispatchCategory Cat, typename T> struct Apply {
    using type = T;
};
template <typename T> struct Apply<Ref, T> {
    using type = T&;
};
template <typename T> struct Apply<ConstRef, T> {
    using type = const T&;
};

template <DispatchCategory Cat, typename T> using apply_t = typename Apply<Cat, T>::type;

template <DispatchCategory Cat, typename To, typename... Ts>
inline constexpr bool is_applicable = (DispatchConvertible<apply_t<Cat, Ts>, To> || ...);

/// @brief Match all applicable alternatives of a variant to the destination type.
template <DispatchCategory Cat, typename To, typename... Ts> struct VariantDispatcher {
    // Use reference collapsing  to perform the following conversions:
    // Cat = Value    -> std::variant<Ts...>&&
    // Cat = Ref      -> std::variant<Ts...>& && -> std::variant<Ts...>&
    // Cat = ConstRef -> const std::variant<Ts...>& && -> const std::variant<Ts...>&
    using variant_type = apply_t<Cat, std::variant<Ts...>>&&;

    // Require at least one of the alternatives to be compatible with the right.
    static_assert(is_applicable<Cat, To, Ts...>);

    /// @brief Match the current alternative in the variant to ``To``.
    ///
    /// If an alternative type is not ``svs::lib::DispatchConvertible`` with ``To``, then
    /// return ``svs::lib::invalid_match``.
    static constexpr int64_t match(const std::variant<Ts...>& x) {
        return std::visit<int64_t>(
            []<typename T>(const T& alternative) {
                using Tx = apply_t<Cat, T>;
                if constexpr (DispatchConvertible<Tx, To>) {
                    return dispatch_match<Tx, To>(alternative);
                } else {
                    return invalid_match;
                }
            },
            x
        );
    }

    /// @brief Dispatch convert the current alternative in the variant to the type ``To``.
    ///
    /// Throws ``svs::ANNException`` if such a conversion is undefined.
    static constexpr To convert(variant_type x) {
        return std::visit<To>(
            [&]<typename T>(T&& alternative) -> To {
                if constexpr (DispatchConvertible<T, To>) {
                    return static_cast<To>(dispatch_convert<T, To>(SVS_FWD(alternative)));
                }
                throw ANNEXCEPTION("Invalid dispatch conversion!");
            },
            SVS_FWD(x)
        );
    }

    /// @brief Document all possible conversion from the variant to ``To``.
    static std::string description() {
        auto matches = std::vector<std::string>();
        auto alternatives = std::vector<size_t>();
        size_t i = 0;

        // Note that multiple alternative types can map to the same target type.
        // We try to document all such combinations.
        auto f = [&i, &matches, &alternatives]<typename From>() {
            if constexpr (DispatchConvertible<From, To>) {
                matches.push_back(dispatch_description<From, To>());
                alternatives.push_back(i);
            }
            ++i;
        };

        // Append all applicable conversions.
        (f.template operator()<apply_t<Cat, Ts>>(), ...);

        assert(!matches.empty());
        return fmt::format(
            "{} -- (union alternative{} {})",
            fmt::join(matches, " OR "),
            alternatives.size() > 1 ? "s" : "",
            fmt::join(alternatives, ", ")
        );
    }
};

} // namespace variant

///// Variant

// Variant by value
template <typename To, typename... Ts>
    requires variant::is_applicable<variant::Value, To, Ts...>
struct DispatchConverter<std::variant<Ts...>, To>
    : variant::VariantDispatcher<variant::Value, To, Ts...> {};

// Variant by rvalue-ref
template <typename To, typename... Ts>
    requires variant::is_applicable<variant::Value, To, Ts...>
struct DispatchConverter<std::variant<Ts...>&&, To>
    : variant::VariantDispatcher<variant::Value, To, Ts...> {};

// Variant by ref
template <typename To, typename... Ts>
    requires variant::is_applicable<variant::Ref, To, Ts...>
struct DispatchConverter<std::variant<Ts...>&, To>
    : variant::VariantDispatcher<variant::Ref, To, Ts...> {};

// variant by const-ref
template <typename To, typename... Ts>
    requires variant::is_applicable<variant::ConstRef, To, Ts...>
struct DispatchConverter<const std::variant<Ts...>&, To>
    : variant::VariantDispatcher<variant::ConstRef, To, Ts...> {};

/////
///// Implementations
/////

/// A dispatch tag meant to dispatch from an ``ExtentArg`` to a staticly-compiled extent.
///
/// The special type ``ExtentTag<svs::Dynamic>`` should be used to target methods that
/// can accept all dimensionality.
template <size_t N> struct ExtentTag {};

/// A dispatch argument for calling dispatch methods with static extent.
struct ExtentArg {
  public:
    /// Construct a new ExtentArg requesting a dynamic extent.
    ExtentArg() = default;

    /// Construct a new ExtentArg matching a compile-time extent.
    ///
    /// @param value The extent to match. Set to ``svs::Dynamic`` to request a generic
    ///     fallback method.
    /// @param force If ``true``, the matching between ``value`` and the compile-time extent
    ///     must be exact. A value of ``false`` implies that generic fallback methods can
    ///     be used if a better match is not found.
    ///
    ExtentArg(size_t value, bool force)
        : value_{value}
        , force_{force} {}

    /// Construct a new ExtentArg matching a compile-time extent.
    ///
    /// @param value The compile-time extent to match. If not such method is available, then
    ///     a dynamic fallback will be selected.
    explicit ExtentArg(size_t value)
        : ExtentArg(value, false) {}

    ///// Members
    size_t value_ = Dynamic;
    bool force_ = false;
};

template <size_t N> struct DispatchConverter<ExtentArg, ExtentTag<N>> {
    static int64_t match(ExtentArg v) {
        if (N == v.value_) {
            return perfect_match;
        }

        if (N == Dynamic && !v.force_) {
            return imperfect_match;
        }
        return invalid_match;
    }

    static ExtentTag<N> convert([[maybe_unused]] ExtentArg x) {
        assert(match(x) >= 0);
        return ExtentTag<N>{};
    }

    static std::string description() {
        if constexpr (N == Dynamic) {
            return "any";
        } else {
            return fmt::format("{}", N);
        }
    }
};

/////
///// Dispatcher
/////

namespace detail {

using DocFunctionPtr = std::string (*)();

// Utility struct to propagate function signatures.
template <typename F> struct Signature;
template <typename Ret, typename... Args> struct Signature<Ret(Args...)> {
    using arg_signature_type = Signature<void(Args...)>;
};

// Helper type to deduce function signatures from callable objects by bouncing through
// `std::function`'s CTAD constructors.
template <typename T> struct DeduceSignature;
template <typename Ret, typename... Args>
struct DeduceSignature<std::function<Ret(Args...)>> {
    using type = Signature<Ret(Args...)>;
};

// N.B: Trying to use `std::function's` CTAD guides directly like:
//
// template <typename F>
// using function_signature_t = decltype(std::function(std::declval<F>()));
//
// Doesn't seem to work on GCC.
//
// Instead - we need to bounce through a helper function.
template <typename F> auto make_std_function(const F& f) { return std::function{f}; }

template <typename F>
using function_signature_t = decltype(make_std_function(std::declval<F>()));

template <typename F>
using signature_t = typename DeduceSignature<function_signature_t<F>>::type;

template <typename F> using arg_signature_t = typename signature_t<F>::arg_signature_type;
template <typename F> signature_t<F> make_signature() { return signature_t<F>{}; }
template <typename F> arg_signature_t<F> make_arg_signature() {
    return arg_signature_t<F>{};
}

// Using a set of dispatch arguments and a set of target arguments, construct a function
// pointer that will call `dispatch_match` on each pair of dispatch-target arguments and
// return the results of all such calls as an array.
template <typename... DispatchArgs, typename... TargetArgs>
auto make_matcher(
    Signature<void(DispatchArgs...)> SVS_UNUSED(dispatch_sig),
    Signature<void(TargetArgs...)> SVS_UNUSED(target_sig)
) {
    static_assert(sizeof...(DispatchArgs) == sizeof...(TargetArgs));
    using return_type = std::array<int64_t, sizeof...(DispatchArgs)>;
    // Stateless lambda -> function pointer using unary "+".
    return +[](const std::remove_cvref_t<DispatchArgs>&... args) -> return_type {
        return std::array<int64_t, sizeof...(DispatchArgs)>{
            dispatch_match<DispatchArgs, TargetArgs>(args)...};
    };
}

template <typename Ret, typename... DispatchArgs, typename... TargetArgs, typename F>
auto make_converter_impl(
    Signature<Ret(DispatchArgs...)> SVS_UNUSED(dispatch_signature),
    Signature<void(TargetArgs...)> SVS_UNUSED(target_signature),
    F&& f_
) -> std::function<Ret(DispatchArgs...)> {
    static_assert(
        std::invocable<const std::remove_cvref_t<F>&, TargetArgs...>,
        "Dispatch targets must be const-invocable with the target arguments!"
    );
    return std::function<Ret(DispatchArgs...)>([f = SVS_FWD(f_)](DispatchArgs&&... args) {
        return f(dispatch_convert<DispatchArgs, TargetArgs>(SVS_FWD(args))...);
    });
}

// Construct a `std::function` with argument types `DispatchArgs...` and return type `Ret`
// that will perform dispatch conversion on each argument to the position-wise corresponding
// argument of `f` before invoking `f`.
//
// Requires that dispatch conversion is defined for each combination of dispatch and target
// argument types.
template <typename Ret, typename... DispatchArgs, typename F>
auto make_converter(Signature<Ret(DispatchArgs...)> dispatch_sig, F&& f)
    -> std::function<Ret(DispatchArgs...)> {
    return make_converter_impl(
        dispatch_sig, make_arg_signature<std::remove_cvref_t<F>>(), SVS_FWD(f)
    );
}

template <typename... DispatchArgs, typename... TargetArgs>
std::unique_ptr<DocFunctionPtr[]> make_descriptors(
    Signature<void(DispatchArgs...)> SVS_UNUSED(dispatch_sig),
    Signature<void(TargetArgs...)> SVS_UNUSED(target_sig)
) {
    static_assert(sizeof...(DispatchArgs) == sizeof...(TargetArgs));
    auto table =
        std::unique_ptr<DocFunctionPtr[]>(new DocFunctionPtr[sizeof...(DispatchArgs)]);
    size_t i = 0;
    auto assignment = [&table, &i]<typename From, typename To>() {
        table[i] = &dispatch_description<From, To>;
        ++i;
    };
    (assignment.template operator()<DispatchArgs, TargetArgs>(), ...);
    return table;
}

} // namespace detail

// Hook to generate matcher documentation and hold this information in the DispatchTarget.
// If this tag is not passed to the DispatchTarget constructor, then the templates to
// generate documentation will never be instantiated and thus should not necessarily show
// up in the final binary.
struct BuildDocsTag {};
struct NoDocsTag {};

/// @brief Tag to build argument-conversion documentation.
inline constexpr BuildDocsTag dispatcher_build_docs{};
/// @brief Tag to suppress argument-conversion documentation.
inline constexpr NoDocsTag dispatcher_no_docs{};

/// @brief Method wrapper for the target of a dispatch operation.
///
/// @tparam Ret The return type of the method.
/// @tparam Args The dispatch argument types of the method.
template <typename Ret, typename... Args> class DispatchTarget {
  public:
    /// The number of arguments this method accepts.
    static constexpr size_t num_args = sizeof...(Args);
    /// The full signature of the dispatch arguments.
    using signature_type = detail::Signature<Ret(Args...)>;
    /// The signature of the dispatch objects without a return type.
    using arg_signature_type = detail::Signature<void(Args...)>;
    /// The type encoding a match based on dispatch values.
    using match_type = std::array<int64_t, num_args>;

    /// @brief Construct a DispatchTarget around the callable ``f`` with no documentation.
    ///
    /// @param tag Indicate no argument conversion documention is required.
    /// @param f The function to wrap for dispatch. The wrapped functor must have a
    ///     const-qualified call operator and *no* non-const-qualified call operator.
    ///
    /// The following requirements must hold:
    ///
    /// 1. The argument types of ``f`` must be deducible (in other words, ``f`` cannot have
    ///    an overloaded call operator nor can its call operator be templated).
    ///
    /// 2. The number of arguments of ``f`` must match ``Dispatcher::num_args``.
    ///
    /// 3. Furthermore, dispatch conversion must be defined between each dispatch argument
    ///    and its corresponding argument in ``f``.
    ///
    /// If any of these requirements fails, this method should not compile.
    template <typename Callable>
    DispatchTarget(NoDocsTag SVS_UNUSED(tag), Callable f)
        : match_{detail::make_matcher(
              arg_signature_type{}, detail::make_arg_signature<Callable>()
          )}
        , call_{detail::make_converter(signature_type{}, std::move(f))} {}

    /// @brief Construct a DispatchTarget around the callable ``f`` with documentation.
    ///
    /// @param tag Indicate argument conversion documention is required.
    /// @param f The function to wrap for dispatch. The wrapped functor must have a
    ///     const-qualified call operator and *no* non-const-qualified call operator.
    ///
    /// The following requirements must hold:
    ///
    /// 1. The argument types of ``f`` must be deducible (in other words, ``f`` cannot have
    ///    an overloaded call operator nor can its call operator be templated).
    ///
    /// 2. The number of arguments of ``f`` must match ``Dispatcher::num_args``.
    ///
    /// 3. Furthermore, dispatch conversion must be defined between each dispatch argument
    ///    and its corresponding argument in ``f``.
    ///
    /// If any of these requirements fails, this method should not compile.
    template <typename Callable>
    DispatchTarget(BuildDocsTag SVS_UNUSED(tag), Callable f)
        : DispatchTarget(dispatcher_no_docs, std::move(f)) {
        documentation_ = detail::make_descriptors(
            arg_signature_type{}, detail::make_arg_signature<Callable>()
        );
    }

    /// @brief Return the result of matching each argument with the wrapped method.
    match_type check_match(const std::remove_cvref_t<Args>&... args) const {
        return match_(args...);
    }

    /// @brief Invoke the wrapped method by dispatch-converting each argument.
    Ret invoke(Args&&... args) const { return call_(SVS_FWD(args)...); }

    /// @brief Return dispatch documentation for argument ``i``.
    ///
    /// If ``i >= num_args``, throws ``svs::ANNException`` indicating a bounds error.
    /// If the DispatchTarget was constructed without documentation, then this function
    /// returns the string "unknown">
    std::string description(size_t i) const {
        if (i >= num_args) {
            throw ANNEXCEPTION(
                "Bound error. Accessing method table of size {} at index {}!", num_args, i
            );
        }

        // Use a fallback value if this target was constructed without documentation.
        if (documentation_ != nullptr) {
            return documentation_[i]();
        }
        return "unknown";
    }

  private:
    // Matchers are required to be stateless - so using a function pointer is okay.
    match_type (*match_)(const std::remove_cvref_t<Args>&...);
    std::function<Ret(Args...)> call_;
    // Documentation per-conversion.
    // Will be left as `nullptr` if not constructed with the `BuildDocsTag`.
    std::unique_ptr<detail::DocFunctionPtr[]> documentation_ = nullptr;
};

/// @brief A dynamic, multi-method dispatcher for registering specializations.
///
/// @tparam Ret The return type of invoking ai contained method.
/// @tparam Args The run-time arguments to dispatch over.
///
/// Multiple target methods can be registered with the dispatcher, provided that each
/// target method has the same number of arguments and dispatch conversion between each
/// target argument type and its corresponding member in ``Args`` is defined.
///
/// When invoked, the dispatcher will find the most applicable registered target by applying
/// ``svs::lib::dispatch_match`` on its arguments and the argument types of each registered
/// method.
///
/// The most specific applicable method will then be invoked by calling
/// ``svs::lib::dispatch_convert`` on each argument to its corresponding target type.
template <typename Ret, typename... Args> class Dispatcher {
  public:
    // Type Aliases
    using target_type = DispatchTarget<Ret, Args...>;
    /// @brief The type used to represent method matches for scoring.
    using match_type = std::array<int64_t, sizeof...(Args)>;
    using return_type = Ret;

    /// @brief Construct an empty dispatcher.
    Dispatcher() = default;

    /// @brief Return the number of registered candidates.
    size_t size() const { return candidates_.size(); }

    /// @brief Return the number of arguments the dispatcher expects to receive.
    static constexpr size_t num_args() { return sizeof...(Args); }

    /// @brief Register a callable with the dispatcher.
    template <typename F> void register_target(F f) {
        candidates_.emplace_back(dispatcher_no_docs, std::move(f));
    }

    /// @brief Register a callable with the dispatcher with conversion documentation.
    template <typename F> void register_target(BuildDocsTag build_docs, F f) {
        candidates_.emplace_back(build_docs, std::move(f));
    }

    /// @brief Get the index and the score of the best match.
    ///
    /// If no match is found, then the optional in the return value will be empty.
    /// In this case, the contents of the ``match_type`` is undefined.
    std::pair<std::optional<size_t>, match_type>
    best_match(const std::remove_cvref_t<Args>&... args) const {
        auto best_match = match_type{};
        auto match_index = std::optional<size_t>{std::nullopt};
        auto resolver = LexicographicResolver();

        for (size_t i = 0, imax = size(); i < imax; ++i) {
            const auto& candidate = candidates_.at(i);
            auto match = candidate.check_match(args...);
            if (is_valid_match(match)) {
                if (!match_index.has_value() || resolver(match, best_match)) {
                    match_index = i;
                    best_match = match;
                }
            }
        }
        return std::make_pair(std::move(match_index), std::move(best_match));
    }

    /// Return whether or not the given collection of arguments can be matched with any
    /// method registered in the dispatcher.
    bool has_match(const std::remove_cvref_t<Args>&... args) const {
        auto [match_index, _] = best_match(args...);
        return match_index.has_value();
    }

    /// Invoke the best matching method and return the result.
    Ret invoke(Args... args) const {
        auto [match_index, _] = best_match(args...);
        if (!match_index) {
            throw ANNEXCEPTION("Could not find a suitable method!");
        }
        return candidates_.at(match_index.value()).invoke(SVS_FWD(args)...);
    }

    /// @brief Return dispatch documentation for the given method and argument.
    ///
    /// Throws an ``svs::ANNException`` if ``method >= size()``  or
    /// ``argument >= num_args()``.
    std::string description(size_t method, size_t argument) const {
        if (method >= size()) {
            throw ANNEXCEPTION(
                "Trying to get documentation for method {} but only {} methods are "
                "registered.",
                method,
                size()
            );
        }
        return candidates_[method].description(argument);
    }

    // Default special member functions.
    Dispatcher(Dispatcher&&) = default;
    Dispatcher& operator=(Dispatcher&&) = default;
    ~Dispatcher() = default;

  private:
    // Make the copy constructor private so we *can* copy if we want, but it is difficult
    // to accidentally trigger a copy operation.
    Dispatcher& operator=(const Dispatcher&) = default;
    Dispatcher(const Dispatcher&) = default;

    // The registered targets
    std::vector<target_type> candidates_ = {};
};

/////
///// Built-in Conversions
/////

template <typename T>
    requires svs::has_datatype_v<T>
struct DispatchConverter<svs::DataType, lib::Type<T>> {
    static constexpr bool match(svs::DataType type) { return type == datatype_v<T>; }

    static constexpr lib::Type<T> convert([[maybe_unused]] svs::DataType type) {
        assert(match(type));
        return {};
    }

    static constexpr std::string_view description() { return name(datatype_v<T>); }
};

} // namespace svs::lib
