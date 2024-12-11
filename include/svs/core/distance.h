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

#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/meta.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"

///
/// @defgroup distance_overload Common distance function overloads.
///

namespace svs {

// Documentation for these classes lives with the classes themselves.
using DistanceL2 = distance::DistanceL2;
using DistanceIP = distance::DistanceIP;
using DistanceCosineSimilarity = distance::DistanceCosineSimilarity;

///
/// @brief Runtime selector for built-in distance functions.
///
enum DistanceType {
    /// Minimize squared L2 distance. See: ``svs::distance::DistanceL2``.
    L2,
    /// Maximize inner product. See: ``svs::distance::DistanceIP``.
    MIP,
    /// Minimize cosine similarity. See: ``svs::distance::DistanceCosineSimilarity``.
    Cosine
};

inline constexpr std::string_view name(DistanceType type) {
    switch (type) {
        case DistanceType::L2: {
            return "L2";
        }
        case DistanceType::MIP: {
            return "MIP";
        }
        case DistanceType::Cosine: {
            return "Cosine";
        }
    }
    throw ANNEXCEPTION("Unknown distance type!");
}

inline DistanceType parse_distance_type(std::string_view str) {
    if (constexpr auto check = name(DistanceType::L2); check == str) {
        return DistanceType::L2;
    } else if (constexpr auto check = name(DistanceType::MIP); check == str) {
        return DistanceType::MIP;
    } else if (constexpr auto check = name(DistanceType::Cosine); check == str) {
        return DistanceType::Cosine;
    }
    throw ANNEXCEPTION("Unknown distance name: {}!", str);
}

namespace detail {
template <typename Distance> struct DistanceTypeEnumMap;

template <> struct DistanceTypeEnumMap<distance::DistanceL2> {
    static constexpr DistanceType value = DistanceType::L2;
};
template <> struct DistanceTypeEnumMap<distance::DistanceIP> {
    static constexpr DistanceType value = DistanceType::MIP;
};
template <> struct DistanceTypeEnumMap<distance::DistanceCosineSimilarity> {
    static constexpr DistanceType value = DistanceType::Cosine;
};
} // namespace detail

///
/// @brief Return the runtime enum for the built-in distance functor.
///
template <typename Distance>
inline constexpr DistanceType distance_type_v =
    detail::DistanceTypeEnumMap<Distance>::value;

template <typename Dist> struct DistanceConverter {
    static constexpr bool match(DistanceType x) { return x == distance_type_v<Dist>; }
    static constexpr Dist convert([[maybe_unused]] DistanceType x) {
        assert(match(x));
        return Dist();
    }
    static std::string_view description() { return name(distance_type_v<Dist>); }
};

template <>
struct lib::DispatchConverter<DistanceType, DistanceL2> : DistanceConverter<DistanceL2> {};
template <>
struct lib::DispatchConverter<DistanceType, DistanceIP> : DistanceConverter<DistanceIP> {};
template <>
struct lib::DispatchConverter<DistanceType, DistanceCosineSimilarity>
    : DistanceConverter<DistanceCosineSimilarity> {};

// Saving and Loading.
namespace lib {
template <> struct Saver<svs::DistanceType> {
    static SaveNode save(svs::DistanceType distance) { return svs::name(distance); }
};

template <> struct Loader<svs::DistanceType> {
    using toml_type = toml::value<std::string>;
    static svs::DistanceType load(const toml_type& val) {
        return svs::parse_distance_type(val.get());
    }
};
} // namespace lib

///
/// @brief Dynamically dispatch from an distance enum to a distance functor.
///
/// Most methods in the library require a distance functor to be given directly. However,
/// the decision of which functor to use is often a runtime decision.
///
/// This class provides a method of converting a ``svs::DistanceType`` enum to one of the
/// built-in functor types.
///
/// **NOTE**: Using a DistanceDispatcher will instantiate code-paths for all built-in
/// functors, which may have an impact on compile-time and binary size.
///
class DistanceDispatcher {
  public:
    ///
    /// @brief Construct a new DistanceDispatcher.
    ///
    /// @param distance_type The distance type to use.
    ///
    explicit DistanceDispatcher(DistanceType distance_type)
        : distance_type_{distance_type} {}

    ///
    /// @brief Dynamically dispatch to the correct distance functor.
    ///
    /// @param f A function who takes distance functor for its first argument. The
    ///     dispatcher will call ``f`` with the functor corresponding to the enum used
    ///     to construct the dispatcher.
    ///
    ///     All other arguments will be forwarded to ``f`` beginning at argument position 2.
    /// @param args Arguements to forward to ``f``.
    ///
    template <typename F, typename... Args> auto operator()(F&& f, Args&&... args) {
        switch (distance_type_) {
            case DistanceType::L2: {
                return f(DistanceL2{}, std::forward<Args>(args)...);
            }
            case DistanceType::MIP: {
                return f(DistanceIP{}, std::forward<Args>(args)...);
            }
            case DistanceType::Cosine: {
                return f(DistanceCosineSimilarity{}, std::forward<Args>(args)...);
            }
        }
        throw ANNEXCEPTION("unreachable reached"); // Make GCC happy
    }

  private:
    DistanceType distance_type_;
};

} // namespace svs

///// Formatting
template <> struct fmt::formatter<svs::DistanceType> : svs::format_empty {
    auto format(auto x, auto& ctx) const {
        return fmt::format_to(ctx.out(), "{}", svs::name(x));
    }
};
