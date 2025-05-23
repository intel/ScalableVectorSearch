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
#include "svs/core/distance/distance_core.h"
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

template <typename svs::arch::MicroArch Arch>
struct DistanceTypeEnumMap<distance::DistanceL2<Arch>> {
    static constexpr DistanceType value = DistanceType::L2;
};
template <typename svs::arch::MicroArch Arch>
struct DistanceTypeEnumMap<distance::DistanceIP<Arch>> {
    static constexpr DistanceType value = DistanceType::MIP;
};
template <typename svs::arch::MicroArch Arch>
struct DistanceTypeEnumMap<distance::DistanceCosineSimilarity<Arch>> {
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

template <svs::arch::MicroArch Arch>
struct lib::DispatchConverter<DistanceType, svs::distance::DistanceL2<Arch>>
    : DistanceConverter<svs::distance::DistanceL2<Arch>> {};
template <typename svs::arch::MicroArch Arch>
struct lib::DispatchConverter<DistanceType, svs::distance::DistanceIP<Arch>>
    : DistanceConverter<svs::distance::DistanceIP<Arch>> {};
template <svs::arch::MicroArch Arch>
struct lib::DispatchConverter<DistanceType, svs::distance::DistanceCosineSimilarity<Arch>>
    : DistanceConverter<svs::distance::DistanceCosineSimilarity<Arch>> {};

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

// Factory for per-architecture distance dispatching
template <DistanceType DT> struct DistanceTag {};

template <DistanceType DT, svs::arch::MicroArch Arch> struct DistanceFactory;

template <svs::arch::MicroArch Arch> struct DistanceFactory<DistanceType::L2, Arch> {
    using type = svs::distance::DistanceL2<Arch>;
};

template <svs::arch::MicroArch Arch> struct DistanceFactory<DistanceType::MIP, Arch> {
    using type = svs::distance::DistanceIP<Arch>;
};

template <svs::arch::MicroArch Arch> struct DistanceFactory<DistanceType::Cosine, Arch> {
    using type = svs::distance::DistanceCosineSimilarity<Arch>;
};

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
    ///     For MicroArch-dispatching, all of this functionality is wrapped in a lambda
    ///     which utilizes the DistanceFactory above to instantiate the distance with
    ///     the correct MicroArch.
    ///
    ///     All other arguments will be forwarded to ``f`` beginning at argument position 2.
    /// @param args Arguments to forward to ``f``.
    ///
    template <typename F, typename... Args> auto operator()(F&& f, Args&&... args) {
        switch (distance_type_) {
            case DistanceType::L2:
                return svs::arch::dispatch_by_arch(
                    [&]<svs::arch::MicroArch Arch>(auto&&... inner_args) -> decltype(auto) {
                        using Distance =
                            typename DistanceFactory<DistanceType::L2, Arch>::type;
                        return f(
                            Distance{}, std::forward<decltype(inner_args)>(inner_args)...
                        );
                    },
                    std::forward<Args>(args)...
                );

            case DistanceType::MIP:
                return svs::arch::dispatch_by_arch(
                    [&]<svs::arch::MicroArch Arch>(auto&&... inner_args) -> decltype(auto) {
                        using Distance =
                            typename DistanceFactory<DistanceType::MIP, Arch>::type;
                        return f(
                            Distance{}, std::forward<decltype(inner_args)>(inner_args)...
                        );
                    },
                    std::forward<Args>(args)...
                );

            case DistanceType::Cosine:
                return svs::arch::dispatch_by_arch(
                    [&]<svs::arch::MicroArch Arch>(auto&&... inner_args) -> decltype(auto) {
                        using Distance =
                            typename DistanceFactory<DistanceType::Cosine, Arch>::type;
                        return f(
                            Distance{}, std::forward<decltype(inner_args)>(inner_args)...
                        );
                    },
                    std::forward<Args>(args)...
                );
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
