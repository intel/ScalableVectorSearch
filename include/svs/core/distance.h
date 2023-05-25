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

#pragma once

#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"
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
            default: {
                throw ANNEXCEPTION("unreachable reached"); // Make GCC happy
            }
        }
    }

  private:
    DistanceType distance_type_;
};

///
/// Hook to allow distance implementations to customize adjusting when performing
/// self-comparison for a dataset.
///
/// Useful when quantization techniques are used that require more processing work to
/// compare two compressed vectors.
///
template <typename Distance, typename VectorType> struct SelfDistance {
    using type = Distance;
    static constexpr type modify(const Distance& distance) {
        return threads::shallow_copy(distance);
    }
};

} // namespace svs
