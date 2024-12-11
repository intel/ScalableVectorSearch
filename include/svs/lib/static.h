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

#include <algorithm>
#include <functional>
#include <span>
#include <stdexcept>

#include "svs/lib/exception.h"
#include "svs/lib/misc.h"

namespace svs {
namespace lib {

///
/// @brief Class representing either a compile-time or run-time size.
///
/// @tparam N A static compile-time value.
///
/// If ``N != svs::Dynamic``, then this struct be empty, suitable for use with the
/// ``[[no_unique_address]]`` attribute to take no space inside other classes.
///
template <size_t N> class MaybeStatic {
  public:
    /// Construct a new instance of this class.
    constexpr MaybeStatic() = default;
    constexpr MaybeStatic(ZeroInitializer)
        : MaybeStatic() {}

    ///
    /// @brief Construct with checking.
    ///
    /// If ``size != N``, throws ``svs::ANNException``.
    ///
    constexpr explicit MaybeStatic(size_t size) {
        if (size != N) {
            throw ANNEXCEPTION(
                "Tying to construct a static length of value {} with a runtime value of "
                "{}!",
                N,
                size
            );
        }
    }

    /// Return the stored size.
    constexpr static size_t size() { return N; };
    constexpr operator size_t() const { return size(); }

    template <size_t Step> constexpr bool islast(size_t i) const {
        constexpr size_t last_iter = Step * (lib::div_round_up(N, Step) - 1);
        return i == last_iter;
    }
};

///
/// @brief Specilization of ``svs::lib::MaybeStatic<N>`` for runtime sizes.
///
/// Instances of this class will have a memory footprint of 8-bytes to store the
/// runtime size.
///
template <> class MaybeStatic<Dynamic> {
  public:
    /// Default constructor is deleted to avoid uninitialized sizes.
    MaybeStatic() = delete;
    constexpr explicit MaybeStatic(ZeroInitializer)
        : MaybeStatic(0) {}

    /// Construct a new instance with the given runtime size.
    constexpr explicit MaybeStatic(size_t size)
        : size_{size} {};

    /// Return the stored size.
    constexpr size_t size() const { return size_; };
    constexpr operator size_t() const { return size(); }

    template <size_t Step> constexpr bool islast(size_t i) const {
        const size_t last_iter = Step * (lib::div_round_up(size(), Step) - 1);
        return i == last_iter;
    }

  private:
    size_t size_;
};

template <size_t N, size_t M>
constexpr bool
operator==(const MaybeStatic<N>& /*unused*/, const MaybeStatic<M>& /*unused*/) {
    return N == M;
}

template <>
constexpr bool operator==(const MaybeStatic<Dynamic>& a, const MaybeStatic<Dynamic>& b) {
    return a.size() == b.size();
}

// Deduction guide to allow dynamic lengths to be constructed using: MaybeStaticlength(int)
template <typename T> MaybeStatic(T x) -> MaybeStatic<Dynamic>;

// Helper function for calling "islast" so we don't have a nasty `length.template islast`
// is code using the `MaybeStatic` type.
template <size_t Step, size_t N> constexpr bool islast(MaybeStatic<N> length, size_t i) {
    return length.template islast<Step>(i);
}
namespace detail {
// Compute the start of the last non-full iteration of dividing a space of size `size` in
// to `step` sized chunks.
constexpr size_t upper(size_t size, size_t step) { return step * (size / step); }
constexpr size_t rest(size_t size, size_t upper) { return size - upper; }

} // namespace detail

template <size_t N, size_t Step> struct ComputeTrailing {
    using rest_type = MaybeStatic<detail::rest(N, detail::upper(N, Step))>;
    static constexpr size_t upper(MaybeStatic<N> /*unused*/) {
        constexpr size_t u = detail::upper(N, Step);
        return u;
    }

    static constexpr rest_type rest(MaybeStatic<N> /*unused*/) { return {}; }
};

template <size_t Step> struct ComputeTrailing<Dynamic, Step> {
    using rest_type = MaybeStatic<Dynamic>;
    static constexpr size_t upper(MaybeStatic<Dynamic> length) {
        return detail::upper(length.size(), Step);
    }

    static constexpr rest_type rest(MaybeStatic<Dynamic> length) {
        return MaybeStatic{detail::rest(length.size(), detail::upper(length.size(), Step))};
    }
};

template <size_t Step, size_t N> constexpr size_t upper(MaybeStatic<N> length) {
    return ComputeTrailing<N, Step>::upper(length);
}

template <size_t Step, size_t N> constexpr auto rest(MaybeStatic<N> length) {
    return ComputeTrailing<N, Step>::rest(length);
}

/////
///// Finding Static Extents
/////

inline constexpr size_t extract_extent(const size_t x, const size_t y) {
    if (x == Dynamic) {
        return y;
    } else if (y == Dynamic || x == y) {
        return x;
    } else {
        // At this point, we have `M != N` and neither are `Dynamic`.
        throw std::logic_error(
            "Trying to propagate a single static extent from two different static extents!"
        );
    }
}

} // namespace lib
} // namespace svs
