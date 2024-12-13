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

// stdlib
#include <algorithm>
#include <cmath>

namespace svs::range {

///
/// Functors
///
template <typename T> class DivideBy {
  public:
    DivideBy(T value)
        : value{value} {}

    template <typename U> auto operator()(U u) { return u / value; }

  private:
    T value;
};

template <typename T> class MulBy {
  public:
    MulBy(T value)
        : value{value} {}

    template <typename U> auto operator()(U u) { return u * value; }

  private:
    T value;
};

class Inverse {
  public:
    Inverse() = default;
    template <typename U> auto operator()(U u) { return static_cast<U>(1) / u; }
};

class Sqrt {
  public:
    Sqrt() = default;
    template <typename U> auto operator()(U u) { return std::sqrt(u); }
};

///
/// transform
///
template <typename T, typename U, typename Op>
void transform(const T& in, U& out, const Op& op) {
    assert(in.size() == out.size());
    std::transform(in.begin(), in.end(), out.begin(), op);
}

template <typename T, typename Op> void transform(T& range, const Op& op) {
    transform(range, range, op);
}

///
/// Negate each element of the input range `in` into the output range `out`.
///
template <typename T, typename U> void negate(const T& in, U& out) {
    transform(in, out, std::negate());
}

///
/// Negate each element of the provided `range`.
///
template <typename T> void negate(T& range) { negate(range, range); }

///
/// Store the square root of each element of the input range `in` into the output
/// range `out`.
///
template <typename T, typename U> void sqrt(const T& in, U& out) {
    transform(in, out, Sqrt());
}

///
/// Perform the in-place square root of each element in `range`.
///
template <typename T> void sqrt(T& range) { sqrt(range, range); }

///
/// Store the multiplicative inverse of each element of the input range `in` into
/// the output range `out`.
///
template <typename T, typename U> void invert(const T& in, U& out) {
    transform(in, out, Inverse());
}

///
/// Perform in-place multiplicative inverse of each element in `range`.
///
template <typename T> void invert(T& range) { invert(range, range); }

} // namespace svs::range
