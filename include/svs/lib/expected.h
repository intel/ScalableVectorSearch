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

// An imperfect, not particularly efficient stand-in for C++23's ``std::expected``.
#include "svs/lib/exception.h"

#include <cassert>
#include <variant>

namespace svs::lib {

template <typename E> class Unexpected {
  public:
    Unexpected() = delete;

    // Construction
    explicit Unexpected(const E& value)
        : value_{value} {}
    explicit Unexpected(E&& value)
        : value_{std::move(value)} {}

    // Access
    const E& value() const& { return value_; }
    E& value() & { return value_; }
    E&& value() && { return std::move(value_); }

    friend bool operator==(const Unexpected&, const Unexpected&) = default;
    friend auto operator<=>(const Unexpected&, const Unexpected&) = default;

  private:
    E value_;
};

// deduction-guide.
template <typename E> Unexpected(E) -> Unexpected<E>;

template <typename T, typename E> class Expected {
  public:
    using value_type = T;
    using error_type = E;
    using unexpected_type = lib::Unexpected<E>;

    // Delete the default-constructor since it is not needed for our purposes.
    Expected() = delete;

    // non-explicit construction from value or unexpected types.
    Expected(const T& value)
        : value_{std::in_place_type<T>, value} {}
    Expected(T&& value)
        : value_{std::in_place_type<T>, std::move(value)} {}
    Expected(const Unexpected<E>& error)
        : value_{std::in_place_type<E>, error.value()} {}
    Expected(Unexpected<E>&& error)
        : value_{std::in_place_type<E>, std::move(error).value()} {}

    // Check whether *this contains an expected value.
    bool has_value() const { return value_.index() == 0; }

    // Contextual boolean conversion.
    explicit operator bool() const { return has_value(); }

    // Dereference operators.
    const T* operator->() const& {
        assert(has_value());
        return std::get_if<0>(&value_);
    }

    const T& operator*() const& {
        assert(has_value());
        return std::get<0>(value_);
    }

    // Checked access.
    const T& value() const& {
        if (!has_value()) {
            throw ANNEXCEPTION("Bad Expected Access to Value!");
        }
        return std::get<0>(value_);
    }
    T&& value() && {
        if (!has_value()) {
            throw ANNEXCEPTION("Bad Expected Access to Value!");
        }
        return std::get<0>(std::move(value_));
    }

    const E& error() const& {
        if (has_value()) {
            throw ANNEXCEPTION("Bad Expected Access to Error!");
        }
        return std::get<1>(value_);
    }
    E&& error() && {
        if (has_value()) {
            throw ANNEXCEPTION("Bad Expected Access to Error!");
        }
        return std::get<1>(std::move(value_));
    }

  private:
    std::variant<T, E> value_;
};

} // namespace svs::lib
