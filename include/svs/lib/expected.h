/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
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
