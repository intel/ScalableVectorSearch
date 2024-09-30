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

/**
 *    Copyright (C) Microsoft Corporation and Intel Corporation
 *
 *    The code in this file is a modified version of code from Microsoft Corporation,
 *    published under an MIT License. This modified version is licensed under the
 *    GNU Affero General Public License version 3
 *
 *    ORIGINAL LICENSE
 *    ------------------------------------------------------------------------------
 *    Copyright (c) 2015 Microsoft Corporation. All rights reserved.
 *
 *    This code is licensed under the MIT License (MIT).
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *    THE SOFTWARE.
 *    ------------------------------------------------------------------------------
 *    Original URL: https://github.com/microsoft/GSL/blob/main/include/gsl/narrow
 *
 *    MODIFIED CODE LICENSE
 *    ------------------------------------------------------------------------------
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 *    ------------------------------------------------------------------------------
 */

#pragma once

#include <exception> // for std::exception
#include <iostream>

#include "svs/lib/preprocessor.h"
#include "svs/lib/type_traits.h"

namespace svs {
namespace lib {

struct narrowing_error : public std::exception {
    const char* what() const noexcept override { return "narrowing_error"; }
};

// Explicitly expresses that narrowing is either acceptable or known impossible.
template <typename T, typename U> constexpr T narrow_cast(U&& u) noexcept {
    return static_cast<T>(std::forward<U>(u));
}

namespace detail {
// Use the `SVS_CLANG_NOINLINE` attribute to avoid inlining this function if building
// with clang in a non-debug context.
//
// This is because clang will detect undefined behavior at compile time for invalid
// conversions (for example, converting `std::numeric_limits<size_t>::max() - 1` to float)
// and optimize out the throwing branch, leading to a silent failure.
//
// Using the `[[gnu::noinline]]` attribute prevents compile-time constant propagation.
//
// Put this on an internal function to avoid pessimizing the case where the type being
// converted to is the same as the original type.
template <class T, class U>
SVS_CLANG_NOINLINE constexpr T narrow_impl(U u) noexcept(false) {
    static_assert(!std::is_same_v<T, std::remove_cv_t<U>>);
    constexpr bool is_different_signedness = (svs::is_signed_v<T> != svs::is_signed_v<U>);
    const T t = narrow_cast<T>(u);

    if (static_cast<U>(t) != u || (is_different_signedness && ((t < T{}) != (u < U{})))) {
        throw narrowing_error{};
    }
    return t;
}
} // namespace detail

// narrow() : a checked version of narrow_cast() that throws if the cast changed the value
template <class T, class U>
    requires svs::is_arithmetic_v<T>
constexpr T narrow(U u) noexcept(false) {
    if constexpr (std::is_same_v<T, U>) {
        return u;
    } else {
        return detail::narrow_impl<T>(u);
    }
}

template <class T, class U>
    requires(!svs::is_arithmetic_v<T>)
constexpr T narrow(U u) noexcept(false) {
    if constexpr (std::is_same_v<T, U>) {
        return u;
    } else {
        const T t = narrow_cast<T>(u);
        if (static_cast<U>(t) != u) {
            throw narrowing_error{};
        }
        return t;
    }
}

/// Behaves like `narrow` but can be lossy is `allow_lossy_conversion<T, U> == true`.
template <class T, class U>
    requires svs::is_arithmetic_v<T>
constexpr T relaxed_narrow(U u) noexcept(false) {
    if constexpr (allow_lossy_conversion<U, T>) {
        return narrow_cast<T>(u);
    } else {
        return narrow<T>(u);
    }
}

} // namespace lib
} // namespace svs
