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

#include <cmath>
#include <iostream>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "svs/lib/float16.h"
#include "svs/lib/narrow.h"

#include "tests/utils/utils.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_random.hpp"

// TODO:
// - Check down-conversion to either correctly convert a larger integer to a smaller one,
//   or throw an exception if such a conversion is not possible.
//
// The random number generators implemented by the Catch2 unit test library do not support
// 8-bit sized types.
//
// Thus, use the `CatchGenerator` type aliases to selectively convert small integers to
// larger intergers, relying on type conversion when they are inserted into the vectors.
//
// This also provides an entry point for intercepting `svs::float16` and doing that
// conversion as well.
namespace svs_test {
template <typename T> struct CatchGenerator {
    using type = T;
};
template <> struct CatchGenerator<uint8_t> {
    using type = uint32_t;
};
template <> struct CatchGenerator<int8_t> {
    using type = int32_t;
};
template <> struct CatchGenerator<svs::Float16> {
    using type = float;
};

// Convenience alias function.
template <typename T> using catch_generator_type_t = typename CatchGenerator<T>::type;
template <typename T, typename U> T convert_to(U x) { return svs::lib::narrow<T>(x); }

// `narrow` doesn't work for converting from `float` to `svs::Float16` because in this
// case we can't easily prevent loss of information.
//
// Fortunately in this case, it doesn't particularly matter.
template <> inline svs::Float16 convert_to(float x) { return svs::Float16{x}; }

// Conveniently convert a bound of type `T` to the appropriate type for use in one of
// the `Catch2` number generators.
template <typename U, typename T> catch_generator_type_t<U> generator_convert(T x) {
    return convert_to<catch_generator_type_t<U>>(x);
}

namespace detail {
// Catch2 does not expose an API for constructing a generator with a given seed, even though
// all the machinery is there.
//
// Here, we replicate much of the functionality in
// RandomIntegerGenerator
// Catch2/src/catch2/generator/catch_generators_random.hpp
// ```
// but expose the option to specify a seed.
template <typename T>
    requires std::is_floating_point_v<T>
Catch::Generators::GeneratorWrapper<T> random(T a, T b, uint32_t seed) {
    return Catch::Generators::GeneratorWrapper<T>(
        Catch::Detail::make_unique<Catch::Generators::RandomFloatingGenerator<T>>(
            a, b, seed
        )
    );
}

template <typename T>
    requires std::is_integral_v<T>
Catch::Generators::GeneratorWrapper<T> random(T a, T b, uint32_t seed) {
    return Catch::Generators::GeneratorWrapper<T>(
        Catch::Detail::make_unique<Catch::Generators::RandomIntegerGenerator<T>>(a, b, seed)
    );
}
} // namespace detail

// Construct a uniform random number generaALtor for data type `U` using the bounds
// `lo` and `hi`.
template <typename U, typename T> auto make_generator(T lo, T hi) {
    return Catch::Generators::random(generator_convert<U>(lo), generator_convert<U>(hi));
}

template <typename U, typename T> auto make_generator(T lo, T hi, uint32_t seed) {
    return detail::random(generator_convert<U>(lo), generator_convert<U>(hi), seed);
}

// Resize vector `v` to `length` and store a random number at each entry.
template <typename Generator> auto generate(Generator& generator) {
    generator.next();
    return generator.get();
}

template <typename T> struct Populator;

template <typename T> struct Populator<std::vector<T>> {
    template <typename Generator>
    static void populate(std::vector<T>& v, Generator&& generator, size_t length) {
        v.resize(length);
        std::for_each(v.begin(), v.end(), [&generator](T& x) {
            x = convert_to<T>(generate(generator));
        });
    }

    template <typename Generator>
    static void populate(std::vector<T>& v, Generator&& generator) {
        populate(v, std::forward<Generator>(generator), v.size());
    }
};

template <typename T> struct Populator<std::unordered_set<T>> {
    template <typename Generator>
    static void populate(std::unordered_set<T>& v, Generator&& generator, size_t length) {
        v.clear();
        for (size_t i = 0; i < length; ++i) {
            v.insert(convert_to<T>(generate(generator)));
        }
    }

    template <typename Generator>
    static void populate(std::unordered_set<T>& v, Generator&& generator) {
        populate(v, std::forward<Generator>(generator), v.size());
    }
};

template <typename T, typename Generator, typename... Args>
void populate(T& v, Generator&& generator, Args... args) {
    return Populator<T>::populate(
        v, std::forward<Generator>(generator), std::forward<Args>(args)...
    );
}
} // namespace svs_test
