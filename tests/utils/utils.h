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

#include <cassert>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include "svs/core/data/simple.h"
#include "svs/core/query_result.h"
#include "svs/lib/float16.h"
#include "svs/lib/misc.h"
#include "svs/third-party/fmt.h"

#include "svs/concepts/distance.h"
#include "svs/core/distance/euclidean.h"

#include "catch2/catch_approx.hpp"

namespace svs_test {

/////
///// File System
/////

// The macro `SVS_TEST_DATA_DIR` is defined in CMake build system.
inline std::filesystem::path data_directory() { return SVS_TEST_DATA_DIR; }

inline std::filesystem::path temp_directory() { return data_directory() / "temp"; }

inline bool cleanup_temp_directory() {
    return std::filesystem::remove_all(temp_directory());
}

inline bool make_temp_directory() {
    return std::filesystem::create_directory(temp_directory());
}

inline bool prepare_temp_directory() {
    cleanup_temp_directory();
    return make_temp_directory();
}

// Check if the contents of two files are identical.
bool compare_files(const std::string& a, const std::string& b);

/////
///// Promote to Float64
/////

template <typename T> double promote(T x) { return static_cast<double>(x); }

/////
///// Timed run of a function.
/////

template <typename F> double timed(size_t repeats, bool ignore_first, F&& f) {
    if (ignore_first) {
        f();
    }

    auto tic = std::chrono::steady_clock::now();
    for (size_t i = 0; i < repeats; ++i) {
        f();
    }
    auto toc = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(toc - tic).count();
}

/////
///// Stateful Euclidean
/////

// The main idea is to introduce a stateful of the basic euclidean distance type.
// Many of the algorithms in the code base should work with both stateless and stateful
// distances, occaisonally with optimizations implemented for the stateless functions.
//
// Introducing a stateful version lets us test the stateful paths.
template <typename T> struct StatefulL2 {
    using compare = std::less<>;

    // Cache a local copy of `data`.
    void fix_argument(std::span<const T> data) {
        data_.resize(data.size());
        std::copy(data.begin(), data.end(), data_.begin());
    }

    // Fallback to the generic implementation of Euclidean distance using our local cached
    // copy of the query.
    template <typename Eb, size_t Db> float compute(std::span<Eb, Db> other) {
        return svs::distance::compute(
            svs::distance::DistanceL2{}, std::span{data_.data(), data_.size()}, other
        );
    }

    // Members
    std::vector<T> data_;
};
static_assert(svs::distance::ShouldFix<StatefulL2<float>, std::span<const float>>);

/////
///// Type Utilities
/////

template <typename T, typename U>
bool isapprox_or_warn(T x, U y, double epsilon = 0.0, double margin = 0.0) {
    bool isapprox = (x == Catch::Approx(y).epsilon(epsilon).margin(margin));
    if (!isapprox) {
        fmt::print("Approcimate comparison failed with values ({}, {})\n", x, y);
    }
    return isapprox;
}

template <auto V> struct Val {};

template <typename T> struct TypeName {};

template <auto V> struct TypeName<Val<V>> {
    static std::string name() { return std::to_string(V); };
};

template <> struct TypeName<uint8_t> {
    static std::string name() { return "uint8"; };
};
template <> struct TypeName<uint16_t> {
    static std::string name() { return "uint16"; };
};
template <> struct TypeName<uint32_t> {
    static std::string name() { return "uint32"; };
};
template <> struct TypeName<uint64_t> {
    static std::string name() { return "uint64"; };
};

template <> struct TypeName<int8_t> {
    static std::string name() { return "int8"; };
};
template <> struct TypeName<int16_t> {
    static std::string name() { return "int16"; };
};
template <> struct TypeName<int32_t> {
    static std::string name() { return "int32"; };
};
template <> struct TypeName<int64_t> {
    static std::string name() { return "int64"; };
};

template <> struct TypeName<svs::Float16> {
    static std::string name() { return "float16"; };
};
template <> struct TypeName<float> {
    static std::string name() { return "float32"; };
};
template <> struct TypeName<double> {
    static std::string name() { return "float64"; };
};

template <typename T> std::string type_name() { return TypeName<T>::name(); }
} // namespace svs_test
