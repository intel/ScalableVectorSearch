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
#include "svs/lib/exception.h"
#include "svs/lib/float16.h"
#include "svs/lib/misc.h"
#include "svs/third-party/fmt.h"
#include "svs/third-party/toml.h"

#include "svs/concepts/distance.h"
#include "svs/core/distance.h"
#include "svs/core/distance/euclidean.h"

#include "catch2/catch_approx.hpp"
#include "catch2/matchers/catch_matchers_templated.hpp"

#include "catch2/catch_test_macros.hpp"

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

inline std::filesystem::path prepare_temp_directory_v2() {
    cleanup_temp_directory();
    make_temp_directory();
    return temp_directory();
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

/////
///// Permute a range of indices.
/////

std::vector<uint64_t> permute_indices(size_t max_id);

/////
///// Matcher
/////

template <typename StringMatcher>
struct ExceptionMatcher : Catch::Matchers::MatcherGenericBase {
    // Type Aliases
    using matcher_type = StringMatcher;

    // Members
    matcher_type matcher_;

    // Methocs
    ExceptionMatcher(matcher_type&& matcher)
        : matcher_{std::move(matcher)} {}

    bool match(const svs::ANNException& exception) const {
        return matcher_.match(exception.what());
    }

    std::string describe() const override {
        return fmt::format("ANNException: {}", matcher_.describe());
    }
};

/////
///// TOML Lens
/////

// A utility for modifying TOML files to test loading failure.
struct Lens {
    ///// Members
    std::vector<std::string> key_chain_;
    std::unique_ptr<toml::node> value_;

    ///// Constructor
    template <typename T>
    Lens(std::initializer_list<std::string> key_chain, T&& value)
        : key_chain_(key_chain)
        , value_{toml::impl::make_node(SVS_FWD(value))} {
        if (key_chain_.empty()) {
            throw ANNEXCEPTION("Cannot create an empty keychain!");
        }
    }

    void apply(toml::table& table, bool expect_exists = true) const {
        return apply(&table, expect_exists);
    }

  private:
    void apply(toml::table* table, bool expect_exists = true) const;
};

void mutate_table(
    const std::filesystem::path& src,
    const std::filesystem::path& dst,
    std::initializer_list<Lens> lenses
);

/////
///// Distance
/////

// Test get_distance for any index, data type, and distance method
struct GetDistanceTester {
    template <typename IndexType>
    static void test(
        IndexType& index, 
        svs::DistanceType distance_type,
        const svs::data::ImmutableMemoryDataset auto& queries,
        const svs::data::ImmutableMemoryDataset auto& dataset
    ) { 
        // Test index vector id
        size_t id = 0;
        
        // Convert distance_type to a distance function
        svs::DistanceDispatcher dispatcher(distance_type);
        dispatcher([&](auto distance) {
            constexpr double TOLERANCE = 1e-5;
            
            // Use a query vector for testing
            if (queries.size() > 10 && index.size() > 0 && dataset.size() > id) {
                auto query_span = queries.get_datum(10);
                using ElementType = std::decay_t<decltype(query_span[0])>;
                std::vector<ElementType> query_vector(query_span.begin(), query_span.end());
                
                // Get distance from index
                double index_distance = index.get_distance(id, query_vector);
                
                // Get the vector directly from the dataset
                auto indexed_span = dataset.get_datum(id);
                
                // Compute expected distance
                auto dist_copy = distance;
                svs::distance::maybe_fix_argument(dist_copy, query_span);
                double expected_distance = svs::distance::compute(dist_copy, query_span, indexed_span);
                
                // Test the distance calculation
                CATCH_REQUIRE(std::abs(index_distance - expected_distance) < TOLERANCE);
            }
            
            // Test out of bounds ID
            if (index.size() > 0) {
                const size_t dim = index.dimensions();
                using DataType = float;  // Default to float
                std::vector<DataType> test_vector(dim, 1.0f);
                CATCH_REQUIRE_THROWS_AS(index.get_distance(index.size() + 100, test_vector), svs::ANNException);
            }
        });
    }
};
} // namespace svs_test
