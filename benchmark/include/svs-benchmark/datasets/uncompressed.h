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

#include "svs-benchmark/datasets.h"

#include "svs/lib/dispatcher.h"

namespace svsbenchmark {

/// A fully typed - uncompressed dataset used as the target for dispatch conversion.
template <typename T> struct TypedUncompressed {};

} // namespace svsbenchmark

namespace svs::lib {

template <typename T>
struct DispatchConverter<svsbenchmark::Uncompressed, svsbenchmark::TypedUncompressed<T>> {
    static bool match(const svsbenchmark::Uncompressed& x) {
        return x.data_type_ == svs::datatype_v<T>;
    }

    static svsbenchmark::TypedUncompressed<T>
    convert([[maybe_unused]] svsbenchmark::Uncompressed x) {
        assert(match(x));
        return svsbenchmark::TypedUncompressed<T>();
    }

    static std::string description() {
        return fmt::format("uncompressed ({})", svs::datatype_v<T>);
    }
};

template <typename T>
struct DispatchConverter<svsbenchmark::Dataset, svsbenchmark::TypedUncompressed<T>> {
    using From = svsbenchmark::Uncompressed;
    using To = svsbenchmark::TypedUncompressed<T>;
    static int64_t match(const svsbenchmark::Dataset& x) {
        if (auto* ptr = std::get_if<From>(&x.kinds_)) {
            return svs::lib::dispatch_match<From, To>(*ptr);
        }
        return svs::lib::invalid_match;
    }
    static To convert(svsbenchmark::Dataset x) {
        return svs::lib::dispatch_convert<From, To>(std::get<From>(std::move(x.kinds_)));
    }

    static std::string description() { return svs::lib::dispatch_description<From, To>(); }
};
} // namespace svs::lib
