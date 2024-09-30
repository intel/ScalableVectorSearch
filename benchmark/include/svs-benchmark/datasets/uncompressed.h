/*
 * Copyright (C) 2024 Intel Corporation
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
