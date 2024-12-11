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

// svs python bindings
#include "svs/python/common.h"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"

// pybind
#include <pybind11/pybind11.h>

// stl
#include <filesystem>

namespace svs::python {

// Type aliases
template <typename T> using Type = svs::lib::Type<T>;
template <size_t N> using Val = svs::lib::Val<N>;
template <auto V> using Const = svs::lib::Const<V>;

// Tagged type to enable dispatch conversion.
struct ReloadFile {
    std::filesystem::path path_;
};

// Introduce a mechanism for transporting numpy arrays through the dispatcher interface.
struct AnonymousVectorData {
  public:
    svs::AnonymousArray<2> array_;

  public:
    // Constructor
    template <typename T>
    AnonymousVectorData(const py_contiguous_array_t<T>& array)
        : array_{
              array.template unchecked<2>().data(0, 0),
              svs::lib::narrow<size_t>(array.shape(0)),
              svs::lib::narrow<size_t>(array.shape(1))} {}

    // Interface.
    svs::DataType type() const { return array_.type(); }
    size_t size() const { return array_.size(0); }
    size_t dimensions() const { return array_.size(1); }
    svs::AnonymousArray<2> underlying() const { return array_; }
};

} // namespace svs::python

template <typename T, size_t N>
struct svs::lib::DispatchConverter<
    svs::python::AnonymousVectorData,
    svs::data::ConstSimpleDataView<T, N>> {
    static int64_t match(const svs::python::AnonymousVectorData& data) {
        // Types *must* match in order to be compatible.
        if (data.type() != svs::datatype_v<T>) {
            return svs::lib::invalid_match;
        }

        // Use default extent-matching semantics.
        return svs::lib::dispatch_match<svs::lib::ExtentArg, svs::lib::ExtentTag<N>>(
            svs::lib::ExtentArg(data.dimensions())
        );
    }

    static svs::data::ConstSimpleDataView<T, N>
    convert(svs::python::AnonymousVectorData data) {
        return svs::data::ConstSimpleDataView<T, N>(data.underlying());
    }
};

namespace svs::python {

// Standard loaders.
using UnspecializedVectorDataLoader = svs::UnspecializedVectorDataLoader<Allocator>;

class UnspecializedGraphLoader {
  public:
    UnspecializedGraphLoader() = delete;
    UnspecializedGraphLoader(const std::filesystem::path& path)
        : path_{path} {}

    const std::filesystem::path& path() const { return path_; }
    const Allocator& allocator() const { return allocator_; }

    svs::graphs::SimpleGraph<uint32_t> load() const {
        using other = std::allocator_traits<Allocator>::rebind_alloc<uint32_t>;
        return svs::graphs::SimpleGraph<uint32_t, other>::load(path_, other(allocator_));
    }

  private:
    std::filesystem::path path_;
    Allocator allocator_{};
};

// Distance Aliases
using DistanceL2 = svs::distance::DistanceL2;
using DistanceIP = svs::distance::DistanceIP;

namespace core {
void wrap(pybind11::module& m);
} // namespace core
} // namespace svs::python
