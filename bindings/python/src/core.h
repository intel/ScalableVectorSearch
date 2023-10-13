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

// quantization
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"
#include "svs/quantization/lvq/lvq.h"

// pybind
#include <pybind11/pybind11.h>

// stl
#include <filesystem>

// Type aliases
template <typename T> using Type = svs::meta::Type<T>;
template <size_t N> using Val = svs::meta::Val<N>;
template <auto V> using Const = svs::lib::Const<V>;
using svs::meta::unwrap;

// Exposed Allocators
// N.B.: As more allocators get implemented, this can be switched to a ``std::variant`` of
// allocators that will get propagated throughout the code.
//
// Support for this might not be fully in place but should be relatively straight-forward
// to add.
using Allocator = svs::HugepageAllocator<std::byte>;

// Functor to wrap an allocator inside a blocked struct.
inline constexpr auto as_blocked = [](const auto& allocator) {
    return svs::data::Blocked<std::decay_t<decltype(allocator)>>{allocator};
};

template <typename T>
using RebindAllocator = typename std::allocator_traits<Allocator>::rebind_alloc<T>;

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

/////
///// LVQ
/////

// Compressors - online compression of existing data
using LVQReloader = svs::quantization::lvq::Reload;

using LVQ8 = svs::quantization::lvq::ProtoLVQLoader<8, 0, Allocator>;
using LVQ4 = svs::quantization::lvq::ProtoLVQLoader<4, 0, Allocator>;
using LVQ4x4 = svs::quantization::lvq::ProtoLVQLoader<4, 4, Allocator>;
using LVQ4x8 = svs::quantization::lvq::ProtoLVQLoader<4, 8, Allocator>;
using LVQ8x8 = svs::quantization::lvq::ProtoLVQLoader<8, 8, Allocator>;

namespace core {
void wrap(pybind11::module& m);
} // namespace core
