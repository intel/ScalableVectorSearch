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

#include <functional>
#include <iostream>

#include "svs/lib/narrow.h"
#include "svs/lib/type_traits.h"

namespace svs {

///
/// Base type for representing `index-distance` pairs.
///
/// Often, search routines will want to add some extra state to an `index-distance` pair.
/// For example, traditional graph searches usually want to store an extra boolean flag
/// to indicate whether a particular neighbor has been expanded.
///
/// We would like to be able to add arbitrary metadata while perserving core functionality
/// like hashing, ordering etc.
///
/// Don't use `operator==()` to compare the equality of neighbor ids because the semantics
/// of equality don't play well with the ordering semantics which are based on distance.
///
/// Furthermore, all retrievals of the underlying `id` and `distance` should go through
/// the `id()` and `distance()` member functions. This allows implementations to bit-packing
/// if desired.
///
template <typename Idx, typename Meta = void> struct Neighbor : public Meta {
    using index_type = Idx;

    constexpr Neighbor() = default;

    template <typename... Args>
    constexpr Neighbor(Idx id, float distance, Args&&... parent_args)
        : Meta{std::forward<Args>(parent_args)...}
        , id_{id}
        , distance_{distance} {}

    // Provide an explicit conversion path from base neighbors.
    template <typename... Args>
    explicit constexpr Neighbor(Neighbor<Idx, void> other, Args&&... parent_args)
        : Meta{std::forward<Args>(parent_args)...}
        , id_{other.id()}
        , distance_{other.distance()} {}

    constexpr float distance() const { return distance_; }
    constexpr void set_distance(float new_distance) { distance_ = new_distance; }
    constexpr Idx id() const { return id_; }

    // Members
    Idx id_;
    float distance_;
};

template <typename Idx> struct Neighbor<Idx, void> {
    using index_type = Idx;

    constexpr Neighbor() = default;
    constexpr Neighbor(Idx id, float distance)
        : id_{id}
        , distance_{distance} {}

    /// Convert a Neighbor with metadata to a Neighbor without any metadata.
    template <typename OtherIdx, typename OtherMeta>
    constexpr Neighbor(const Neighbor<OtherIdx, OtherMeta>& other)
        : Neighbor(lib::narrow<Idx>(other.id()), other.distance()) {}

    constexpr float distance() const { return distance_; }
    constexpr Idx id() const { return id_; }

    // Members
    Idx id_;
    float distance_;
};

///
/// Return `true` if the distance stored in `x` is *less* than the distance stored in `y`.
/// Otherwise, return `false`.
///
template <typename Idx, typename Meta>
constexpr bool operator<(const Neighbor<Idx, Meta>& x, const Neighbor<Idx, Meta>& y) {
    return x.distance() < y.distance();
}

///
/// Return `true` if the distance stored in `x` is *greater* than the distance stored in
/// `y`. Otherwise, return `false`.
///
template <typename Idx, typename Meta>
constexpr bool operator>(const Neighbor<Idx, Meta>& x, const Neighbor<Idx, Meta>& y) {
    return x.distance() > y.distance();
}

///
/// Return `true` if `x` and `y` have the same ids.
///
template <typename Idx, typename Meta>
constexpr bool equal_id(const Neighbor<Idx, Meta>& x, const Neighbor<Idx, Meta>& y) {
    return x.id() == y.id();
}

///
/// Check if two neighbors are equal.
///
template <typename Idx, typename Meta>
constexpr bool equal(const Neighbor<Idx, Meta>& x, const Neighbor<Idx, Meta>& y) {
    return (x.id() == y.id()) && (x.distance() == y.distance()) &&
           static_cast<const Meta&>(x) == static_cast<const Meta&>(y);
}

// specialize for `void` since it doesn't have a parent.
template <typename Idx>
constexpr bool equal(const Neighbor<Idx, void>& x, const Neighbor<Idx, void>& y) {
    return (x.id() == y.id()) && (x.distance() == y.distance());
}

struct NeighborEqual {
    template <typename Idx, typename Meta>
    constexpr bool operator()(const Neighbor<Idx, Meta>& x, const Neighbor<Idx, Meta>& y) {
        return equal(x, y);
    }
};

template <typename Cmp> class TotalOrder {
  private:
    [[no_unique_address]] Cmp cmp_{};

  public:
    TotalOrder(const Cmp& cmp)
        : cmp_{cmp} {}

    template <typename Idx, typename Meta>
    constexpr bool operator()(const Neighbor<Idx, Meta>& x, const Neighbor<Idx, Meta>& y) {
        return cmp_(x, y) ? true : (cmp_(y, x) ? false : x.id() < y.id());
    }
};

///
/// Allow containers of `Neighbor<Idx>` to be initialized properly.
///
namespace type_traits {
template <typename Idx, typename Cmp> struct Sentinel<Neighbor<Idx>, Cmp> {
    static constexpr Neighbor<Idx> value =
        Neighbor<Idx>{std::numeric_limits<Idx>::max(), sentinel_v<float, Cmp>};
};
} // namespace type_traits

// Provide mechanisms for hashing and comparison.
class IDHash {
  public:
    template <typename Idx, typename Meta>
    size_t operator()(const Neighbor<Idx, Meta>& neighbor) const {
        return std::hash<Idx>()(neighbor.id());
    }
};

class IDEqual {
  public:
    template <typename Idx, typename Meta>
    size_t operator()(const Neighbor<Idx, Meta>& x, const Neighbor<Idx, Meta>& y) const {
        return equal_id(x, y);
    }
};

/////
///// NeighborLike Concept
/////

template <typename T> inline constexpr bool is_neighborlike_v = false;
template <typename Idx, typename Meta>
inline constexpr bool is_neighborlike_v<Neighbor<Idx, Meta>> = true;

template <typename T>
concept NeighborLike = is_neighborlike_v<T>;

/////
///// Search Neighbor
/////

struct Visited {
    constexpr Visited(bool visited = false)
        : visited_{visited} {}
    constexpr bool visited() const { return visited_; }
    constexpr void set_visited() { visited_ = true; }
    constexpr void clear_visited() { visited_ = false; }

    // members
    bool visited_;
};

inline constexpr bool operator==(Visited x, Visited y) {
    return x.visited() == y.visited();
}

///
/// Type alias for the new search neighbor
///
template <typename Idx> using SearchNeighbor = Neighbor<Idx, Visited>;

/////
///// ValidNeighbor
/////

///
/// Small metadata class to indicate whether a neighbor for graph search has been visited
/// and whether or not this neighbor should be included in the final results (valid)
/// or not.
///
/// Internally, we use a bit set to mark these states.
///
class ValidVisit {
  public:
    static constexpr uint8_t visited_mask = uint8_t{0x01};
    static constexpr uint8_t valid_mask = uint8_t{0x02};

    constexpr ValidVisit(bool valid = true)
        : value_{valid ? valid_mask : uint8_t{0x0}} {}

    constexpr void set_visited() { value_ |= visited_mask; }
    constexpr void clear_visited() { value_ &= ~visited_mask; }
    constexpr bool visited() const { return (value_ & visited_mask) != 0; }
    constexpr bool valid() const { return (value_ & valid_mask) != 0; }

    friend constexpr bool operator==(ValidVisit, ValidVisit) = default;

  private:
    uint8_t value_{0};
};

/// Type alias for skippable neighbor.
template <typename Idx> using PredicatedSearchNeighbor = Neighbor<Idx, ValidVisit>;

} // namespace svs
