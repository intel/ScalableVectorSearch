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

// svs
#include "svs/lib/datatype.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"

#include "svs/third-party/fmt.h"

// stl
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace svs {
namespace threads {

///
/// Number of threads participating in a parallel job.
///
struct ThreadCount {
    uint64_t value;
    constexpr explicit operator size_t() const { return value; }
};

///
/// Thread Exception. Thrown to indicate that something crashed while in a threading run.
///
class ThreadingException : public std::runtime_error {
  public:
    explicit ThreadingException(const std::string& message)
        : std::runtime_error{message} {};
};

///
/// Iterator Pair
///

template <typename I>
concept PartitionableIterator =
    requires {
        // `I` must be a random access iterator.
        requires std::random_access_iterator<I>;

        // Furthermore, the difference type must "play nicely" with integers.
        requires std::convertible_to<std::iter_difference_t<I>, size_t>;
        requires std::convertible_to<uint64_t, std::iter_difference_t<I>>;
        requires std::convertible_to<int64_t, std::iter_difference_t<I>>;
    };

template <std::random_access_iterator I> struct IteratorPair : std::pair<I, I> {
    // type aliases
    using difference_type = std::iter_difference_t<I>;
    using reference = std::iter_reference_t<I>;

    // constructor
    IteratorPair(I begin, I end)
        : std::pair<I, I>{begin, end} {}

    I begin() const { return this->first; }
    I end() const { return this->second; }

    reference front() const { return *begin(); }
    reference back() const { return *(end() - 1); }

    reference operator[](difference_type n) const { return begin()[n]; }

    size_t size() const { return lib::narrow<size_t>(std::distance(begin(), end())); }

    bool empty() const { return size() == 0; }
};

template <std::random_access_iterator I> IteratorPair(I, I) -> IteratorPair<I>;

///
/// The ID of a thread in a pool.
/// Guarenteed to be in the half open interval `[0,pool.size())` and unique to each
/// worker thread in the pool.
///
class SequentialThreadID {
  private:
    uint64_t id_;

  public:
    SequentialThreadID() = default;
    explicit SequentialThreadID(uint64_t id)
        : id_{id} {}

    explicit operator uint64_t() const { return id_; }
};

template <std::integral T = uint64_t> class IndexIterator {
  public:
    // Iterator Traits
    using difference_type = std::make_signed_t<T>;
    using value_type = T;
    // Do not define `pointer` since we return dereferenced items by value.
    // using pointer = void;

    // No difference between the `reference` and `const_reference` types because we return
    // by value.
    using reference = value_type;
    using const_reference = value_type;
    using iterator_category = std::random_access_iterator_tag;

    // Constructors
    IndexIterator() = default;
    explicit IndexIterator(T value)
        : value_{value} {}

    // Forward Iterator
    constexpr const_reference operator*() const { return value_; }
    constexpr IndexIterator& operator++() {
        ++value_;
        return *this;
    }
    constexpr IndexIterator operator++(int) {
        IndexIterator tmp{value_};
        ++value_;
        return tmp;
    }

    // Bidirectional Iterator
    constexpr IndexIterator& operator--() {
        --value_;
        return *this;
    }

    constexpr IndexIterator operator--(int) {
        IndexIterator tmp{value_};
        --value_;
        return tmp;
    }

    // Partial fulfillment for `std::sized_sentinel_for`.
    constexpr difference_type operator-(IndexIterator i) const {
        return static_cast<difference_type>(value_) - static_cast<difference_type>(*i);
    }

    // Random Access Iterator
    constexpr IndexIterator& operator+=(difference_type n) {
        value_ += n;
        return *this;
    }

    constexpr IndexIterator operator+(difference_type n) const {
        return IndexIterator{value_ + n};
    }

    constexpr IndexIterator& operator-=(difference_type n) {
        value_ -= n;
        return *this;
    }

    constexpr IndexIterator operator-(difference_type n) const {
        return IndexIterator{value_ - n};
    }

    constexpr reference operator[](difference_type n) const { return value_ + n; }

    // Spaceship Operator!!
    friend constexpr auto operator<=>(IndexIterator, IndexIterator) = default;

  private:
    T value_{0};
};

// Bidirectional Iterator Misc Methods.
template <typename I> constexpr IndexIterator<I> operator+(int64_t n, IndexIterator<I> j) {
    return IndexIterator<I>{n + *j};
}

template <typename I> constexpr IndexIterator<I> operator-(int64_t n, IndexIterator<I> j) {
    return IndexIterator<I>{n - *j};
}

/////
///// Concept Fulfillment
/////

static_assert(std::forward_iterator<IndexIterator<uint64_t>>);
static_assert(std::bidirectional_iterator<IndexIterator<uint64_t>>);
static_assert(std::random_access_iterator<IndexIterator<uint64_t>>);

///
/// Lightweight random-access container representing the half-open interval `[start, stop)`.
///
template <typename T>
    requires std::is_integral_v<T>
class UnitRange {
  private:
    // Invariants:
    // * stop_ >= start_
    T start_{0};
    T stop_{0};

  public:
    // Container
    using value_type = T;
    using reference = const T&; // Contents of range are immutable.
    using const_reference = const T&;
    using iterator = IndexIterator<T>;
    using const_iterator = IndexIterator<T>;
    using difference_type = int64_t;
    using size_type = uint64_t;

    // Constructors
    UnitRange() = default;
    UnitRange(T start, T stop)
        : start_{start}
        , stop_{std::max(stop, start)} {}

    UnitRange(const IteratorPair<IndexIterator<T>>& pair)
        : UnitRange{pair.front(), pair.back() + T(1)} {}

    const_iterator cbegin() const { return IndexIterator<T>{start_}; }
    const_iterator begin() const { return cbegin(); }

    const_iterator cend() const { return IndexIterator<T>{stop_}; }
    const_iterator end() const { return cend(); }

    bool operator==(UnitRange y) const {
        return (start_ == y.start_) && (stop_ == y.stop_);
    }
    bool operator!=(UnitRange y) const {
        return (start_ != y.start_) || (stop_ != y.stop_);
    }

    size_type size() const { return stop_ - start_; }
    static size_type max_size() { return std::numeric_limits<size_type>::max(); }
    bool empty() const { return start_ == stop_; }

    // Random Access.
    T operator[](size_t i) const { return start_ + i; }
    T at(size_t i) const {
        if (i >= size()) {
            auto message = "Index " + std::to_string(i) + " is out of range " +
                           std::to_string(size()) + ".";
            throw std::out_of_range(message);
        }
        return operator[](i);
    }

    ///
    /// Return `true` if the value `i` is in `[front(), back()]`. Otherwise, return `false`.
    ///
    bool contains(const T& i) const { return front() <= i && i <= back(); }

    ///
    /// Return a unit range over the valid indices of the current range.
    ///
    UnitRange<size_t> eachindex() const { return UnitRange<size_t>{0, size()}; }

    T front() const { return start_; }
    T back() const { return stop_ - 1; }

    T start() const { return start_; }
    T stop() const { return stop_; }
};

// Deduction guides.
template <typename T>
    requires std::is_integral_v<T>
UnitRange(T, T) -> UnitRange<T>;

// clang-format off
template <typename Start, typename Stop>
    requires std::is_integral_v<Start> && std::is_integral_v<Stop>
UnitRange(Start, Stop) -> UnitRange<std::common_type_t<Start, Stop>>;

template<typename T>
    requires std::is_integral_v<T>
UnitRange(IteratorPair<IndexIterator<T>>) -> UnitRange<T>;
// clang-format on

template <typename T>
std::ostream& operator<<(std::ostream& stream, const UnitRange<T>& r) {
    return stream << fmt::format("{}", r);
}

///
/// Partition up an iteration domain of size `n` for thread `tid` among a team of
/// size `nthreads`.
///
/// Return a `UnitRange` for this thread's start and stop points.
///
/// Credit to `https://github.com/oneapi-src/oneDNN/` for the `balance211`
/// algorithm.
///
template <std::integral T, std::integral U> UnitRange<T> balance(T n, U nthreads, U tid) {
    // If the iteration space is empty or the team size is 1, than the partition
    // is the whole space.
    if (nthreads <= 1 || n == 0) {
        return UnitRange<T>(0, n);
    }
    // * b1: The primary batch size to use to partition the work.
    // * b2: Slightly smaller batch size to use to more optimally assign work in the
    //       case where `nthreads` doesn't evenly divide `n`.
    T b1 = lib::div_round_up(n, nthreads);
    T b2 = b1 - 1;

    // The number of threads using batchsize `p1`.
    T team1 = n - b2 * nthreads;
    bool in_team1 = static_cast<T>(tid) < team1;
    T this_b = in_team1 ? b1 : b2;
    T start = in_team1 ? (b1 * tid) : ((b1 * team1) + b2 * (tid - team1));
    return UnitRange<T>(start, std::min(start + this_b, n));
}

// Helper to handle mismatched types.
template <std::integral T, std::integral U, std::integral V>
UnitRange<T> balance(T n, U nthreads, V tid) {
    return balance(n, nthreads, lib::narrow<U>(tid));
}

/////
///// Schedules
/////

template <PartitionableIterator I> struct StaticPartition : public IteratorPair<I> {
    using parent_type = IteratorPair<I>;

    ///
    /// Construct a static partition directly from an iterator pair.
    ///
    explicit StaticPartition(parent_type pair)
        : parent_type{std::move(pair)} {}

    ///
    /// Construct a static partition of the sequence of numbers `[0, length)`.
    ///
    template <std::integral T>
    explicit StaticPartition(T length)
        : parent_type{IndexIterator<T>{0}, IndexIterator<T>{length}} {}

    ///
    /// Construct a static partition of the sequence of numbers `[start, stop)`.
    ///
    template <std::integral T>
    StaticPartition(T start, T stop)
        : parent_type{IndexIterator{start}, IndexIterator{stop}} {}

    ///
    /// Construct a static partition of the random access range `range`.
    ///
    template <typename /*std::ranges::random_access_range*/ R>
        requires(!std::integral<R>) // Needed since clang-13 doesn't support ranges
    explicit StaticPartition(const R& range)
        : parent_type{std::begin(range), std::end(range)} {}
};

// Deduction guides
template <std::integral T> StaticPartition(T) -> StaticPartition<IndexIterator<T>>;
template <std::integral T> StaticPartition(T, T) -> StaticPartition<IndexIterator<T>>;
// template <std::ranges::random_access_range R>
// StaticPartition(const R&) -> StaticPartition<std::ranges::iterator_t<const R>>;
template <typename /*std::ranges::random_access_range*/ R>
    requires(!std::integral<R>)
StaticPartition(const R&) -> StaticPartition<typename R::const_iterator>;

template <PartitionableIterator I> struct DynamicPartition : public IteratorPair<I> {
    using parent_type = IteratorPair<I>;

    ///
    /// Construct a static partition directly from an iterator pair.
    ///
    DynamicPartition(parent_type pair)
        : parent_type{std::move(pair)} {}

    ///
    /// Construct a static partition of the sequence of numbers `[0, length)`.
    ///
    template <std::integral T>
    explicit DynamicPartition(T length, size_t grainsize)
        : parent_type{IndexIterator<T>{0}, IndexIterator<T>{length}}
        , grainsize{grainsize} {}

    ///
    /// Construct a static partition of the sequence of numbers `[start, stop)`.
    ///
    template <std::integral T>
    DynamicPartition(T start, T stop, size_t grainsize)
        : parent_type{IndexIterator{start}, IndexIterator{stop}}
        , grainsize{grainsize} {}

    ///
    /// Construct a static partition of the random access range `range`.
    ///
    template <typename /*std::ranges::random_access_range*/ R>
        requires(!std::integral<R>) // Needed because clang-12 doesn't support ranges.
    DynamicPartition(const R& range, size_t grainsize)
        : parent_type{std::begin(range), std::end(range)}
        , grainsize{grainsize} {}

    // Members
    uint64_t grainsize;
};

template <std::integral T>
DynamicPartition(T, size_t) -> DynamicPartition<IndexIterator<T>>;
template <std::integral T>
DynamicPartition(T, T, size_t) -> DynamicPartition<IndexIterator<T>>;
// template <std::ranges::random_access_range R>
// DynamicPartition(const R&, size_t) -> DynamicPartition<std::ranges::iterator_t<const R>>;
template <typename /*std::ranges::random_access_range*/ R>
    requires(!std::integral<R>)
DynamicPartition(const R&, size_t) -> DynamicPartition<typename R::const_iterator>;

} // namespace threads
} // namespace svs

///// Formatting
template <typename T>
struct fmt::formatter<svs::threads::UnitRange<T>> : svs::format_empty {
    auto format(const svs::threads::UnitRange<T>& x, auto& ctx) const {
        return fmt::format_to(
            ctx.out(), "UnitRange<{}>({}, {})", svs::datatype_v<T>, x.start(), x.stop()
        );
    }
};
