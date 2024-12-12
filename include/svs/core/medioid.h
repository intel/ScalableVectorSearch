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

#include "svs/concepts/data.h"
#include "svs/lib/array.h"
#include "svs/lib/misc.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads.h"

// stl
#include <tuple>
#include <vector>

namespace svs::utils {

///
/// Parameters controlling the behavior of pairwise summation.
///
struct PairwiseSumParameters {
    PairwiseSumParameters() = default;
    PairwiseSumParameters(size_t linear_threshold)
        : linear_threshold{linear_threshold} {}
    PairwiseSumParameters(size_t linear_threshold, size_t thread_batchsize)
        : linear_threshold{linear_threshold}
        , thread_batchsize{thread_batchsize} {}

    // The threshold where recursive halving stops and a linear sum is computed instead.
    size_t linear_threshold = 1024;
    // Batch size for each thread.
    size_t thread_batchsize = 1'000'000;
};

struct CountSum {
  public:
    // Type aliases.
    using vector_type = std::vector<double>;
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;

    // Constructor
    CountSum(size_t ndimensions)
        : count{0}
        , sums(ndimensions, 0.0) {}

    // Iterators
    iterator begin() { return sums.begin(); }
    const_iterator begin() const { return sums.begin(); }
    const_iterator cbegin() const { return sums.cbegin(); }

    iterator end() { return sums.end(); }
    const_iterator end() const { return sums.end(); }
    const_iterator cend() const { return sums.cend(); }

    size_t size() const { return sums.size(); }

    ///
    /// Return a similar but uninitialized container.
    ///
    CountSum similar() const { return CountSum(size()); }

    CountSum& operator+=(const CountSum& other) {
        std::transform(begin(), end(), other.begin(), begin(), std::plus());
        count += other.count;
        return *this;
    }

    template <typename T> CountSum& add(const T& other) {
        assert(other.size() == size());
        std::transform(begin(), end(), other.begin(), begin(), std::plus());
        ++count;
        return *this;
    }

    std::vector<double> finish() const {
        std::vector<double> mean(size());
        std::transform(
            begin(),
            end(),
            mean.begin(),
            [&, count = lib::narrow_cast<double>(count)](double v) { return v / count; }
        );
        return mean;
    }

    // Members
    size_t count;
    std::vector<double> sums;
};

struct CountVariance {
  public:
    // Type aliases.
    using vector_type = std::vector<double>;
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;

    // Constructor
    CountVariance(const vector_type& means)
        : count{0}
        , means{std::make_shared<vector_type>(means)}
        , variances(means.size(), 0.0) {}

    CountVariance(const std::shared_ptr<vector_type>& means)
        : count{0}
        , means{means}
        , variances((*means).size(), 0.0) {}

    // Iterators
    iterator begin() { return variances.begin(); }
    const_iterator begin() const { return variances.begin(); }
    const_iterator cbegin() const { return variances.cbegin(); }

    iterator end() { return variances.end(); }
    const_iterator end() const { return variances.end(); }
    const_iterator cend() const { return variances.cend(); }

    size_t size() const { return variances.size(); }

    ///
    /// Return a similar but uninitialized container.
    ///
    CountVariance similar() const { return CountVariance{means}; }

    CountVariance& operator+=(const CountVariance& other) {
        assert(means->size() == other.means->size());
        assert(std::equal(means->begin(), means->end(), other.means->begin()));

        std::transform(begin(), end(), other.begin(), begin(), std::plus());
        count += other.count;
        return *this;
    }

    template <typename T> CountVariance& add(const T& other) {
        assert(other.size() == size());
        const auto& means_deref = *means;
        for (size_t i = 0; i < size(); ++i) {
            double temp = other[i] - means_deref[i];
            variances[i] += temp * temp;
        }
        ++count;
        return *this;
    }

    std::vector<double> finish() const {
        std::vector<double> averaged(size());
        std::transform(
            begin(),
            end(),
            averaged.begin(),
            [&, count = lib::narrow_cast<double>(count)](double v) { return v / count; }
        );
        return averaged;
    }

    // Members
    size_t count;
    std::shared_ptr<std::vector<double>> means;
    std::vector<double> variances;
};

template <
    data::ImmutableMemoryDataset Data,
    typename Op,
    typename Pred = lib::ReturnsTrueType,
    typename Map = lib::identity>
Op op_pairwise(
    const Data& data,
    const Op& op,
    const threads::UnitRange<size_t>& indices,
    const Pred& predicate = lib::ReturnsTrueType(),
    Map& map = lib::identity(), // <--- N.B.: lvalue reference. Maps may be statful!
    const PairwiseSumParameters& parameters = {}
) {
    // Haven't reached the bottom of recursion.
    // Divide the current range in half and recurse.
    if (indices.size() > parameters.linear_threshold) {
        size_t start = indices.start();
        size_t stop = indices.stop();
        size_t mid = (start + stop) / 2;
        Op left = op_pairwise(
            data, op, threads::UnitRange(start, mid), predicate, map, parameters
        );
        Op right = op_pairwise(
            data, op, threads::UnitRange(mid, stop), predicate, map, parameters
        );

        // Accumulate the sum components reusing the existing allocation for `left`.
        left += right;
        return left;
    }

    // End of recursion reached.
    // Perform a linear sum over the region.
    Op accum{op.similar()};
    for (const auto& i : indices) {
        if (!predicate(i)) {
            continue;
        }
        const auto& datum = data.get_datum(i);
        accum.add(map(datum));
    }
    return accum;
}

///
/// Compute the component-wise means of `dataset`, returning the result as a
/// `std::vector<double>`.
///
/// Optional argument `predicate` can be used to skip arbitray data points.
/// Only indices `i` where `predicate(i)` returns `true` will be accumulated in the
/// medioid computation.
///
/// Implementation Notes: Pairwise summation is used to provide better numeric
/// accuracy than naive summation.
///
/// See: https://en.wikipedia.org/wiki/Pairwise_summation
///
template <
    data::ImmutableMemoryDataset Data,
    typename Op,
    threads::ThreadPool Pool,
    typename Pred = lib::ReturnsTrueType,
    typename Map = lib::identity>
std::vector<double> op_pairwise(
    const Data& data,
    const Op& op,
    Pool& threadpool,
    Pred&& predicate = lib::ReturnsTrueType(),
    Map&& map = lib::identity(),
    PairwiseSumParameters parameters = {}
) {
    size_t batchsize = parameters.thread_batchsize;

    // Threaded run.
    threads::SequentialTLS<Op> tls{op.similar(), threadpool.size()};
    threads::run(
        threadpool,
        threads::DynamicPartition(data.size(), batchsize),
        [&](const auto& indices, uint64_t tid) {
            threads::UnitRange range{indices};
            auto map_local = map;
            tls.at(tid) += op_pairwise(data, op, range, predicate, map_local, parameters);
        }
    );

    // Merge per-thread results.
    Op accum{op.similar()};
    tls.visit([&](const Op& partial_accum) {
        assert(partial_accum.size() == accum.size());
        accum += partial_accum;
    });
    return accum.finish();
}

template <
    data::ImmutableMemoryDataset Data,
    threads::ThreadPool Pool,
    typename Pred = lib::ReturnsTrueType,
    typename Map = lib::identity>
std::vector<double> compute_medioid(
    const Data& data,
    Pool& threadpool,
    Pred&& predicate = lib::ReturnsTrueType(),
    Map&& map = lib::identity(),
    PairwiseSumParameters parameters = {}
) {
    return op_pairwise(
        data, CountSum(data.dimensions()), threadpool, predicate, map, parameters
    );
}

template <
    data::ImmutableMemoryDataset Data,
    threads::ThreadPool Pool,
    typename Map = lib::identity,
    typename Pred = lib::ReturnsTrueType>
size_t find_medioid(
    const Data& data,
    Pool& threadpool,
    Pred&& predicate = lib::ReturnsTrueType(),
    Map&& map = lib::identity(),
    const PairwiseSumParameters& parameters = {}
) {
    // Compute the medioid.
    std::vector<double> medioid =
        compute_medioid(data, threadpool, predicate, map, parameters);

    // Find the closest element to the medioid that satisfies the predicate.
    // Thread local nearest neighbors.
    threads::SequentialTLS<Neighbor<size_t>> closest_neighbors(
        type_traits::sentinel_v<Neighbor<size_t>, std::less<>>, threadpool.size()
    );

    threads::run(
        threadpool,
        threads::StaticPartition{data.size()},
        [&](const auto& ids, uint64_t tid) {
            auto& best = closest_neighbors.at(tid);
            auto map_local = map;
            for (auto i : ids) {
                if (!predicate(i)) {
                    continue;
                }

                double distance = 0;
                const auto& datum = data.get_datum(i);
                const auto& mapped = map_local(datum);
                assert(datum.size() == medioid.size());
                for (size_t k = 0, upper = mapped.size(); k < upper; ++k) {
                    double diff = medioid[k] - static_cast<double>(mapped[k]);
                    distance += diff * diff;
                }

                if (distance < best.distance()) {
                    best = {i, lib::narrow_cast<float>(distance)};
                }
            }
        }
    );

    // Return the minimum index.
    auto global_min = type_traits::sentinel_v<Neighbor<size_t>, std::less<>>;
    closest_neighbors.visit([&](const auto& neighbor) {
        if (neighbor < global_min) {
            global_min = neighbor;
        }
    });
    return global_min.id();
}

///
/// @brief Return the index of the medoid of the dataset.
///
template <data::ImmutableMemoryDataset Data, typename... Args>
size_t find_medioid(const Data& data, size_t num_threads, Args&&... args) {
    auto threadpool = threads::NativeThreadPool(num_threads);
    return find_medioid(data, threadpool, std::forward<Args>(args)...);
}

} // namespace svs::utils
