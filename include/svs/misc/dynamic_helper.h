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

///
/// @file dynamic_helper.h
///
/// A simple helper for testing dynamic indexes.
///

#pragma once

// svs
#include "svs/core/logging.h"
#include "svs/lib/algorithms.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads.h"
#include "svs/lib/timing.h"

#include "svs/core/data.h"
#include "svs/core/data/view.h"
#include "svs/core/query_result.h"

#include "svs/index/flat/flat.h"

// // tsl
// #include "tsl/robin_set.h"

// stdlib
#include <random>
#include <unordered_set>
#include <vector>

namespace svs::misc {

template <typename I, typename Alloc, typename Rng>
void shuffle(std::vector<I, Alloc>& v, Rng& rng) {
    std::shuffle(std::begin(v), std::end(v), rng);
}

using RNGType = std::mt19937_64;

template <typename Idx> std::vector<Idx> init_indices(size_t n) {
    auto iota = threads::UnitRange<Idx>(0, n);
    return std::vector<Idx>{iota.begin(), iota.end()};
}

template <typename Idx> struct Bucket {
    Bucket(threads::UnitRange<Idx> ids, const Matrix<Neighbor<Idx>>& groundtruth)
        : ids_{ids}
        , groundtruth_{groundtruth} {}

    Bucket(threads::UnitRange<Idx> ids, Matrix<Neighbor<Idx>>&& groundtruth)
        : ids_{ids}
        , groundtruth_{std::move(groundtruth)} {}

    /// The indices of the main dataset that belong in this bucket.
    threads::UnitRange<Idx> ids_;
    /// The groundtruth of the queryset for this bucket.
    Matrix<Neighbor<Idx>> groundtruth_;
};

///
/// @brief Helper class for verifying and characterizing mutable indexes.
///
/// The main idea here is to divide the base dataset into chunks (called "buckets"):
/// For example, if we have a dataset with 10 vectors, we could divide it into three
/// chunks like so:
///
/// 0 -+
/// 1  | Bucket 0
/// 2  |
/// 3 -+
/// 4 -+
/// 5  | Bucket 1
/// 6  |
/// 7 -+
/// 8 -+ Bucket 2
/// 9 -+
///
/// The main problem when working with mutable indexes is computing the groundtruth for any
/// particular state of the index.
/// Using this bucket approach allows us to accelerate the groundtruth computation.
/// We compute the groundtruth between the queries and the vectors within each bucket.
/// If we add and remove vectors from the dataset at the bucket granularity, then we can
/// compute the current groundtruth by merging the groundtruth of each bucket within the
/// dataset.
///
///
template <typename Idx, typename Eltype, size_t N, typename Dist> class ReferenceDataset {
  public:
    // Type Aliases
    using data_type = data::SimpleData<Eltype, N>;

  private:
    ///// Members
    /// @brief The full base dataset the data is taken from.
    data_type data_;
    /// @brief The number of queries used when constructing the reference.
    size_t num_queries_;
    /// @brief The number of neighbors to return for groundtruth computations.
    size_t num_neighbors_;
    /// @brief The configured number of IDs in each bucket.
    size_t bucket_size_;
    /// @brief The distance computation to use.
    Dist distance_;
    /// @brief Threads to use when merging the groundtruth for buckets in the dataset.
    threads::NativeThreadPool threadpool_;
    bool extra_checks_ = false;
    /// @brief Associative data structure for all IDs currently in the dataset.
    std::unordered_set<Idx> valid_{};
    /// @brief The data buckets that are currently in the dataset.
    std::vector<Bucket<Idx>> buckets_in_dataset_{};
    /// @brief Reserve buckets to be used when adding points.
    std::vector<Bucket<Idx>> reserve_buckets_{};
    /// @brief Random number generator for deterministic runs.
    RNGType rng_ = {};

  public:
    // Methods

    ///
    /// @param data The dataset to use
    /// @param distance The distance functor to use.
    /// @param num_threads Number of threads to use for groundtruth computation.
    /// @param bucket_size Target number of IDs to use per bucket.
    /// @param num_neighbors The number of neighbors to retrieve when computing the base
    ///     ground truth.
    /// @param queries The query set that will be used.
    /// @param rng_seed The seed to use for random number generator initialization.
    ///
    template <data::ImmutableMemoryDataset Queries>
    ReferenceDataset(
        data_type data,
        Dist distance,
        size_t num_threads,
        size_t bucket_size,
        size_t num_neighbors,
        const Queries& queries,
        uint64_t rng_seed
    )
        : data_{std::move(data)}
        , num_queries_{queries.size()}
        , num_neighbors_{num_neighbors}
        , bucket_size_{bucket_size}
        , distance_(std::move(distance))
        , threadpool_{num_threads}
        , rng_{rng_seed} {
        // Perform some sanity checks.
        if (bucket_size_ < num_neighbors) {
            throw ANNEXCEPTION(
                "Bucket size {} is less than number of neighbors {}",
                bucket_size_,
                num_neighbors_
            );
        }

        auto timer = lib::Timer();
        size_t start = 0;
        size_t datasize = data_.size();
        size_t num_queries = queries.size();
        while (start < datasize) {
            // Create a bucket of sequential IDs.
            // Compute the groundtruth between the dataset elements in this bucket
            // and the queries.
            // Then, create a "Bucket" with this information and append it to the list
            // of reserve buckets.
            auto handler = timer.push_back("compute groundtruth");
            auto stop = std::min(start + bucket_size, datasize);
            auto ids =
                threads::UnitRange<Idx>(lib::narrow<Idx>(start), lib::narrow<Idx>(stop));
            auto view = data::make_const_view(data_, ids);

            auto index = index::flat::temporary_flat_index(view, distance_, threadpool_);
            auto groundtruth =
                svs::index::search_batch(index, queries.cview(), num_neighbors);

            // Unpack the QueryResult
            const auto& indices = groundtruth.indices();
            const auto& distances = groundtruth.distances();

            // Construct a neighbor-matrix from the groundtruth, reindexing the returned
            // ID's to make them global.
            auto bucket_groundtruth =
                make_dense_array<Neighbor<Idx>>(num_queries, num_neighbors);
            for (size_t i = 0; i < num_queries; ++i) {
                for (size_t j = 0; j < num_neighbors; ++j) {
                    auto parent_id = lib::narrow<Idx>(view.parent_id(indices.at(i, j)));
                    bucket_groundtruth.at(i, j) =
                        Neighbor<Idx>{parent_id, distances.at(i, j)};
                }
            }

            reserve_buckets_.emplace_back(ids, std::move(bucket_groundtruth));
            start = stop;
        }
        svs::logging::debug("{}", timer);
    }

    /// @brief Return the total number of elements in the dataset.
    size_t size() const { return data_.size(); }

    ///
    /// Return the number of elements that is expected to be currently resident in the
    /// mutable index
    ///
    size_t valid() const { return valid_.size(); }

    /// @brief Return the configured size of each bucket of vectors.
    size_t bucket_size() const { return bucket_size_; }

    bool extra_checks_enabled() const { return extra_checks_; }
    void configure_extra_checks(bool enable) { extra_checks_ = enable; }

    ///
    /// @brief Return whether index `i` is a valid index.
    ///
    bool is_valid(size_t i) const { return valid_.contains(i); }

    ///
    /// @brief Ensure that all IDs present in the ``QueryResult`` are valid.
    ///
    /// This ensures that the mutable index does not return stale IDs that should have been
    /// removed.
    ///
    void check_ids(const Matrix<size_t>& indices) const {
        for (auto e : indices) {
            if (e != std::numeric_limits<size_t>::max() && !is_valid(e)) {
                throw ANNEXCEPTION("Index return ID {} which is invalid!", e);
            }
        }
    }

    void check_ids(const QueryResult<size_t>& result) const {
        return check_ids(result.indices());
    }

    QueryResult<size_t> groundtruth() {
        auto gt = make_dense_array<Neighbor<Idx>>(num_queries_, num_neighbors_);
        // Initially, fill the groundtruth with sentinel types.
        using Cmp = distance::compare_t<Dist>;
        std::fill(gt.begin(), gt.end(), type_traits::sentinel_v<Neighbor<Idx>, Cmp>);

        auto reserve = make_dense_array<Neighbor<Idx>>(num_queries_, num_neighbors_);
        auto cmp = distance::comparator(distance_);

        for (const auto& bucket : buckets_in_dataset_) {
            const auto& bucket_gt = bucket.groundtruth_;
            if (bucket_gt.size() != gt.size()) {
                throw ANNEXCEPTION("What?");
            }

            threads::run(
                threadpool_,
                threads::StaticPartition(num_queries_),
                [&](auto is, auto /*tid*/) {
                    for (auto i : is) {
                        auto dst = reserve.slice(i);
                        lib::ranges::bounded_merge(
                            gt.slice(i), bucket_gt.slice(i), dst, cmp
                        );
                    }
                }
            );
            std::swap(gt, reserve);
        }

        // Construct the query result.
        auto result = QueryResult<size_t>(num_queries_, num_neighbors_);
        for (size_t i = 0; i < num_queries_; ++i) {
            for (size_t j = 0; j < num_neighbors_; ++j) {
                const auto& neighbor = gt.at(i, j);
                result.index(i, j) = neighbor.id();
                result.distance(i, j) = neighbor.distance();
            }
        }
        return result;
    }

    size_t
    get_num_points(const std::vector<Bucket<Idx>>& buckets, size_t max_points) const {
        // Start from the back and move our way forward.
        // Accumulate the size of each bucket until we would exceed the max points.
        size_t sz = 0;
        for (auto it = buckets.crbegin(); it != buckets.crend(); ++it) {
            const auto& bucket = *it;
            size_t next_sz = sz + bucket.ids_.size();
            if (next_sz > max_points) {
                return sz;
            }
            sz = next_sz;
        }
        return sz;
    }

    std::pair<data::SimpleData<Eltype, N>, std::vector<Idx>> generate(size_t max_points) {
        // Make sure we don't exceed the actual maximum number of points.
        size_t max_addable_points = size() - valid();
        max_points = std::min(max_points, max_addable_points);
        shuffle(reserve_buckets_, rng_);
        size_t num_points = get_num_points(reserve_buckets_, max_points);
        if (num_points == 0) {
            throw ANNEXCEPTION("Something went wrong!");
        }

        // Marshall the new points to add.
        auto vectors = data::SimpleData<Eltype, N>(num_points, data_.dimensions());
        std::vector<Idx> points(num_points);

        // We want to add the points in a shuffled order.
        // Once we know exactly how many points "N" we are going to add, we can construct a
        // permutation vector containing "[0, N)" and shuffle that vector.
        auto permutation = init_indices<size_t>(num_points);
        shuffle(permutation, rng_);

        size_t count = 0;
        while (!reserve_buckets_.empty() && count != num_points) {
            // Peek at the last reserve bucket.
            // If we can add it without exceeing the point requirement, then we remove it
            // from the reserve buckets and add it to the new points.
            auto& back_bucket = reserve_buckets_.back();
            if (count + back_bucket.ids_.size() > num_points) {
                break;
            }

            for (auto id : back_bucket.ids_) {
                valid_.insert(id);
                auto dest = permutation.at(count);
                points.at(dest) = id;
                vectors.set_datum(dest, data_.get_datum(id));
                ++count;
            }

            // Move this bucket to the mark it as belonging to the dataset.
            buckets_in_dataset_.emplace_back(std::move(back_bucket));
            reserve_buckets_.pop_back();
        }
        if (count != num_points) {
            throw ANNEXCEPTION(
                "Trying to add {} points but only found {}!", num_points, count
            );
        }

        return std::make_pair(std::move(vectors), std::move(points));
    }

    ///
    /// @brief Add ``num_points`` new unique vectors to ``index``.
    ///
    /// @returns The number of points added and the time spend adding those points.
    ///
    template <typename MutableIndex>
    std::pair<size_t, double> add_points(MutableIndex& index, size_t num_points) {
        auto [vectors, indices] = generate(num_points);
        // Add the points to the index.
        auto tic = lib::now();
        index.add_points(vectors, indices);
        double time = lib::time_difference(tic);
        return std::make_pair(indices.size(), time);
    }

    std::vector<Idx> get_delete_points(size_t max_points) {
        // Don't empty the dataset.
        size_t max_deletable_points = valid();
        max_points = std::min(max_points, max_deletable_points);
        shuffle(buckets_in_dataset_, rng_);
        size_t num_points = get_num_points(buckets_in_dataset_, max_points);
        if (num_points == 0) {
            throw ANNEXCEPTION("Something went wrong!");
        }

        std::vector<Idx> points{};
        points.reserve(num_points);

        while (points.size() < num_points && !buckets_in_dataset_.empty()) {
            auto& back_bucket = buckets_in_dataset_.back();
            auto ids = back_bucket.ids_;
            points.insert(points.end(), ids.begin(), ids.end());
            for (auto id : ids) {
                valid_.erase(id);
            }
            reserve_buckets_.emplace_back(std::move(back_bucket));
            buckets_in_dataset_.pop_back();
        }

        if (points.size() != num_points) {
            throw ANNEXCEPTION("Mismatch in the number of points to be deleted!");
        }
        shuffle(points, rng_);
        return points;
    }

    template <typename MutableIndex>
    std::pair<size_t, double> delete_points(MutableIndex& index, size_t num_points) {
        auto points = get_delete_points(num_points);
        auto tic = svs::lib::now();
        index.delete_entries(points);
        double time = svs::lib::time_difference(tic);
        return std::make_pair(num_points, time);
    }

    ///
    /// @brief Verify that the reference and mutable index contain the same IDs.
    ///
    template <typename MutableIndex> void check_equal_ids(MutableIndex& index) {
        // Baseline Checks
        const size_t index_size = index.size();
        if (index_size != valid()) {
            throw ANNEXCEPTION(
                "Index claims to have {} valid IDs when it should have {}!",
                index_size,
                valid()
            );
        }

        // Abort early if additional checks aren't enabled.
        if (!extra_checks_) {
            return;
        }

        // Make sure all valid ID's in the Reference are in ``index``.
        for (auto e : valid_) {
            if (!index.has_id(e)) {
                throw ANNEXCEPTION("Index does not have id {} when it should!", e);
            }
        }

        // Now, make sure all ID's in the index are valid.
        for (auto e : index.external_ids()) {
            if (!valid_.contains(e)) {
                throw ANNEXCEPTION("Index contains a invalid id {}!", e);
            }
        }
    }
};

} // namespace svs::misc
