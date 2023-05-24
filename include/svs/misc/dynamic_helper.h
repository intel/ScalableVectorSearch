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

///
/// @file dynamic_helper.h
///
/// A simple helper for testing dynamic indexes.
///

#pragma once

// svs
#include "svs/lib/threads.h"
#include "svs/lib/timing.h"

#include "svs/core/data.h"
#include "svs/core/query_result.h"

#include "svs/index/flat/flat.h"

// tsl
#include "tsl/robin_set.h"

// stdlib
#include <random>
#include <vector>

namespace svs::misc {

template <typename I, typename Alloc, typename Rng>
void shuffle(std::vector<I, Alloc>& v, Rng& rng) {
    std::shuffle(std::begin(v), std::end(v), rng);
}

using RNGType = decltype(std::default_random_engine());

template <typename Idx> std::vector<Idx> init_indices(size_t n) {
    auto iota = threads::UnitRange<Idx>(0, n);
    return std::vector<Idx>{iota.begin(), iota.end()};
}

template <typename Idx, typename Eltype, size_t N, typename Dist> class ReferenceDataset {
  public:
    // Type Aliases
    using data_type = data::SimplePolymorphicData<Eltype, N>;

  private:
    // Members
    data_type data_;
    size_t num_threads_;
    Dist distance_;
    bool extra_checks_ = false;
    std::vector<Idx> indices_{};
    tsl::robin_set<Idx> valid_{};
    RNGType rng_ = std::default_random_engine();

  public:
    // Methods

    // Create a reference dataset.
    ReferenceDataset(data_type data, size_t num_threads, Dist distance)
        : data_{std::move(data)}
        , num_threads_(num_threads)
        , distance_(std::move(distance))
        , indices_{init_indices<Idx>(data_.size())} {}

    size_t size() const { return data_.size(); }
    size_t valid() const { return valid_.size(); }

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
    void check_ids(const QueryResult<size_t>& result) {
        const auto& indices = result.indices();
        for (auto e : indices) {
            if (!is_valid(e)) {
                throw ANNEXCEPTION("Index return ID ", e, " which is invalid!");
            }
        }
    }

    ///
    /// @brief Compute the groundtruth for the current state of the index.
    ///
    /// @param queries The query set.
    /// @param num_neighbors The number of neighbors to compute.
    ///
    template <data::ImmutableMemoryDataset Queries>
    QueryResult<size_t> groundtruth(const Queries& queries, size_t num_neighbors) const {
        auto threadpool = threads::NativeThreadPool(num_threads_);
        auto index = index::flat::temporary_flat_index(data_, distance_, threadpool);
        return index.search(queries, num_neighbors, [&](size_t i) { return is_valid(i); });
    }

    std::pair<data::SimpleData<Eltype, N>, std::vector<Idx>> generate(size_t num_points) {
        // Make sure we don't exceed the actual maximum number of points.
        size_t max_addable_points = size() - valid();
        num_points = std::min(num_points, max_addable_points);
        shuffle(indices_, rng_);

        // Marshall the new points to add.
        auto vectors = data::SimpleData<Eltype, N>(num_points, N);
        std::vector<Idx> points{};
        points.reserve(num_points);

        size_t count = 0;
        for (size_t i = 0, imax = size(); i < imax && count < num_points; ++i) {
            const auto& j = indices_.at(i);
            // Don't add a point multiple times.
            if (is_valid(j)) {
                continue;
            }

            // Now, we have a point that doesn't yet exist in the dataset.
            // Copy it into the ``vectors`` buffer and record its index.
            vectors.set_datum(count, data_.get_datum(j));
            points.push_back(j);
            valid_.insert(j);
            ++count;
        }

        size_t points_actually_added = points.size();
        if (points_actually_added != num_points) {
            throw ANNEXCEPTION(
                "Trying to add ",
                num_points,
                " points but only found ",
                points_actually_added,
                '!'
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

    template <typename MutableIndex>
    std::pair<size_t, double> delete_points(MutableIndex& index, size_t num_points) {
        // Don't empty the dataset.
        size_t max_deletable_points = valid();
        num_points = std::min(num_points, max_deletable_points);
        shuffle(indices_, rng_);
        std::vector<Idx> points{};
        points.reserve(num_points);

        for (size_t i = 0, imax = size(); i < imax && points.size() < num_points; ++i) {
            const auto& j = indices_[i];
            if (is_valid(j)) {
                points.push_back(j);
                valid_.erase(j);
            }
        }

        if (points.size() != num_points) {
            throw ANNEXCEPTION("Mismatch in the number of points to be deleted!");
        }

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
                "Index claims to have ",
                index_size,
                " valid IDs when it should have ",
                valid(),
                '!'
            );
        }

        // Abort early if additional checks aren't enabled.
        if (!extra_checks_) {
            return;
        }

        // Make sure all valid ID's in the Reference are in ``index``.
        for (auto e : valid_) {
            if (!index.has_id(e)) {
                throw ANNEXCEPTION("Index does not have id ", e, " when it should!");
            }
        }

        // Now, make sure all ID's in the index are valid.
        for (auto e : index.external_ids()) {
            if (!valid_.contains(e)) {
                throw ANNEXCEPTION("Index contains a invalid id ", e, '!');
            }
        }
    }
};

} // namespace svs::misc
