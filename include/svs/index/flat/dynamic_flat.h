/*
 * Copyright 2025 Intel Corporation
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

// stdlib
#include <memory>

// Include the flat index
#include "svs/index/flat/flat.h"

// svs
#include "svs/concepts/distance.h"
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/loading.h"
#include "svs/core/logging.h"
#include "svs/core/query_result.h"
#include "svs/core/translation.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/invoke.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/threads.h"

namespace svs::index::flat {

///
/// Metadata tracking the state of a particular data index.
/// The following states have the given meaning for their corresponding slot:
///
/// * Valid: Valid and present in the associated dataset.
/// * Deleted: Exists in the associated dataset, but should be considered as "deleted"
/// and not returned from any search algorithms.
/// * Empty: Non-existent and unreachable from standard entry points.
///
/// Only used for `DynamicFlatIndex`.
///
enum class SlotMetadata : uint8_t { Empty = 0x00, Valid = 0x01, Deleted = 0x02 };

///
/// @brief Dynamic Flat Index with insertion and deletion support
///
/// @tparam Data The full type of the dataset being indexed.
/// @tparam Dist The distance functor used to compare queries with the elements of the
///     dataset.
///
/// A flat index implementation that supports dynamic insertion and deletion of vectors
/// while maintaining exhaustive search capabilities.
///
template <data::ImmutableMemoryDataset Data, typename Dist> class DynamicFlatIndex {
  public:
    using distance_type = Dist;
    using data_type = Data;
    using search_parameters_type = FlatParameters;

  private:
    data_type data_;
    std::vector<SlotMetadata> status_;
    size_t first_empty_ = 0;
    IDTranslator translator_;
    distance_type distance_;
    threads::ThreadPoolHandle threadpool_;
    search_parameters_type search_parameters_{};
    svs::logging::logger_ptr logger_;

  public:
    // Constructors
    template <typename ExternalIds, typename ThreadPoolProto>
    DynamicFlatIndex(
        Data data,
        const ExternalIds& external_ids,
        Dist distance_function,
        ThreadPoolProto threadpool_proto,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : data_{std::move(data)}
        , status_(data_.size(), SlotMetadata::Valid)
        , first_empty_{data_.size()}
        , translator_()
        , distance_{std::move(distance_function)}
        , threadpool_{threads::as_threadpool(std::move(threadpool_proto))}
        , search_parameters_{}
        , logger_{std::move(logger)} {
        translator_.insert(
            external_ids, threads::UnitRange<size_t>(0, external_ids.size())
        );
    }

    ///// Core interface matching FlatIndex

    /// @brief Getter method for logger
    svs::logging::logger_ptr get_logger() const { return logger_; }

    /// @brief Return the number of independent entries in the index.
    size_t size() const {
        size_t count = 0;
        for (const auto& status : status_) {
            if (status == SlotMetadata::Valid) {
                ++count;
            }
        }
        return count;
    }

    /// @brief Return the logical number of dimensions of the indexed vectors.
    size_t dimensions() const { return data_.dimensions(); }

    /// @brief Add the points with the given external IDs to the dataset.
    ///
    /// When `delete_entries` is called, a soft deletion is performed, marking the entries
    /// as `deleted`. When `consolidate` is called, the state of these deleted entries
    /// becomes `empty`. When `add_points` is called with the `reuse_empty` flag enabled,
    /// the memory is scanned from the beginning to locate and fill these empty entries with
    /// new points.
    ///
    /// @param points Dataset of points to add.
    /// @param external_ids The external IDs of the corresponding points. Must be a
    ///     container implementing forward iteration.
    /// @param reuse_empty A flag that determines whether to reuse empty entries that may
    /// exist after deletion and consolidation. When enabled, scan from the beginning to
    /// find and fill these empty entries when adding new points.
    ///
    template <data::ImmutableMemoryDataset Points, class ExternalIds>
    std::vector<size_t> add_points(
        const Points& points, const ExternalIds& external_ids, bool reuse_empty = false
    ) {
        const size_t num_points = points.size();
        const size_t num_ids = external_ids.size();
        if (num_points != num_ids) {
            throw ANNEXCEPTION(
                "Number of points ({}) not equal to the number of external ids ({})!",
                num_points,
                num_ids
            );
        }

        // Gather all empty slots.
        std::vector<size_t> slots{};
        slots.reserve(num_points);
        bool have_room = false;

        size_t s = reuse_empty ? 0 : first_empty_;
        size_t smax = status_.size();
        for (; s < smax; ++s) {
            if (status_[s] == SlotMetadata::Empty) {
                slots.push_back(s);
            }
            if (slots.size() == num_points) {
                have_room = true;
                break;
            }
        }

        // Check if we have enough indices. If we don't, we need to resize the data.
        if (!have_room) {
            size_t needed = num_points - slots.size();
            size_t current_size = data_.size();
            size_t new_size = current_size + needed;
            data_.resize(new_size);
            status_.resize(new_size, SlotMetadata::Empty);

            // Append the correct number of extra slots.
            threads::UnitRange<size_t> extra_points{current_size, current_size + needed};
            slots.insert(slots.end(), extra_points.begin(), extra_points.end());
        }
        assert(slots.size() == num_points);

        // Try to update the id translation now that we have internal ids.
        // If this fails, we still haven't mutated the index data structure so we're safe
        // to throw an exception.
        translator_.insert(external_ids, slots);

        // Copy the given points into the data.
        copy_points(points, slots);

        // Mark all added entries as valid.
        for (const auto& i : slots) {
            status_[i] = SlotMetadata::Valid;
        }

        if (!slots.empty()) {
            first_empty_ = std::max(first_empty_, slots.back() + 1);
        }
        return slots;
    }

    ///
    /// @brief Call the functor with all external IDs in the index.
    ///
    /// @param f A functor with an overloaded ``operator()(size_t)`` method. Called on
    ///     each external ID in the index.
    ///
    template <typename F> void on_ids(F&& f) const {
        for (auto pair : translator_) {
            f(pair.first);
        }
    }

  private:
    /// @brief Copy points from the source dataset into the specified slots.
    template <data::ImmutableMemoryDataset Points>
    void copy_points(const Points& points, const std::vector<size_t>& slots) {
        assert(points.size() == slots.size());
        for (size_t i = 0; i < points.size(); ++i) {
            data_.set_datum(slots[i], points.get_datum(i));
        }
    }

  public:
};

///// Deduction Guides.
template <typename Data, typename Dist, typename ExternalIds>
DynamicFlatIndex(Data, const ExternalIds&, Dist, size_t) -> DynamicFlatIndex<Data, Dist>;

template <typename Data, typename Dist, typename ExternalIds, threads::ThreadPool Pool>
DynamicFlatIndex(Data, const ExternalIds&, Dist, Pool) -> DynamicFlatIndex<Data, Dist>;

///
/// @brief Entry point for creating a Dynamic Flat index.
///
template <typename DataProto, typename Distance, typename ThreadPoolProto>
auto auto_dynamic_assemble(
    DataProto&& data_proto,
    Distance distance,
    ThreadPoolProto threadpool_proto,
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data = svs::detail::dispatch_load(std::forward<DataProto>(data_proto), threadpool);

    // For initial construction, create sequential external IDs
    auto external_ids = threads::UnitRange<size_t>(0, data.size());

    return DynamicFlatIndex(
        std::move(data),
        external_ids,
        std::move(distance),
        std::move(threadpool),
        std::move(logger)
    );
}

} // namespace svs::index::flat
