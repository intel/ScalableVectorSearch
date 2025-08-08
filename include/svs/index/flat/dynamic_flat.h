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
