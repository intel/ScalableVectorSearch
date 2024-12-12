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

// Flat index utilities
#include "svs/index/flat/inserters.h"
#include "svs/index/index.h"

// svs
#include "svs/concepts/distance.h"
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/loading.h"
#include "svs/core/query_result.h"
#include "svs/lib/invoke.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads.h"

// stdlib
#include <tuple>

namespace svs::index::flat {

namespace extensions {

struct FlatDistance {
    template <typename Data, typename Distance>
    svs::svs_invoke_result_t<FlatDistance, const Data&, const Distance&>
    operator()(const Data& data, const Distance& distance) const {
        return svs::svs_invoke(*this, data, distance);
    }
};

struct FlatAccessor {
    template <typename Data>
    svs::svs_invoke_result_t<FlatAccessor, Data> operator()(const Data& data) const {
        return svs::svs_invoke(*this, data);
    }
};

// Customization point objects.
inline constexpr FlatDistance distance{};
inline constexpr FlatAccessor accessor{};

// Default implementations.
template <typename Data, typename Distance>
Distance svs_invoke(
    svs::tag_t<distance>, const Data& SVS_UNUSED(dataset), const Distance& distance
) {
    return threads::shallow_copy(distance);
}

template <typename Data>
data::GetDatumAccessor svs_invoke(svs::tag_t<accessor>, const Data& SVS_UNUSED(data)) {
    return data::GetDatumAccessor{};
}

} // namespace extensions

// The flat index is "special" because we wish to enable the `FlatIndex` to either:
// (1) Own the data and thread pool.
// (2) Reference an existing dataset and thread pool.
//
// The latter option allows other index implementations like the VamanaIndex to launch a
// scoped `FlatIndex` to perform exhaustive searches on demand (useful when validating
// the behavior of the dynamic index).
//
// To that end, we allow the actual storage of the data and the threadpool to either be
// owning (by value) or non-owning (by reference).
struct OwnsMembers {
    template <typename T> using storage_type = T;
};
struct ReferencesMembers {
    template <typename T> using storage_type = T&;
};

template <typename Ownership, typename T>
using storage_type_t = typename Ownership::template storage_type<T>;

struct FlatParameters {
    FlatParameters() = default;
    FlatParameters(size_t data_batch_size, size_t query_batch_size)
        : data_batch_size_{data_batch_size}
        , query_batch_size_{query_batch_size} {}

    ///// Members
    size_t data_batch_size_ = 0;
    size_t query_batch_size_ = 0;
};

///
/// @brief Implementation of the Flat index.
///
/// @tparam Data The full type of the dataset being indexed.
/// @tparam Dist The distance functor used to compare queries with the elements of the
///     dataset.
/// @tparam Ownership Implementation detail and may be ommitted for most use cases.
///
/// The mid-level implementation for the flat index that uses exhaustive search to find
/// the exact nearest neighbors (within the limitations of possibly quantization error
/// for the dataset or floating-point error for some distance functors).
///
/// **NOTE**: This method is not as performant as other index methods. It is meant to
/// return the exact rather than approximate nearest neighbors and thus must exhaustively
/// search the whole dataset.
///
template <
    data::ImmutableMemoryDataset Data,
    typename Dist,
    typename Ownership = OwnsMembers>
class FlatIndex {
  public:
    using const_value_type = data::const_value_type_t<Data>;

    /// The type of the distance functor.
    using distance_type = Dist;
    /// The type of dataset.
    using data_type = Data;
    using thread_pool_type = threads::NativeThreadPool;
    using compare = distance::compare_t<Dist>;
    using sorter_type = BulkInserter<Neighbor<size_t>, compare>;

    static const size_t default_data_batch_size = 100'000;

    // Compute data and threadpool storage types.
    using data_storage_type = storage_type_t<Ownership, Data>;
    using thread_storage_type = storage_type_t<Ownership, thread_pool_type>;

    // Search parameters
    using search_parameters_type = FlatParameters;

  private:
    data_storage_type data_;
    [[no_unique_address]] distance_type distance_;
    thread_storage_type threadpool_;

    // Constructs controlling the iteration strategy over the data and queries.
    search_parameters_type search_parameters_{};

    // Helpers methods to obtain automatic batch sizing.

    // Automatic behavior: Use the default batch size.
    size_t compute_data_batch_size(const search_parameters_type& p) const {
        auto sz = p.data_batch_size_;
        if (sz == 0) {
            return default_data_batch_size;
        }
        return std::min(sz, data_.size());
    }

    // Automatic behavior: Evenly divide queries over the threads.
    size_t
    compute_query_batch_size(const search_parameters_type& p, size_t num_queries) const {
        auto sz = p.query_batch_size_;
        if (sz == 0) {
            return lib::div_round_up(num_queries, threadpool_.size());
        }
        return std::min(sz, num_queries);
    }

  public:
    search_parameters_type get_search_parameters() const { return search_parameters_; }

    void set_search_parameters(const search_parameters_type& search_parameters) {
        search_parameters_ = search_parameters;
    }

    ///
    /// @brief Construct a new index from constituent parts.
    ///
    /// @tparam ThreadPoolProto The type of the threadpool proto type. See notes on the
    ///     corresponding parameter below.
    ///
    /// @param data The data to use for the index. The resulting index will take ownership
    ///     of the passed argument.
    /// @param distance The distance functor to use to compare queries with dataset
    ///     elements.
    /// @param threadpool_proto Something that can be used to build a threadpool using
    ///     ``threads::as_threadpool``. In practice, this means that ``threapool_proto``
    ///     can be either a threadpool directly, or an integer. In the latter case, a new
    ///     threadpool will be constructed using ``threadpool_proto`` as the number of
    ///     threads to create.
    ///
    template <typename ThreadPoolProto>
    FlatIndex(Data data, Dist distance, ThreadPoolProto&& threadpool_proto)
        requires std::is_same_v<Ownership, OwnsMembers>
        : data_{std::move(data)}
        , distance_{std::move(distance)}
        , threadpool_{
              threads::as_threadpool(std::forward<ThreadPoolProto>(threadpool_proto))} {}

    FlatIndex(Data& data, Dist distance, threads::NativeThreadPool& threadpool)
        requires std::is_same_v<Ownership, ReferencesMembers>
        : data_{data}
        , distance_{std::move(distance)}
        , threadpool_{threadpool} {}

    ////// Dataset Interface

    /// Return the number of independent entries in the index.
    size_t size() const { return data_.size(); }

    /// Return the logical number of dimensions of the indexed vectors.
    size_t dimensions() const { return data_.dimensions(); }

    /// @anchor flat_class_search_mutating
    /// @brief Fill the result with the ``num_neighbors`` nearest neighbors for each query.
    ///
    /// @tparam Queries The full type of the queries.
    /// @tparam Pred The type of the optional predicate.
    ///
    /// @param result The result data structure to populate.
    ///     Row `i` in the result corresponds to the neighbors for the `i`th query.
    ///     Neighbors within each row are ordered from nearest to furthest.
    ///     ``num_neighbors`` is computed from the number of columns in ``result``.
    /// @param queries A dense collection of queries in R^n.
    /// @param search_parameters search parameters to use for the search.
    /// @param cancel A predicate called during the search to determine if the search should be cancelled.
    //      Return ``true`` if the search should be cancelled. This functor must implement ``bool operator()()``.
    //      Note: This predicate should be thread-safe as it can be called concurrently by different threads during the search.
    /// @param predicate A predicate functor that can be used to exclude certain dataset
    ///     elements from consideration. This functor must implement
    ///     ``bool operator()(size_t)`` where the ``size_t`` argument is an index in
    ///     ``[0, data.size())``. If the predicate returns ``true``, that dataset element
    ///     will be considered.
    ///
    /// **Preconditions:**
    ///
    /// The following pre-conditions must hold. Otherwise, the behavior is undefined.
    /// - ``result.n_queries() == queries.size()``
    /// - ``result.n_neighbors() == num_neighbors``.
    /// - The value type of ``queries`` is compatible with the value type of the index
    ///     dataset with respect to the stored distance functor.
    ///
    /// **Implementation Details**
    ///
    /// The internal call stack looks something like this.
    ///
    /// @code{}
    /// search: Prepare scratch space and perform tiling over the dataset.
    ///   |
    ///   +-> search_subset: multi-threaded search of all queries over the current subset
    ///       of the dataset. Partitions up the queries according to query batch size
    ///       and dynamically load balances query partition among worker threads.
    ///         |
    ///         +-> search_patch: Bottom level routine meant to run on a single thread.
    ///             Compute the distances between a subset of the queries and a subset
    ///             of the data and maintines the `num_neighbors` best results seen so far.
    /// @endcode{}
    ///
    template <typename QueryType, typename Pred = lib::Returns<lib::Const<true>>>
    void search(
        QueryResultView<size_t> result,
        const data::ConstSimpleDataView<QueryType>& queries,
        const search_parameters_type& search_parameters,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>()),
        Pred predicate = lib::Returns(lib::Const<true>())
    ) {
        const size_t data_max_size = data_.size();

        // Partition the data into `data_batch_size_` chunks.
        // This will keep all threads at least working on the same sub-region of the dataset
        // to provide somewhat better locality.
        auto data_batch_size = compute_data_batch_size(search_parameters);

        // Allocate query processing space.
        size_t num_neighbors = result.n_neighbors();
        sorter_type scratch{queries.size(), num_neighbors, compare()};
        scratch.prepare();

        size_t start = 0;
        while (start < data_.size()) {
            // Check if request to cancel the search
            if (cancel()) {
                scratch.cleanup();
                return;
            }
            size_t stop = std::min(data_max_size, start + data_batch_size);
            search_subset(
                queries,
                threads::UnitRange(start, stop),
                scratch,
                search_parameters,
                cancel,
                predicate
            );
            start = stop;
        }

        // By this point, all queries have been compared with all dataset elements.
        // Perform any necessary post-processing on the sorting network and write back
        // the results.
        scratch.cleanup();
        threads::run(
            threadpool_,
            threads::StaticPartition(queries.size()),
            [&](const auto& query_indices, uint64_t /*tid*/) {
                for (auto i : query_indices) {
                    const auto& neighbors = scratch.result(i);
                    for (size_t j = 0; j < num_neighbors; ++j) {
                        result.set(neighbors[j], i, j);
                    }
                }
            }
        );
    }

    template <typename QueryType, typename Pred = lib::Returns<lib::Const<true>>>
    void search_subset(
        const data::ConstSimpleDataView<QueryType>& queries,
        const threads::UnitRange<size_t>& data_indices,
        sorter_type& scratch,
        const search_parameters_type& search_parameters,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>()),
        Pred predicate = lib::Returns(lib::Const<true>())
    ) {
        // Process all queries.
        threads::run(
            threadpool_,
            threads::DynamicPartition{
                queries.size(),
                compute_query_batch_size(search_parameters, queries.size())},
            [&](const auto& query_indices, uint64_t /*tid*/) {
                // Broadcast the distance functor so each thread can process all queries
                // in its current batch.
                distance::BroadcastDistance distances{
                    extensions::distance(data_, distance_), query_indices.size()};

                search_patch(
                    queries,
                    data_indices,
                    threads::UnitRange(query_indices),
                    scratch,
                    distances,
                    cancel,
                    predicate
                );
            }
        );
    }

    // Perform all distance computations between the queries and the stored dataset over
    // the cartesian product of `query_indices` x `data_indices`.
    //
    // Insert the computed distance for each query/distance pair into `scratch`, which
    // will maintain the correct number of nearest neighbors.
    template <
        typename QueryType,
        typename DistFull,
        typename Pred = lib::Returns<lib::Const<true>>>
    void search_patch(
        const data::ConstSimpleDataView<QueryType>& queries,
        const threads::UnitRange<size_t>& data_indices,
        const threads::UnitRange<size_t>& query_indices,
        sorter_type& scratch,
        distance::BroadcastDistance<DistFull>& distance_functors,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>()),
        Pred predicate = lib::Returns(lib::Const<true>())
    ) {
        assert(distance_functors.size() >= query_indices.size());
        auto accessor = extensions::accessor(data_);

        // Fix arguments
        for (size_t i = 0; i < query_indices.size(); ++i) {
            distance::maybe_fix_argument(
                distance_functors[i], queries.get_datum(query_indices[i])
            );
        }

        for (auto data_index : data_indices) {
            // Check if request to cancel the search
            if (cancel()) {
                return;
            }

            // Skip this index if it doesn't pass the predicate.
            if (!predicate(data_index)) {
                continue;
            }

            auto datum = accessor(data_, data_index);

            // Loop over the queries.
            // Compute the distance between each query and the dataset element and insert
            // it into the sorting network.
            for (size_t i = 0; i < query_indices.size(); ++i) {
                auto query_index = query_indices[i];
                auto d = distance::compute(
                    distance_functors[i], queries.get_datum(query_index), datum
                );
                scratch.insert(query_index, {data_index, d});
            }
        }
    }

    // Threading Interface

    /// Return whether this implementation can dynamically change the number of threads.
    static bool can_change_threads() { return true; }

    ///
    /// @brief Return the current number of threads used for search.
    ///
    /// @sa set_num_threads
    size_t get_num_threads() const { return threadpool_.size(); }

    ///
    /// @brief Set the number of threads used for search.
    ///
    /// @param num_threads The new number of threads to use.
    ///
    /// Implementation note: The number of threads cannot be zero. If zero is passed to
    /// this method, it will be silently changed to 1.
    ///
    /// @sa get_num_threads
    ///
    void set_num_threads(size_t num_threads) {
        num_threads = std::max(num_threads, size_t(1));
        threadpool_.resize(num_threads);
    }
};

///
/// @class hidden_flat_auto_assemble
///
/// data_loader
/// ===========
///
/// The data loader should be any object loadable via ``svs::detail::dispatch_load``
/// returning a Vamana compatible dataset. Concrete examples include:
///
/// * An instance of ``VectorDataLoader``.
/// * An implementation of ``svs::data::ImmutableMemoryDataset`` (passed by value).
///

///
/// @brief Entry point for loading a Flat index.
///
/// @param data_proto Data prototype. See expanded notes.
/// @param distance The distance **functor** to use to compare queries with elements of the
///     dataset.
/// @param threadpool_proto Precursor for the thread pool to use. Can either be a threadpool
///     instance of an integer specifying the number of threads to use.
///
/// This method provides much of the heavy lifting for constructing a Flat index from
/// a data file on disk or a dataset in memory.
///
/// @copydoc hidden_flat_auto_assemble
///
template <typename DataProto, typename Distance, typename ThreadPoolProto>
auto auto_assemble(
    DataProto&& data_proto, Distance distance, ThreadPoolProto threadpool_proto
) {
    auto threadpool = threads::as_threadpool(threadpool_proto);
    auto data = svs::detail::dispatch_load(std::forward<DataProto>(data_proto), threadpool);
    return FlatIndex(std::move(data), std::move(distance), std::move(threadpool));
}

/// @brief Alias for a short-lived flat index.
template <data::ImmutableMemoryDataset Data, typename Dist>
using TemporaryFlatIndex = FlatIndex<Data, Dist, ReferencesMembers>;

template <data::ImmutableMemoryDataset Data, typename Dist>
TemporaryFlatIndex<Data, Dist>
temporary_flat_index(Data& data, Dist distance, threads::NativeThreadPool& threadpool) {
    return TemporaryFlatIndex<Data, Dist>{data, distance, threadpool};
}

} // namespace svs::index::flat
