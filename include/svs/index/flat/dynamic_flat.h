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
#include <filesystem>
#include <memory>

// Include the flat index
#include "svs/index/flat/flat.h"
#include "svs/index/flat/inserters.h"

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
#include "svs/lib/misc.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/threads.h"

namespace svs::index::flat {

///
/// Metadata tracking the state of a particular data index for DynamicFlatIndex.
/// The following states have the given meaning for their corresponding slot:
///
/// * Valid: Valid and present in the associated dataset.
/// * Empty: Available slot that can be used for new data.
///
/// For flat indices, deletion directly transitions from Valid to Empty.
///
enum class SlotMetadata : uint8_t { Empty = 0x00, Valid = 0x01 };

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
template <typename Data, typename Dist> class DynamicFlatIndex {
  public:
    // Traits
    static constexpr bool supports_insertions = true;
    static constexpr bool supports_deletions = true;
    static constexpr bool supports_saving = true;
    static constexpr bool needs_id_translation = true;

    // Type Aliases
    using internal_id_type = size_t;
    using external_id_type = size_t;
    using distance_type = Dist;
    using data_type = Data;
    using search_parameters_type = FlatParameters;
    using compare = distance::compare_t<Dist>;
    using sorter_type = BulkInserter<Neighbor<size_t>, compare>;

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

    // Constructor that accepts a pre-existing translator (for loading from saved state)
    template <typename ThreadPoolProto>
    DynamicFlatIndex(
        Data data,
        IDTranslator translator,
        Dist distance_function,
        ThreadPoolProto threadpool_proto,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : data_{std::move(data)}
        , status_(data_.size(), SlotMetadata::Valid)
        , first_empty_{data_.size()}
        , translator_{std::move(translator)}
        , distance_{std::move(distance_function)}
        , threadpool_{threads::as_threadpool(std::move(threadpool_proto))}
        , search_parameters_{}
        , logger_{std::move(logger)} {
        // Validate that the translator size matches or is smaller than data size
        if (translator_.size() > data_.size()) {
            throw ANNEXCEPTION(
                "Translator size ({}) exceeds data size ({})!",
                translator_.size(),
                data_.size()
            );
        }
    }

    /// @brief Getter method for logger
    svs::logging::logger_ptr get_logger() const { return logger_; }

    /// Return the number of independent entries in the index.
    size_t size() const {
        // NB: Index translation should always be kept in-sync with the number of valid
        // elements.
        return translator_.size();
    }

    /// Return the logical number of dimensions of the indexed vectors.
    size_t dimensions() const { return data_.dimensions(); }

    /// @brief Get the current search parameters.
    search_parameters_type get_search_parameters() const { return search_parameters_; }

    /// @brief Set the search parameters.
    void set_search_parameters(const search_parameters_type& params) {
        search_parameters_ = params;
    }

    ///// Index translation.

    /// @brief Get the internal ID mapped to be `e`.
    size_t translate_external_id(size_t e) const { return translator_.get_internal(e); }

    /// @brief Get the external ID mapped to be `i`.
    size_t translate_internal_id(size_t i) const { return translator_.get_external(i); }

    ///
    /// @brief Check whether the external ID `e` exists in the index.
    ///
    bool has_id(size_t e) const { return translator_.has_external(e); }

    ///
    /// @brief Get the raw data for external id `e`.
    ///
    auto get_datum(size_t e) const { return data_.get_datum(translate_external_id(e)); }

    /// @brief Iterate over all external IDs.
    template <typename F> void on_ids(F&& f) const {
        // Use the translator to iterate over all external IDs
        for (size_t i = 0; i < data_.size(); ++i) {
            if (is_valid(i)) {
                f(translator_.get_external(i));
            }
        }
    }

    /// @brief Get external IDs (compatibility method for dynamic_helper.h)
    auto external_ids() const {
        std::vector<size_t> ids;
        ids.reserve(size());
        on_ids([&ids](size_t id) { ids.push_back(id); });
        return ids;
    }

    /// @brief Add the points with the given external IDs to the dataset.
    ///
    /// When `delete_entries` is called, vectors are directly removed and their slots
    /// are marked as empty, making them immediately available for reuse.
    /// When `add_points` is called with the `reuse_empty` flag enabled,
    /// the memory is scanned from the beginning to locate and fill these empty entries with
    /// new points.
    ///
    /// @param points Dataset of points to add.
    /// @param external_ids The external IDs of the corresponding points. Must be a
    ///     container implementing forward iteration.
    /// @param reuse_empty A flag that determines whether to reuse empty entries that may
    /// exist after deletion. When enabled, scan from the beginning to
    /// find and fill these empty entries when adding new points.
    ///
    template <typename Points, class ExternalIds>
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
    /// Delete all IDs stored in the random-access container `ids`.
    ///
    /// Pre-conditions:
    /// * All indices present in `ids` belong to valid slots.
    ///
    /// Post-conditions:
    /// * Removed slots will not be returned in future calls `search`.
    /// * Removed slots are immediately available for reuse in future insertions.
    ///
    /// Implementation Notes:
    /// * The deletion directly removes the vectors from the dataset by marking their
    ///   slots as empty, making them immediately available for reuse.
    ///
    template <typename T> size_t delete_entries(const T& ids) {
        translator_.check_external_exist(ids.begin(), ids.end());
        for (auto i : ids) {
            delete_entry(translator_.get_internal(i));
        }
        translator_.delete_external(ids);
        return ids.size();
    }

    void delete_entry(size_t i) {
        SlotMetadata& meta = getindex(status_, i);
        assert(meta == SlotMetadata::Valid);
        meta = SlotMetadata::Empty;
        // Update first_empty_ to potentially reuse this slot sooner
        first_empty_ = std::min(first_empty_, i);
    }

    bool is_empty(size_t i) const { return status_[i] != SlotMetadata::Valid; }

    bool is_valid(size_t i) const { return status_[i] == SlotMetadata::Valid; }

    ///
    /// @brief Return all the valid internal IDs.
    ///
    /// Returns only entries that are currently valid (not empty).
    ///
    std::vector<size_t> nonmissing_indices() const {
        auto indices = std::vector<size_t>();
        indices.reserve(size());
        for (size_t i = 0, imax = status_.size(); i < imax; ++i) {
            if (is_valid(i)) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    ///
    /// @brief Compact the data structure.
    ///
    /// @param batch_size Granularity at which points are shuffled. Setting this higher can
    ///     improve performance but requires more working memory.
    ///
    void compact(size_t batch_size = 1'000) {
        // Step 1: Compute a prefix-sum matching each valid internal index to its new
        // internal index.
        //
        // In the returned data structure, an entry `j` at index `i` means that the
        // data at index `j` is to be moved to index `i`.
        auto new_to_old_id_map = nonmissing_indices();

        // Compact the data.
        data_.compact(lib::as_const_span(new_to_old_id_map), threadpool_, batch_size);

        ///// Finishing steps.
        size_t max_index = new_to_old_id_map.size();
        // Resize the data.
        data_.resize(max_index);
        first_empty_ = max_index;

        // Compact metadata and ID remapping.
        for (size_t new_id = 0; new_id < max_index; ++new_id) {
            auto old_id = getindex(new_to_old_id_map, new_id);
            // No work to be done if there was no remapping.
            if (new_id == old_id) {
                continue;
            }

            auto status = getindex(status_, old_id);
            status_[new_id] = status;
            if (status == SlotMetadata::Valid) {
                translator_.remap_internal_id(old_id, new_id);
            }
        }
        status_.resize(max_index);
    }

    ///// Consolidation

    /// @brief Consolidate the data structure (no-op for flat index with direct deletion).
    ///
    /// In the flat index implementation, deletion is performed directly, so this method
    /// is a no-op for compatibility with the dynamic index interface.
    ///
    void consolidate() {
        // No-op: direct deletion means no intermediate states to consolidate
    }

    ///// Saving

    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) {
        compact();

        // Save data structures and translation to config directory
        lib::save_to_disk(
            lib::SaveOverride([&](const lib::SaveContext& ctx) {
                return lib::SaveTable(
                    "dynamic_flat_config",
                    save_version,
                    {
                        {"name", lib::save(name())},
                        {"translation", lib::save(translator_, ctx)},
                    }
                );
            }),
            config_directory
        );

        // Save the dataset in the separate data directory
        lib::save_to_disk(data_, data_directory);
    }
    constexpr std::string_view name() const { return "dynamic flat index"; }

    ///// Thread Pool Management

    /// @brief Get the number of threads in the thread pool.
    size_t get_num_threads() const { return threadpool_.size(); }

    /// @brief Set the thread pool for the index.
    void set_threadpool(threads::ThreadPoolHandle threadpool) {
        threadpool_ = std::move(threadpool);
    }

    /// @brief Get a reference to the thread pool handle.
    threads::ThreadPoolHandle& get_threadpool_handle() { return threadpool_; }

    /// @brief Get a const reference to the thread pool handle.
    const threads::ThreadPoolHandle& get_threadpool_handle() const { return threadpool_; }

    ///// Search Interface

    static const size_t default_data_batch_size = 100'000;

    size_t compute_data_batch_size(const search_parameters_type& p) const {
        auto sz = p.data_batch_size_;
        if (sz == 0) {
            return default_data_batch_size;
        }
        return std::min(sz, data_.size());
    }

    size_t
    compute_query_batch_size(const search_parameters_type& p, size_t num_queries) const {
        auto sz = p.query_batch_size_;
        if (sz == 0) {
            return lib::div_round_up(num_queries, threadpool_.size());
        }
        return std::min(sz, num_queries);
    }

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
    /// @param cancel A predicate called during the search to determine if the search should
    /// be cancelled.
    //      Return ``true`` if the search should be cancelled. This functor must implement
    //      ``bool operator()()``. Note: This predicate should be thread-safe as it can be
    //      called concurrently by different threads during the search.
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
        // the results after translating to external IDs.
        scratch.cleanup();
        threads::parallel_for(
            threadpool_,
            threads::StaticPartition(queries.size()),
            [&](const auto& query_indices, uint64_t /*tid*/) {
                for (auto q : query_indices) {
                    const auto& neighbors = scratch.result(q);
                    for (size_t j = 0; j < neighbors.size(); ++j) {
                        // Translate internal ID to external ID before setting result
                        auto external_neighbor = Neighbor<size_t>(
                            translate_internal_id(neighbors[j].id()),
                            neighbors[j].distance()
                        );
                        result.set(external_neighbor, q, j);
                    }
                }
            }
        );
    }

    /// @brief Copy points from the source dataset into the specified slots.
    template <typename Points>
    void copy_points(const Points& points, const std::vector<size_t>& slots) {
        assert(points.size() == slots.size());
        for (size_t i = 0; i < points.size(); ++i) {
            data_.set_datum(slots[i], points.get_datum(i));
        }
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
        threads::parallel_for(
            threadpool_,
            threads::DynamicPartition{
                queries.size(), compute_query_batch_size(search_parameters, queries.size())
            },
            [&](const auto& query_indices, uint64_t /*tid*/) {
                // Broadcast the distance functor so each thread can process all queries
                // in its current batch.
                distance::BroadcastDistance distances{
                    extensions::distance(data_, distance_), query_indices.size()
                };

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
        auto accessor = data::GetDatumAccessor{};

        // Fix arguments
        for (size_t i = 0; i < query_indices.size(); ++i) {
            distance::maybe_fix_argument(
                distance_functors[i], queries.get_datum(query_indices[i])
            );
        }

        // Iterate over data indices
        for (auto data_index : data_indices) {
            // Check if request to cancel the search
            if (cancel()) {
                return;
            }

            // Skip empty entries
            if (!is_valid(data_index)) {
                continue;
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

    ///// Distance

    /// @brief Compute the distance between an external vector and a vector in the index.
    template <typename ExternalId, typename Query>
    double get_distance(const ExternalId& external_id, const Query& query) const {
        // Check if the external ID exists
        if (!has_id(external_id)) {
            throw ANNEXCEPTION(
                "ID {} is out of bounds for index of size {}!", external_id, size()
            );
        }
        // Verify dimensions match
        const size_t query_size = query.size();
        const size_t index_vector_size = dimensions();
        if (query_size != index_vector_size) {
            throw ANNEXCEPTION(
                "Incompatible dimensions. Query has {} while the index expects {}.",
                query_size,
                index_vector_size
            );
        }

        // Translate external ID to internal ID
        auto internal_id = translate_external_id(external_id);

        // Call extension for distance computation
        return svs::index::flat::extensions::get_distance_ext(
            data_, distance_, internal_id, query
        );
    }
};

///// Deduction Guides.
template <typename Data, typename Dist, typename ExternalIds>
DynamicFlatIndex(Data, const ExternalIds&, Dist, size_t) -> DynamicFlatIndex<Data, Dist>;

template <typename Data, typename Dist, typename ExternalIds, threads::ThreadPool Pool>
DynamicFlatIndex(Data, const ExternalIds&, Dist, Pool) -> DynamicFlatIndex<Data, Dist>;

namespace detail {

struct FlatStateLoader {
    ///// Loading
    static bool
    check_load_compatibility(std::string_view schema, const lib::Version& version) {
        // We can load from dynamic flat configs
        return (schema == "dynamic_flat_config" && version == lib::Version(0, 0, 0));
    }

    // Load the ID translator
    static FlatStateLoader
    load(const lib::LoadTable& table, bool debug_load_from_static, size_t assume_datasize) {
        if (debug_load_from_static) {
            return FlatStateLoader{IDTranslator::Identity(assume_datasize)};
        }

        return FlatStateLoader{
            svs::lib::load_at<IDTranslator>(table, "translation"),
        };
    }

    ///// Members
    IDTranslator translator_;
};

} // namespace detail

///
/// @brief Entry point for creating a Dynamic Flat index.
///
template <typename DataLoader, typename Distance, typename ThreadPoolProto>
auto auto_dynamic_assemble(
    const std::filesystem::path& config_path,
    DataLoader&& data_loader,
    Distance distance,
    ThreadPoolProto threadpool_proto,
    // Set this to `true` to use the identity map for ID translation.
    // This allows us to read files generated by the static index construction routines
    // to easily benchmark the static versus dynamic implementation.
    //
    // This is an internal API and should not be considered officially supported nor stable.
    bool debug_load_from_static = false,
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    // Load the dataset
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data =
        svs::detail::dispatch_load(std::forward<DataLoader>(data_loader), threadpool);

    // Load the ID translator from the config directory
    auto datasize = data.size();

    auto [translator] = lib::load_from_disk<detail::FlatStateLoader>(
        config_path, debug_load_from_static, datasize
    );

    // Validate the translator
    auto translator_size = translator.size();
    if (translator_size != datasize) {
        throw ANNEXCEPTION(
            "Translator has {} IDs but should have {}", translator_size, datasize
        );
    }

    return DynamicFlatIndex(
        std::move(data),
        std::move(translator),
        std::move(distance),
        std::move(threadpool),
        std::move(logger)
    );
}

} // namespace svs::index::flat