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

// stdlib
#include <memory>

// Include the flat index to spin-up exhaustive searches on demand.
#include "svs/index/flat/flat.h"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/core/loading.h"
#include "svs/core/logging.h"
#include "svs/core/medioid.h"
#include "svs/core/query_result.h"
#include "svs/core/recall.h"
#include "svs/core/translation.h"
#include "svs/index/vamana/consolidate.h"
#include "svs/index/vamana/dynamic_search_buffer.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/index.h"
#include "svs/index/vamana/vamana_build.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/threads.h"

namespace svs::index::vamana {

/////
///// MutableVamanaIndex
/////

///
/// Metadata tracking the state of a particular data index.
/// The following states have the given meaning for their corresponding slot:
///
/// * Valid: Valid and present in the associated dataset.
/// * Deleted: Exists in the associated dataset, but should be considered as "deleted"
/// and not returned from any search algorithms.
/// * Empty: Non-existant and unreachable from standard entry points.
///
/// Only used for `MutableVamanaIndex`.
///
enum class SlotMetadata : uint8_t { Empty = 0x00, Valid = 0x01, Deleted = 0x02 };

template <SlotMetadata Metadata> inline constexpr std::string_view name();
template <> inline constexpr std::string_view name<SlotMetadata::Empty>() {
    return "Empty";
}
template <> inline constexpr std::string_view name<SlotMetadata::Valid>() {
    return "Valid";
}
template <> inline constexpr std::string_view name<SlotMetadata::Deleted>() {
    return "Deleted";
}

// clang-format off
inline constexpr std::string_view name(SlotMetadata metadata) {
    #define SVS_SWITCH_RETURN(x) case x: { return name<x>(); }
    switch (metadata) {
        SVS_SWITCH_RETURN(SlotMetadata::Empty)
        SVS_SWITCH_RETURN(SlotMetadata::Valid)
        SVS_SWITCH_RETURN(SlotMetadata::Deleted)
    }
    #undef SVS_SWITCH_RETURN
    throw ANNEXCEPTION("Unreachable!");
}
// clang-format on

class ValidBuilder {
  public:
    ValidBuilder(const std::vector<SlotMetadata>& status)
        : status_{status} {}

    template <typename I>
    constexpr PredicatedSearchNeighbor<I> operator()(I i, float distance) const {
        bool invalid = getindex(status_, i) == SlotMetadata::Deleted;
        // This neighbor should be skipped if the metadata corresponding to the given index
        // marks this slot as deleted.
        return PredicatedSearchNeighbor<I>(i, distance, !invalid);
    }

  private:
    const std::vector<SlotMetadata>& status_;
};

template <graphs::MemoryGraph Graph, typename Data, typename Dist>
class MutableVamanaIndex {
  public:
    // Traits
    static constexpr bool supports_insertions = true;
    static constexpr bool supports_deletions = true;
    static constexpr bool supports_saving = true;
    static constexpr bool needs_id_translation = true;

    // Type Aliases
    using Idx = typename Graph::index_type;
    using internal_id_type = Idx;
    using external_id_type = size_t;
    using value_type = typename Data::value_type;
    using const_value_type = typename Data::const_value_type;
    static constexpr size_t extent = Data::extent;

    using distance_type = Dist;
    using search_buffer_type = MutableBuffer<Idx, distance::compare_t<Dist>>;

    using graph_type = Graph;
    using data_type = Data;
    using entry_point_type = std::vector<Idx>;
    /// The type of the configurable search parameters.
    using search_parameters_type = VamanaSearchParameters;
    using inner_scratch_type =
        svs::tag_t<extensions::single_search_setup>::result_t<Data, Dist>;
    using scratchspace_type = SearchScratchspace<search_buffer_type, inner_scratch_type>;

    // Members
  private:
    // Invariants:
    // * The ID translator should track only valid IDs.
    // TODO:
    // * Maybe merge some of the `status` metadata tracker with the IDTranslator to reduce
    //   memory requirements. There are probably some bits we can reclaim there to
    //   facilitate that.

    graph_type graph_;
    data_type data_;
    entry_point_type entry_point_;
    std::vector<SlotMetadata> status_;
    IDTranslator translator_;

    // Thread local data structures.
    distance_type distance_;
    threads::NativeThreadPool threadpool_;
    lib::ReadWriteProtected<VamanaSearchParameters> search_parameters_;

    // Configurations
    size_t construction_window_size_;
    size_t max_candidates_;
    size_t prune_to_;
    float alpha_ = 1.2;
    bool use_full_search_history_ = true;

    // Methods
  public:
    // This is because some dataset may not yet support single-searching, which is required
    // by the BatchIterator.
    SVS_TEMPORARY_DISABLE_SINGLE_SEARCH static constexpr bool
    temporary_disable_batch_iterator() {
        return extensions::temporary_disable_single_search<data_type>();
    }

    // Constructors
    template <typename ExternalIds>
    MutableVamanaIndex(
        Graph graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        const ExternalIds& external_ids,
        threads::NativeThreadPool threadpool
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{entry_point}
        , status_(data_.size(), SlotMetadata::Valid)
        , translator_()
        , distance_{std::move(distance_function)}
        , threadpool_{std::move(threadpool)}
        , search_parameters_{vamana::construct_default_search_parameters(data_)}
        , construction_window_size_{2 * graph.max_degree()} {
        translator_.insert(external_ids, threads::UnitRange<Idx>(0, external_ids.size()));
    }

    template <typename ExternalIds>
    MutableVamanaIndex(
        Graph graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        const ExternalIds& external_ids,
        size_t num_threads
    )
        : MutableVamanaIndex(
              std::move(graph),
              std::move(data),
              entry_point,
              std::move(distance_function),
              external_ids,
              threads::NativeThreadPool(num_threads)
          ) {}

    ///
    /// Build a graph from scratch.
    ///
    template <typename ExternalIds>
    MutableVamanaIndex(
        const VamanaBuildParameters& parameters,
        Data data,
        const ExternalIds& external_ids,
        Dist distance_function,
        size_t num_threads
    )
        : graph_(Graph{data.size(), parameters.graph_max_degree})
        , data_(std::move(data))
        , entry_point_{}
        , status_(data_.size(), SlotMetadata::Valid)
        , translator_()
        , distance_(std::move(distance_function))
        , threadpool_(num_threads)
        , search_parameters_(vamana::construct_default_search_parameters(data_))
        , construction_window_size_(parameters.window_size)
        , max_candidates_(parameters.max_candidate_pool_size)
        , prune_to_(parameters.prune_to)
        , alpha_(parameters.alpha)
        , use_full_search_history_{parameters.use_full_search_history} {
        // Setup the initial translation of external to internal ids.
        translator_.insert(external_ids, threads::UnitRange<Idx>(0, external_ids.size()));

        // Compute the entry point.
        entry_point_.push_back(extensions::compute_entry_point(data_, threadpool_));

        // Perform graph construction.
        auto sp = get_search_parameters();
        auto prefetch_parameters =
            GreedySearchPrefetchParameters{sp.prefetch_lookahead_, sp.prefetch_step_};
        auto builder = VamanaBuilder(
            graph_, data_, distance_, parameters, threadpool_, prefetch_parameters
        );
        builder.construct(1.0f, entry_point_[0]);
        builder.construct(parameters.alpha, entry_point_[0]);
    }

    /// @brief Post re-load constructor.
    ///
    /// Preconditions
    ///
    /// * data.size() == graph.n_nodes(): The graph and the data have the same number of
    ///   entries.
    /// * The data and graph were saved with no "holes". In otherwords, the index was
    ///   consolidated and compacted prior to saving.
    /// * The span of internal ID's in translator covers exactly ``[0, data.size())``.
    MutableVamanaIndex(
        const VamanaIndexParameters& config,
        data_type data,
        graph_type graph,
        const Dist& distance_function,
        IDTranslator translator,
        threads::NativeThreadPool threadpool
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{lib::narrow<Idx>(config.entry_point)}
        , status_{data_.size(), SlotMetadata::Valid}
        , translator_{std::move(translator)}
        , distance_{distance_function}
        , threadpool_{std::move(threadpool)}
        , search_parameters_{config.search_parameters}
        , construction_window_size_{config.build_parameters.window_size}
        , max_candidates_{config.build_parameters.max_candidate_pool_size}
        , prune_to_{config.build_parameters.prune_to}
        , alpha_{config.build_parameters.alpha}
        , use_full_search_history_{config.build_parameters.use_full_search_history} {}

    ///// Scratchspace
    scratchspace_type scratchspace(const search_parameters_type& sp) const {
        return scratchspace_type{
            search_buffer_type(
                sp.buffer_config_,
                distance::comparator(distance_),
                sp.search_buffer_visited_set_
            ),
            extensions::single_search_setup(data_, distance_),
            {sp.prefetch_lookahead_, sp.prefetch_step_}};
    }

    scratchspace_type scratchspace() const { return scratchspace(get_search_parameters()); }

    ///// Accessors

    /// @brief Get the alpha value used for pruning while mutating the graph.
    float get_alpha() const { return alpha_; }
    /// @brief Set the alpha value used for pruning while mutating the graph.
    void set_alpha(float alpha) { alpha_ = alpha; }

    /// @brief Get the ``graph_max_degree`` used while mutating the graph.
    size_t get_graph_max_degree() const { return graph_.max_degree(); }

    /// @brief Get the max candidate pool size used while mutating the graph.
    size_t get_max_candidates() const { return max_candidates_; }
    /// @brief Set the max candidate pool size to be used while mutating the graph.
    void set_max_candidates(size_t max_candidates) { max_candidates_ = max_candidates; }
    /// @brief Get the prune_to value used while mutating the graph.
    size_t get_prune_to() const { return prune_to_; }
    /// @brief Set the prune_to value to be used while mutating the graph.
    void set_prune_to(size_t prune_to) { prune_to_ = prune_to; }

    /// @brief Get the window size used while mutating the graph.
    size_t get_construction_window_size() const { return construction_window_size_; }
    /// @brief Set the window size to be used while mutating the graph.
    void set_construction_window_size(size_t window_size) {
        construction_window_size_ = window_size;
    }

    /// @brief Return whether the full search history is being used while mutating
    /// the graph.
    bool get_full_search_history() const { return use_full_search_history_; }
    /// @brief Enable using the full search history for candidate generation while
    /// mutating the graph.
    void set_full_search_history(bool enable) { use_full_search_history_ = enable; }


    ///// Index translation.

    ///
    /// @brief Get the internal ID mapped to be `e`.
    ///
    /// @param e The external ID to translate to an internal ID.
    ///
    /// Requires that mapping for `e` exists. Otherwise, all bets are off.
    ///
    /// @see has_id, translate_internal_id
    ///
    Idx translate_external_id(size_t e) const { return translator_.get_internal(e); }

    ///
    /// @brief Check whether the external ID `e` exists in the index.
    ///
    bool has_id(size_t e) const { return translator_.has_external(e); }

    ///
    /// @brief Get the external ID mapped to be `i`.
    ///
    /// @param i The internal ID to translate to an external ID.
    ///
    /// Requires that mapping for `i` exists. Otherwise, all bets are off.
    ///
    size_t translate_internal_id(Idx i) const { return translator_.get_external(i); }

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

    ///
    /// @brief Return a vector of all valid external IDs present in the index.
    ///
    std::vector<size_t> external_ids() const {
        std::vector<size_t> ids{};
        on_ids([&ids](size_t id) { ids.push_back(id); });
        return ids;
    }

    /// @brief Return the number of **valid** (non-deleted) entries in the index.
    size_t size() const {
        // NB: Index translation should always be kept in-sync with the number of valid
        // elements.
        return translator_.size();
    }

    ///
    /// @brief Translate in-place a collection of internal IDs to external IDs.
    ///
    /// @param ids The ``DenseArray`` of internal IDs to modify.
    ///
    /// Modifies each entry in `ids` in place, assumes that entry is an internal ID and
    /// remaps it to its external ID.
    ///
    /// This is used as a post-processing step following search to return the correct
    /// external neighbors to the caller, allowing inner search routines to simply return
    /// local IDs.
    ///
    /// Several implementation notes:
    /// (1) This is definitely not safe to call multiple times on the same array for obvious
    ///     reasons.
    ///
    /// (2) All entries in `ids` should have valid translations. Otherwise, this function's
    ///     behavior is undefined.
    ///
    template <class Dims, class Base>
        requires(std::tuple_size_v<Dims> == 2)
    void translate_to_external(DenseArray<size_t, Dims, Base>& ids) {
        // N.B.: lib::narrow_cast should be valid because the origin of the IDs is internal.
        threads::run(
            threadpool_,
            threads::StaticPartition{getsize<0>(ids)},
            [&](const auto is, uint64_t /*tid*/) {
                for (auto i : is) {
                    for (size_t j = 0, jmax = getsize<1>(ids); j < jmax; ++j) {
                        auto internal = lib::narrow_cast<Idx>(ids.at(i, j));
                        ids.at(i, j) = translate_internal_id(internal);
                    }
                }
            }
        );
    }

    ///
    /// @brief Get the raw data for external id `e`.
    ///
    auto get_datum(size_t e) const { return data_.get_datum(translate_external_id(e)); }

    ///
    /// @brief Return the dimensionality of the stored dataset.
    ///
    /// TODO (MH): This somewhat limits us to using only R^n type datasets. I'd like to see
    /// this generalized somewhat.
    ///
    size_t dimensions() const { return data_.dimensions(); }

    // Return a `greedy_search` compatible builder for this index.
    // This is an internal method, mostly used to help implement the batch iterator.
    ValidBuilder internal_search_builder() const { return ValidBuilder{status_}; }

    auto greedy_search_closure(
        GreedySearchPrefetchParameters prefetch_parameters,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) const {
        return [&, prefetch_parameters](
                   const auto& query, auto& accessor, auto& distance, auto& buffer
               ) {
            // Perform the greedy search using the provided resources.
            greedy_search(
                graph_,
                data_,
                accessor,
                query,
                distance,
                buffer,
                vamana::EntryPointInitializer<Idx>{lib::as_const_span(entry_point_)},
                internal_search_builder(),
                prefetch_parameters,
                cancel
            );
            // Take a pass over the search buffer to remove any deleted elements that
            // might remain.
            buffer.cleanup();
        };
    }

    // Single Search
    template <typename Query>
    void search(const Query& query, scratchspace_type& scratch) const {
        extensions::single_search(
            data_,
            scratch.buffer,
            scratch.scratch,
            query,
            greedy_search_closure(scratch.prefetch_parameters)
        );
    }

    template <typename I, data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<I> results,
        const Queries& queries,
        const search_parameters_type& sp,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        threads::run(
            threadpool_,
            threads::StaticPartition{queries.size()},
            [&](const auto is, uint64_t SVS_UNUSED(tid)) {
                size_t num_neighbors = results.n_neighbors();
                auto buffer =
                    search_buffer_type{sp.buffer_config_, distance::comparator(distance_)};

                auto prefetch_parameters = GreedySearchPrefetchParameters{
                    sp.prefetch_lookahead_, sp.prefetch_step_};

                // Legalize search buffer for this search.
                if (buffer.target() < num_neighbors) {
                    buffer.change_maxsize(num_neighbors);
                }
                auto scratch = extensions::per_thread_batch_search_setup(data_, distance_);

                extensions::per_thread_batch_search(
                    data_,
                    buffer,
                    scratch,
                    queries,
                    results,
                    threads::UnitRange{is},
                    greedy_search_closure(prefetch_parameters, cancel),
                    cancel
                );
            }
        );

        // Check if request to cancel the search
        if (cancel()) {
            return;
        }

        // After the search procedure, the indices in `results` are internal.
        // Perform one more pass to convert these to external ids.
        translate_to_external(results.indices());
    }

    ///
    /// @brief Return a unique instance of the distance function.
    ///
    Dist distance_function() const { return threads::shallow_copy(distance_); }

    ///
    /// Perform an exhaustive search on the current state of the index.
    /// Useful to understand how well the graph search is doing after index mutation.
    ///
    template <typename QueryType, typename I>
    void exhaustive_search(
        const data::ConstSimpleDataView<QueryType>& queries,
        size_t num_neighbors,
        QueryResultView<I> result
    ) {
        auto temp_index = temporary_flat_index(data_, distance_, threadpool_);
        temp_index.search(queries, num_neighbors, result, [&](size_t i) {
            return getindex(status_, i) == SlotMetadata::Valid;
        });

        // After the search procedure, the indices in `results` are internal.
        // Perform one more pass to convert these to external ids.
        translate_to_external(result.indices());
    }

    ///
    /// Descriptive Name
    ///
    // TODO (Mark): Make descriptions better.
    constexpr std::string_view name() const { return "dynamic vamana index"; }

    ///// Mutable Interface

    template <data::ImmutableMemoryDataset Points>
    void copy_points(const Points& points, const std::vector<size_t>& slots) {
        assert(points.size() == slots.size());
        threads::run(
            threadpool_,
            threads::StaticPartition{slots.size()},
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    data_.set_datum(slots[i], points.get_datum(i));
                }
            }
        );
    }

    ///
    /// @brief Clear the adjacency lists for the given local ids.
    ///
    /// This ensures that during the rebuild-phase, we don't get any zombie (previously
    /// deleted nodes) occuring in the new adjacency lists.
    ///
    template <std::integral I> void clear_lists(const std::vector<I>& local_ids) {
        threads::run(
            threadpool_,
            threads::StaticPartition(local_ids),
            [&](const auto& thread_local_ids, uint64_t /*tid*/) {
                for (auto id : thread_local_ids) {
                    graph_.clear_node(id);
                }
            }
        );
    }

    ///
    /// @brief Add the points with the given external IDs to the dataset.
    ///
    /// @param points Dataset of points to add.
    /// @param external_ids The external IDs of the corresponding points. Must be a
    ///     container implementing forward iteration.
    ///
    template <data::ImmutableMemoryDataset Points, class ExternalIds>
    std::vector<size_t> add_points(const Points& points, const ExternalIds& external_ids) {
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
        for (size_t i = 0, imax = status_.size(); i < imax; ++i) {
            if (status_[i] == SlotMetadata::Empty) {
                slots.push_back(i);
            }
            if (slots.size() == num_points) {
                have_room = true;
                break;
            }
        }

        // Check if we have enough indices. If we don't, we need to resize the data and
        // the graph.
        if (!have_room) {
            size_t needed = num_points - slots.size();
            size_t current_size = data_.size();
            size_t new_size = current_size + needed;
            data_.resize(new_size);

            // Graph resizing marked as un-safe because graph contain internal references
            // and thus it's not a good idea to go around shrinking the graph without care.
            //
            // However, we are only growing here, so resizing will not change any
            // invariants.
            graph_.unsafe_resize(new_size);
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

        // Copy the given points into the data and clear the adjacency lists for the graph.
        copy_points(points, slots);
        clear_lists(slots);

        // Patch in the new neighbors.
        auto parameters = VamanaBuildParameters{
            alpha_,
            graph_.max_degree(),
            construction_window_size_,
            max_candidates_,
            prune_to_,
            use_full_search_history_};

        auto sp = get_search_parameters();
        auto prefetch_parameters =
            GreedySearchPrefetchParameters{sp.prefetch_lookahead_, sp.prefetch_step_};
        VamanaBuilder builder{
            graph_, data_, distance_, parameters, threadpool_, prefetch_parameters};
        builder.construct(alpha_, entry_point(), slots, logging::Level::Trace);
        // Mark all added entries as valid.
        for (const auto& i : slots) {
            status_[i] = SlotMetadata::Valid;
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
    /// * Deleted slots will not be returned in future calls `search`.
    ///
    /// Implementation Nodes:
    /// * The deletion that happens is a "soft" deletion. This means that the corresponding
    ///   entries are still present in both the dataset and the graph, and will be navigated
    ///   through during searched.
    ///
    ///   However, entries marked as `deleted` will not be returned from searches.
    ///
    /// * Delete consolidation should happen once a large enough percentage of slots have
    ///   been soft deleted.
    ///
    ///   Delete consolidation performs the actual removal of deleted entries from the
    ///   graph.
    ///
    template <typename T> void delete_entries(const T& ids) {
        translator_.check_external_exist(ids.begin(), ids.end());
        for (auto i : ids) {
            delete_entry(translator_.get_internal(i));
        }
        translator_.delete_external(ids);
    }

    void delete_entry(size_t i) {
        SlotMetadata& meta = getindex(status_, i);
        assert(meta == SlotMetadata::Valid);
        meta = SlotMetadata::Deleted;
    }

    bool is_deleted(size_t i) const { return status_[i] != SlotMetadata::Valid; }

    Idx entry_point() const {
        assert(entry_point_.size() == 1);
        return entry_point_[0];
    }

    ///
    /// @brief Return all the non-missing internal IDs.
    ///
    /// This includes both valid and soft-deleted entries.
    ///
    std::vector<Idx> nonmissing_indices() const {
        auto indices = std::vector<Idx>();
        indices.reserve(size());
        for (size_t i = 0, imax = status_.size(); i < imax; ++i) {
            if (!is_deleted(i)) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    ///
    /// @brief Compact the data and the graph.
    ///
    /// @param batch_size Granularity at which points are shuffled. Setting this higher can
    ///     improve performance but requires more working memory.
    ///
    void compact(Idx batch_size = 1'000) {
        // Step 1: Compute a prefix-sum matching each valid internal index to its new
        // internal index.
        //
        // In the returned data structure, an entry `j` at index `i` means that the
        // data at index `j` is to be moved to index `i`.
        auto new_to_old_id_map = nonmissing_indices();

        // Construct an associative data structure to facilitate graph adjacency list
        // remapping.
        auto old_to_new_id_map = tsl::robin_map<Idx, Idx>{};
        for (Idx new_id = 0, imax = new_to_old_id_map.size(); new_id < imax; ++new_id) {
            Idx old_id = new_to_old_id_map.at(new_id);
            old_to_new_id_map.insert({old_id, new_id});
        }

        // Compact the data.
        data_.compact(lib::as_const_span(new_to_old_id_map), threadpool_, batch_size);

        // Manually compact the graph.
        auto temp_graph = graphs::SimpleGraph<Idx>(batch_size, graph_.max_degree());

        // TODO: Write helper classes to do this partitioning.
        Idx start = 0;
        Idx max_index = new_to_old_id_map.size();
        while (start < max_index) {
            Idx stop = std::min(start + batch_size, max_index);
            // Remapping of start index to stop index.
            auto batch_to_new_id_map = threads::UnitRange{start, stop};
            auto this_batch = batch_to_new_id_map.eachindex();

            // Copy the graph into the temporary buffer and remap the IDs.
            threads::run(
                threadpool_,
                threads::StaticPartition(this_batch),
                [&](const auto& batch_ids, uint64_t /*tid*/) {
                    std::vector<Idx> buffer{};
                    for (auto batch_id : batch_ids) {
                        auto new_id = batch_to_new_id_map[batch_id];
                        auto old_id = new_to_old_id_map[new_id];

                        const auto& list = graph_.get_node(old_id);
                        buffer.resize(list.size());

                        // Transform the adjacency list from old to new.
                        std::transform(
                            list.begin(),
                            list.end(),
                            buffer.begin(),
                            [&old_to_new_id_map](Idx old_id) {
                                return old_to_new_id_map.at(old_id);
                            }
                        );

                        temp_graph.replace_node(batch_id, buffer);
                    }
                }
            );

            // Copy the entries in the temporary graph to the original graph.
            threads::run(
                threadpool_,
                threads::StaticPartition(this_batch),
                [&](const auto& batch_ids, uint64_t /*tid*/) {
                    for (auto batch_id : batch_ids) {
                        auto new_id = batch_to_new_id_map[batch_id];
                        graph_.replace_node(new_id, temp_graph.get_node(batch_id));
                    }
                }
            );
            start = stop;
        }

        ///// Finishing steps.
        // Resize the graph and data.
        graph_.unsafe_resize(max_index);
        data_.resize(max_index);

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

        // Update entry points.
        for (auto& ep : entry_point_) {
            ep = old_to_new_id_map.at(ep);
        }
    }

    ///// Threading Interface
    static bool can_change_threads() { return true; }
    size_t get_num_threads() const { return threadpool_.size(); }
    void set_num_threads(size_t num_threads) {
        num_threads = std::max(num_threads, size_t(1));
        threadpool_.resize(num_threads);
    }

    ///// Window Interface
    VamanaSearchParameters get_search_parameters() const {
        return search_parameters_.get();
    }

    void populate_search_parameters(VamanaSearchParameters& parameters) const {
        parameters = get_search_parameters();
    }

    void set_search_parameters(const VamanaSearchParameters& parameters) {
        search_parameters_.set(parameters);
    }

    ///
    /// @brief Reset performance parameters to their default values for this index.
    ///
    /// Parameters affected are only those that modify throughput on a given architecture.
    /// Accuracy results should not change as a side-effect of calling this function.
    ///
    void reset_performance_parameters() {
        auto sp = get_search_parameters();
        auto prefetch_parameters = extensions::estimate_prefetch_parameters(data_);
        sp.prefetch_lookahead_ = prefetch_parameters.lookahead;
        sp.prefetch_step_ = prefetch_parameters.step;
        set_search_parameters(sp);
    }

    ///// Mutation
    void consolidate() {
        auto check_is_deleted = [&](size_t i) { return this->is_deleted(i); };
        std::function<bool(size_t)> valid = [&](size_t i) {
            return !(this->is_deleted(i));
        };

        // Determine if the entry point is deleted.
        // If so - we need to pick a new one.
        assert(entry_point_.size() == 1);
        auto entry_point = entry_point_[0];
        if (status_.at(entry_point) == SlotMetadata::Deleted) {
            auto logger = svs::logging::get();
            svs::logging::debug(logger, "Replacing entry point.");
            auto new_entry_point =
                extensions::compute_entry_point(data_, threadpool_, valid);
            svs::logging::debug(logger, "New point: {}", new_entry_point);
            assert(!is_deleted(new_entry_point));
            entry_point_[0] = new_entry_point;
        }

        // Perform graph consolidation.
        svs::index::vamana::consolidate(
            graph_,
            data_,
            threadpool_,
            prune_to_,
            max_candidates_,
            alpha_,
            distance_,
            check_is_deleted
        );

        // After consolidation - set all `Deleted` slots to `Empty`.
        for (auto& status : status_) {
            if (status == SlotMetadata::Deleted) {
                status = SlotMetadata::Empty;
            }
        }
    }

    ///// Saving

    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& graph_directory,
        const std::filesystem::path& data_directory
    ) {
        // Post-consolidation, all entries should be "valid".
        // Therefore, we don't need to save the slot metadata.
        consolidate();
        compact();

        // Save auxiliary data structures.
        lib::save_to_disk(
            lib::SaveOverride([&](const lib::SaveContext& ctx) {
                // Save the construction parameters.
                auto parameters = VamanaIndexParameters{
                    entry_point_.front(),
                    {alpha_,
                     graph_.max_degree(),
                     get_construction_window_size(),
                     get_max_candidates(),
                     prune_to_,
                     get_full_search_history()},
                    get_search_parameters()};

                return lib::SaveTable(
                    "vamana_dynamic_auxiliary_parameters",
                    save_version,
                    {
                        {"name", lib::save(name())},
                        {"parameters", lib::save(parameters, ctx)},
                        {"translation", lib::save(translator_, ctx)},
                    }
                );
            }),
            config_directory
        );

        // Save the dataset.
        lib::save_to_disk(data_, data_directory);
        // Save the graph.
        lib::save_to_disk(graph_, graph_directory);
    }

    /////
    ///// Calibrate
    /////

    // Return the maximum degree of the graph.
    size_t max_degree() const { return graph_.max_degree(); }

    // Experimental algorithm.
    //
    // Optimize search_window_size and capacity.
    // See calibrate.h for more details.
    template <
        data::ImmutableMemoryDataset Queries,
        data::ImmutableMemoryDataset GroundTruth>
    VamanaSearchParameters calibrate(
        const Queries& queries,
        const GroundTruth& groundtruth,
        size_t num_neighbors,
        double target_recall,
        const CalibrationParameters& calibration_parameters = {}
    ) {
        // Preallocate the destination for search.
        // Further, reference the search lambda in the recall lambda.
        auto results = svs::QueryResult<size_t>{queries.size(), num_neighbors};

        auto do_search = [&](const search_parameters_type& p) {
            this->search(results.view(), queries, p);
        };

        auto compute_recall = [&](const search_parameters_type& p) {
            // Calling `do_search` will mutate `results`.
            do_search(p);
            return svs::k_recall_at_n(results, groundtruth, num_neighbors, num_neighbors);
        };

        auto p = vamana::calibrate(
            calibration_parameters,
            *this,
            num_neighbors,
            target_recall,
            compute_recall,
            do_search
        );

        set_search_parameters(p);
        return p;
    }

    /// @brief Reconstruct vectors.
    ///
    /// Reconstruct each vector indexed by an external ID and store the results into
    /// ``dst``.
    ///
    /// Preconditions:
    /// - ``ids.size() == svs::getsize<0>(dst)``: Each ID has a corresponding entry in the
    ///     destination array.
    /// - All indices are valid external IDs for this index.
    /// - ``svs::getsize<1>(dst) == dimensions()``: The space allocated for each vector in
    ///     ``dst`` is correct.
    ///
    /// An exception will be thrown if any of these pre-conditions does not hold.
    /// If such an exception is thrown, the argument ``dst`` will be left unmodified.
    template <std::unsigned_integral I, svs::Arithmetic T>
    void reconstruct_at(data::SimpleDataView<T> dst, std::span<const I> ids) {
        const size_t ids_size = ids.size();
        const size_t dst_size = dst.size();
        const size_t dst_dims = dst.dimensions();

        if (ids_size != dst_size) {
            throw ANNEXCEPTION(
                "IDs span has size {} but destination has {} vectors!", ids_size, dst_size
            );
        }

        if (dst_dims != dimensions()) {
            throw ANNEXCEPTION(
                "Destination has dimensions {} but index is {}!", dst_dims, dimensions()
            );
        }

        // Bounds checking.
        for (size_t i = 0; i < ids_size; ++i) {
            I id = ids[i]; // inbounds by loop bounds.
            if (!has_id(id)) {
                throw ANNEXCEPTION("ID {} with value {} is out of bounds!", i, id);
            }
        }

        // Prerequisites checked - proceed with the operation.
        // TODO: Communicate the requested decompression type to the backend dataset to
        // allow more fine-grained specialization?
        auto threaded_function = [&](auto is, uint64_t SVS_UNUSED(tid)) {
            auto accessor = extensions::reconstruct_accessor(data_);
            for (auto i : is) {
                auto id = translate_external_id(ids[i]);
                dst.set_datum(i, accessor(data_, id));
            }
        };
        threads::run(threadpool_, threads::StaticPartition{ids_size}, threaded_function);
    }

    /// Invoke the provided callable with constant references to the contained graph, data,
    /// and entry points.
    ///
    /// This function is meant to provide a means for implementing experimental algorithms
    /// on the contained data structures.
    template <typename F> void experimental_escape_hatch(F&& f) const {
        std::invoke(SVS_FWD(f), graph_, data_, distance_, lib::as_const_span(entry_point_));
    }

    /////
    ///// Debug
    /////

    const Data& view_data() const { return data_; }
    const Graph& view_graph() const { return graph_; }

    ///
    /// @brief Verify the invariants of this data structure.
    ///
    /// @param allow_deleted Enable or disable deleted entries.
    ///
    void debug_check_invariants(bool allow_deleted) const {
        debug_check_size();
        debug_check_graph_consistency(allow_deleted);
    }

    ///
    /// Make sure that the capacities of the main data structures (graph, data, metadata)
    /// agree.
    ///
    void debug_check_size() const {
        size_t data_size = data_.size();
        auto throw_size_error = [=](const std::string& name, size_t other_size) {
            throw ANNEXCEPTION(
                "SIZE INVARIANT: Data size is {} but {} is {}.", data_size, name, other_size
            );
        };

        size_t graph_size = graph_.n_nodes();
        if (data_size != graph_size) {
            throw_size_error("graph", graph_size);
        }

        size_t status_size = status_.size();
        if (data_size != status_size) {
            throw_size_error("metadata", status_size);
        }
    }

    ///
    /// @brief Ensure the graph is in a consistent state.
    ///
    /// @param allow_deleted Flag to indicate if nodes marked as `Deleted` are okay
    ///    for consideration. Following a consolidation, this should be ``false``.
    ///    Otherwise, this should be ``true``.
    ///
    /// In this case, consistency means the that the adjacency lists for all non-deleted
    /// vertices contain only non-deleted vertices.
    ///
    /// This operation should be run after ``debug_check_size()`` to ensure that
    /// the sizes of the underlying data structures are consistent.
    ///
    void debug_check_graph_consistency(bool allow_deleted = false) const {
        auto is_valid = [&, allow_deleted = allow_deleted](size_t i) {
            const auto& metadata = status_[i];
            // Use a switch to get a compiler error is we add states to `SlotMetadata`.
            switch (metadata) {
                case SlotMetadata::Valid: {
                    return true;
                }
                case SlotMetadata::Deleted: {
                    return allow_deleted;
                }
                case SlotMetadata::Empty: {
                    return false;
                }
            }
            // Make GCC happy.
            return false;
        };

        for (size_t i = 0, imax = graph_.n_nodes(); i < imax; ++i) {
            if (!is_valid(i)) {
                continue;
            }

            size_t count = 0;
            for (auto j : graph_.get_node(i)) {
                if (!is_valid(j)) {
                    const auto& metadata = status_[j];
                    throw ANNEXCEPTION(
                        "Node number {} has an invalid ({}) neighbor ({}) at position {}!",
                        i,
                        index::vamana::name(metadata),
                        j,
                        count
                    );
                }
                count++;
            }
        }
    }
};

///// Deduction Guides.
// Guide for building.
template <typename Data, typename Dist, typename ExternalIds>
MutableVamanaIndex(const VamanaBuildParameters&, Data, const ExternalIds&, Dist, size_t)
    -> MutableVamanaIndex<graphs::SimpleBlockedGraph<uint32_t>, Data, Dist>;

namespace detail {

struct VamanaStateLoader {
    ///// Loading
    static bool
    check_load_compatibility(std::string_view schema, const lib::Version& version) {
        // We provide the option to load from a static index.
        return VamanaIndexParameters::check_load_compatibility(schema, version) ||
               (schema == "vamana_dynamic_auxiliary_parameters" &&
                version == lib::Version(0, 0, 0));
    }

    // Provide a compatibility path for loading static datasets.
    static VamanaStateLoader
    load(const lib::LoadTable& table, bool debug_load_from_static, size_t assume_datasize) {
        if (debug_load_from_static) {
            return VamanaStateLoader{
                lib::load<VamanaIndexParameters>(table),
                IDTranslator::Identity(assume_datasize)};
        }

        return VamanaStateLoader{
            SVS_LOAD_MEMBER_AT_(table, parameters),
            svs::lib::load_at<IDTranslator>(table, "translation"),
        };
    }

    ///// Members
    VamanaIndexParameters parameters_;
    IDTranslator translator_;
};

} // namespace detail

// Assembly
template <typename GraphLoader, typename DataLoader, typename Distance>
auto auto_dynamic_assemble(
    const std::filesystem::path& config_path,
    GraphLoader&& graph_loader,
    DataLoader&& data_loader,
    Distance distance,
    size_t num_threads,
    // Set this to `true` to use the identity map for ID translation.
    // This allows us to read files generated by the static index construction routines
    // to easily benchmark the static versus dynamic implementation.
    //
    // This is an internal API and should not be considered officially supported nor stable.
    bool debug_load_from_static = false
) {
    // Load the dataset
    auto threadpool = threads::NativeThreadPool(num_threads);
    auto data = svs::detail::dispatch_load(SVS_FWD(data_loader), threadpool);

    // Load the graph.
    auto graph = svs::detail::dispatch_load(SVS_FWD(graph_loader), threadpool);

    // Make sure the data and the graph have the same size.
    auto datasize = data.size();
    auto graphsize = graph.n_nodes();
    if (datasize != graphsize) {
        throw ANNEXCEPTION(
            "Reloaded data has {} nodes while the graph has {} nodes!", datasize, graphsize
        );
    }

    // // Unload the ID translator and config parameters.
    // auto reloader = lib::LoadOverride{[&](const lib::LoadTable& table) {
    //     // If loading from the static index, then the table we recieve is itself the
    //     // parameters table.
    //     //
    //     // There will also be no index translation, so we use the identity translation
    //     // since the internal and external IDs for the static index are the samen.
    //     if (debug_load_from_static) {
    //         return std::make_tuple(
    //             // TODO: Provide a better method of loading directly from a load-table
    //             // and correctly handling contexts.
    //             lib::load<VamanaIndexParameters>(table.unwrap(), ctx),
    //             IDTranslator(IDTranslator::Identity(datasize))
    //         );
    //     } else {
    //         return std::make_tuple(
    //             lib::load_at<VamanaIndexParameters>(table, "parameters", ctx),
    //             lib::load_at<IDTranslator>(table, "translation", ctx)
    //         );
    //     }
    // }};
    // auto [parameters, translator] = lib::load_from_disk(reloader, config_path);
    auto [parameters, translator] = lib::load_from_disk<detail::VamanaStateLoader>(
        config_path, debug_load_from_static, datasize
    );

    // Make sure that the translator covers all the IDs in the graph and data.
    auto translator_size = translator.size();
    if (translator_size != datasize) {
        throw ANNEXCEPTION(
            "Translator has {} IDs but should have {}", translator_size, datasize
        );
    }

    for (size_t i = 0; i < datasize; ++i) {
        if (!translator.has_internal(i)) {
            throw ANNEXCEPTION("Translator is missing internal id {}", i);
        }
    }

    // At this point, we should be completely validated.
    // Construct the index!
    return MutableVamanaIndex{
        parameters,
        std::move(data),
        std::move(graph),
        std::move(distance),
        std::move(translator),
        std::move(threadpool)};
}

} // namespace svs::index::vamana
