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
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>

// Include the flat index to spin-up exhaustive searches on demand.
#include "svs/index/flat/flat.h"

// svs
#include "svs/concepts/distance.h"
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/concurrent/graph.h"
#include "svs/core/loading.h"
#include "svs/core/logging.h"
#include "svs/core/medioid.h"
#include "svs/core/query_result.h"
#include "svs/core/recall.h"
#include "svs/concurrent/translation.h"
#include "svs/concurrent/consolidate.h"
#include "svs/concurrent/dynamic_search_buffer.h"
#include "svs/concurrent/greedy_search.h"
#include "svs/index/vamana/index.h"
#include "svs/concurrent/vamana_build.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/segmented_vector.h"
#include "svs/lib/threads.h"

namespace svs::index::vamana::concurrent {

namespace detail {
// Monotonically pull an atomic down to `value` if it is currently larger.
inline void atomic_min(std::atomic<size_t>& target, size_t value) {
    size_t prev = target.load(std::memory_order_relaxed);
    while (value < prev &&
           !target.compare_exchange_weak(
               prev, value, std::memory_order_acq_rel, std::memory_order_relaxed
           )) {
        // prev reloaded by compare_exchange_weak on failure.
    }
}

// Monotonically push an atomic up to `value` if it is currently smaller.
inline void atomic_max(std::atomic<size_t>& target, size_t value) {
    size_t prev = target.load(std::memory_order_relaxed);
    while (value > prev &&
           !target.compare_exchange_weak(
               prev, value, std::memory_order_acq_rel, std::memory_order_relaxed
           )) {
        // prev reloaded by compare_exchange_weak on failure.
    }
}
} // namespace detail

// Forward declaration
template <typename Index, typename QueryType> class BatchIterator;
template <graphs::MemoryGraph Graph, typename Data, typename Dist>
class MultiMutableVamanaIndex;

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
/// * Empty: Non-existent and unreachable from standard entry points.
///
/// Only used for `MutableVamanaIndex`.
///
enum class SlotMetadata : uint8_t {
    Empty = 0x00,
    Valid = 0x01,
    Deleted = 0x02,
    // Reserved by an in-flight add_points: slot owned by the adder, vector
    // copied, adjacency list being built. Invisible to search, consolidate,
    // and subsequent add_points until promoted to Valid.
    Pending = 0x04,
};

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
template <> inline constexpr std::string_view name<SlotMetadata::Pending>() {
    return "Pending";
}

// clang-format off
inline constexpr std::string_view name(SlotMetadata metadata) {
    #define SVS_SWITCH_RETURN(x) case x: { return name<x>(); }
    switch (metadata) {
        SVS_SWITCH_RETURN(SlotMetadata::Empty)
        SVS_SWITCH_RETURN(SlotMetadata::Valid)
        SVS_SWITCH_RETURN(SlotMetadata::Deleted)
        SVS_SWITCH_RETURN(SlotMetadata::Pending)
    }
    #undef SVS_SWITCH_RETURN
    throw ANNEXCEPTION("Unreachable!");
}
// clang-format on

class ValidBuilder {
  public:
    ValidBuilder(const lib::SegmentedVector<SlotMetadata>& status)
        : status_{status} {}

    template <typename I>
    constexpr PredicatedSearchNeighbor<I> operator()(I i, float distance) const {
        // A neighbor is returnable only if its slot is Valid. Deleted slots
        // must be skipped; Pending slots are reserved by an in-flight add and
        // their vectors/edges are not yet fully published. Empty slots should
        // never be reached via a valid edge, but we defend anyway.
        bool invalid =
            std::atomic_ref<SlotMetadata>(const_cast<SlotMetadata&>(getindex(status_, i)))
                .load(std::memory_order_acquire) != SlotMetadata::Valid;
        // This neighbor should be skipped if the metadata corresponding to the given index
        // marks this slot as deleted.
        return PredicatedSearchNeighbor<I>(i, distance, !invalid);
    }

  private:
    const lib::SegmentedVector<SlotMetadata>& status_;
};

template <graphs::MemoryGraph Graph, typename Data, typename Dist>
class MutableVamanaIndex {
    friend class MultiMutableVamanaIndex<Graph, Data, Dist>;

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
    // Grow-stable per-slot metadata: search reads status_[i] lock-free via ValidBuilder
    // while a concurrent add_points grows it. See svs/lib/segmented_vector.h.
    lib::SegmentedVector<SlotMetadata> status_;
    // a lower bound such that the append path
    // (reuse_empty == false) scans for Empty slots starting here.
    std::unique_ptr<std::atomic<size_t>> first_empty_{
        std::make_unique<std::atomic<size_t>>(0)};
    // lower bound such that no slot with index
    // < *first_reusable_ is Empty.
    std::unique_ptr<std::atomic<size_t>> first_reusable_{
        std::make_unique<std::atomic<size_t>>(0)};
    IDTranslator translator_;
    // Count of Valid slots. Maintained atomically in add_points/delete_entry.
    // Wrapped in unique_ptr because std::atomic is not movable.
    std::unique_ptr<std::atomic<size_t>> num_valid_{
        std::make_unique<std::atomic<size_t>>(0)};
    // Protects translator access: exclusive for writes (add/consolidate/compact),
    // shared for reads (delete/search). Wrapped in unique_ptr for movability.
    std::unique_ptr<std::shared_mutex> translator_mutex_{
        std::make_unique<std::shared_mutex>()};
    // Reserves slot ownership against compact(). Search and the other readers
    // (get_distance/reconstruct_at/batch-iterator) hold this shared so that
    // compact()'s shrink — which frees trailing segments of the grow-stable
    // containers — drains in-flight readers before destroying storage.
    // Writers (add_points, delete_entries, consolidate) also hold it shared;
    // compact() holds it exclusive.
    //
    // Lock acquisition order across the codebase:
    //   compact_mutex_ -> slot_alloc_mutex_   (never reversed)
    //   compact_mutex_ -> translator_mutex_   (never reversed)
    // slot_alloc_mutex_ and translator_mutex_ are never held simultaneously
    // (add_points takes them sequentially), so they have no relative order.
    std::unique_ptr<std::shared_mutex> compact_mutex_{
        std::make_unique<std::shared_mutex>()};
    // Writer-only mutex serializing slot allocation in add_points
    std::unique_ptr<std::mutex> slot_alloc_mutex_{std::make_unique<std::mutex>()};

    // Thread local data structures.
    distance_type distance_;
    threads::ThreadPoolHandle threadpool_;
    lib::ReadWriteProtected<VamanaSearchParameters> search_parameters_;

    // Configurations
    size_t construction_window_size_;
    size_t max_candidates_;
    size_t prune_to_;
    float alpha_ = 1.2;
    bool use_full_search_history_ = true;

    // Construction parameters
    VamanaBuildParameters build_parameters_{};

    // SVS logger for per index logging
    svs::logging::logger_ptr logger_;

    // Methods
  public:
    // Constructors
    template <typename ExternalIds, typename ThreadPoolProto>
    MutableVamanaIndex(
        Graph graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        const ExternalIds& external_ids,
        ThreadPoolProto threadpool_proto,
        // Optional logger parameter
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{entry_point}
        , status_(data_.size(), SlotMetadata::Valid)
        , first_empty_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , first_reusable_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , translator_()
        , num_valid_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , distance_{std::move(distance_function)}
        , threadpool_{threads::as_threadpool(std::move(threadpool_proto))}
        , search_parameters_{vamana::construct_default_search_parameters(data_)}
        , construction_window_size_{2 * graph.max_degree()}
        // Ctor accept logger in parameter
        , logger_{std::move(logger)} {
        translator_.insert(external_ids, threads::UnitRange<Idx>(0, external_ids.size()));
        graph_.enable_reverse_edges();
        graph_.rebuild_reverse_edges(threadpool_);
    }

    ///
    /// Build a graph from scratch.
    ///
    template <typename ExternalIds, typename ThreadPoolProto>
    MutableVamanaIndex(
        const VamanaBuildParameters& parameters,
        Data data,
        const ExternalIds& external_ids,
        Dist distance_function,
        ThreadPoolProto threadpool_proto,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : graph_(Graph{data.size(), parameters.graph_max_degree})
        , data_(std::move(data))
        , entry_point_{}
        , status_(data_.size(), SlotMetadata::Valid)
        , first_empty_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , first_reusable_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , translator_()
        , num_valid_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , distance_(std::move(distance_function))
        , threadpool_(threads::as_threadpool(std::move(threadpool_proto)))
        , search_parameters_(vamana::construct_default_search_parameters(data_))
        , build_parameters_(parameters)
        , logger_{std::move(logger)} {
        // Verify and set defaults directly on the input parameters
        verify_and_set_default_index_parameters(build_parameters_, distance_function);

        // Set graph again as verify function might change graph_max_degree parameter
        graph_ = Graph{data_.size(), build_parameters_.graph_max_degree};
        construction_window_size_ = build_parameters_.window_size;
        max_candidates_ = build_parameters_.max_candidate_pool_size;
        prune_to_ = build_parameters_.prune_to;
        alpha_ = build_parameters_.alpha;
        use_full_search_history_ = build_parameters_.use_full_search_history;

        // Setup the initial translation of external to internal ids.
        translator_.insert(external_ids, threads::UnitRange<Idx>(0, external_ids.size()));

        // Compute the entry point.
        entry_point_.push_back(extensions::compute_entry_point(data_, threadpool_));

        // Perform graph construction.
        auto sp = get_search_parameters();
        auto prefetch_parameters =
            GreedySearchPrefetchParameters{sp.prefetch_lookahead_, sp.prefetch_step_};
        auto builder = VamanaBuilder(
            graph_,
            data_,
            distance_,
            build_parameters_,
            threadpool_,
            prefetch_parameters,
            logger_
        );
        builder.construct(1.0f, entry_point_[0], logging::Level::Trace, logger_);
        builder.construct(
            build_parameters_.alpha, entry_point_[0], logging::Level::Trace, logger_
        );

        graph_.enable_reverse_edges();
        graph_.rebuild_reverse_edges(threadpool_);
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
    template <threads::ThreadPool Pool>
    MutableVamanaIndex(
        const VamanaIndexParameters& config,
        data_type data,
        graph_type graph,
        const Dist& distance_function,
        IDTranslator translator,
        Pool threadpool,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{lib::narrow<Idx>(config.entry_point)}
        , status_{data_.size(), SlotMetadata::Valid}
        , first_empty_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , first_reusable_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , translator_{std::move(translator)}
        , num_valid_{std::make_unique<std::atomic<size_t>>(data_.size())}
        , distance_{distance_function}
        , threadpool_{std::move(threadpool)}
        , search_parameters_{config.search_parameters}
        , construction_window_size_{config.build_parameters.window_size}
        , max_candidates_{config.build_parameters.max_candidate_pool_size}
        , prune_to_{config.build_parameters.prune_to}
        , alpha_{config.build_parameters.alpha}
        , use_full_search_history_{config.build_parameters.use_full_search_history}
        , logger_{std::move(logger)} {
        graph_.enable_reverse_edges();
        graph_.rebuild_reverse_edges(threadpool_);
    }

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
    /// @brief Getter method for logger
    svs::logging::logger_ptr get_logger() const { return logger_; }

    /// @brief Get the alpha value used for pruning while mutating the graph.
    float get_alpha() const { return alpha_; }
    /// @brief Set the alpha value used for pruning while mutating the graph.
    void set_alpha(float alpha) { alpha_ = alpha; }

    /// @brief Get the ``graph_max_degree`` used while mutating the graph.
    size_t get_graph_max_degree() const { return graph_.max_degree(); }

    /// @brief Return the bytes allocated by each index component.
    ///
    /// Reports the capacity-based bytes reserved by the graph adjacency lists, the vector
    /// data, and the dynamic metadata (per-slot status, entry-point list, and the
    /// external/internal ID translation maps). Capacity-based accounting includes the
    /// block over-allocation so integrators can report the true memory footprint.
    MemoryBreakdown get_memory_breakdown() const {
        MemoryBreakdown usage{};
        usage.graph_bytes = svs::data::detail::dataset_allocated_bytes(graph_.get_data());
        usage.data_bytes = svs::data::detail::dataset_allocated_bytes(data_);

        size_t metadata_bytes = status_.capacity() * sizeof(SlotMetadata);
        metadata_bytes +=
            entry_point_.capacity() * sizeof(typename entry_point_type::value_type);
        // The IDTranslator holds two tsl::robin_map instances (external->internal and
        // internal->external), neither of which exposes its allocated byte count. We
        // approximate the storage as the id pair held in each of the two directions. This
        // ignores the maps' load-factor slack and control bytes, so it is an estimate of
        // the hash-map overhead that is accurate to within a few percent.
        metadata_bytes += 2 * translator_.size() *
                          (sizeof(IDTranslator::external_id_type) +
                           sizeof(IDTranslator::internal_id_type));
        usage.metadata_bytes = metadata_bytes;
        return usage;
    }

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
    //
    // The translator is a pair of hash maps mutated by `add_points` (insert),
    // `consolidate` (erase), and `compact` (remap) under `translator_mutex_` exclusive. A
    // hash-map read concurrent with an insert is a data race in the strict sense and a
    // crash in practice: an insert can rehash and free the bucket array a reader is walking.
    //
    // So every read of `translator_` happens under `translator_mutex_` shared. Each
    // translation operation comes in two flavours:
    //
    //   * `foo(...)`          -- takes the shared lock itself. Use this by default.
    //   * `unsafe_foo(...)`   -- assumes the caller already holds the shared lock (via
    //                            `lock_for_translation()`). Use this to translate a *batch*
    //                            of ids under one lock acquisition, and inside code that
    //                            already holds the lock.
    //
    // The split is not merely an optimization: `std::shared_mutex` is not recursive, so a
    // self-locking accessor called from a context that already holds the lock is a latent
    // deadlock (a writer arriving between the two shared acquisitions blocks the second).
    // That is why the batch paths below, and `BatchIterator::next`, use the `unsafe_`
    // variants under an explicit lock rather than paying per-id locking.

    ///
    /// @brief Get the internal ID mapped to be `e`.
    ///
    /// @param e The external ID to translate to an internal ID.
    ///
    /// Requires that mapping for `e` exists. Otherwise, all bets are off.
    ///
    /// @see has_id, translate_internal_id
    ///
    Idx translate_external_id(size_t e) const {
        std::shared_lock lock{*translator_mutex_};
        return unsafe_translate_external_id(e);
    }

    /// @copydoc translate_external_id
    /// Requires the caller to hold `lock_for_translation()`.
    Idx unsafe_translate_external_id(size_t e) const { return translator_.get_internal(e); }

    /// @brief Translate external ID, returning `default_val` if not mapped.
    ///
    /// Unlike `translate_external_id`, this does not throw on a missing key.
    /// Intended for best-effort readers (e.g. search buffer top-up) that may
    /// race with `consolidate()` erasing translator entries.
    Idx translate_external_id_or(size_t e, Idx default_val) const {
        std::shared_lock lock{*translator_mutex_};
        return unsafe_translate_external_id_or(e, default_val);
    }

    /// @copydoc translate_external_id_or
    /// Requires the caller to hold `lock_for_translation()`.
    Idx unsafe_translate_external_id_or(size_t e, Idx default_val) const {
        return translator_.get_internal_or(e, default_val);
    }

    ///
    /// @brief Check whether the external ID `e` exists in the index.
    ///
    bool has_id(size_t e) const {
        std::shared_lock lock{*translator_mutex_};
        return unsafe_has_id(e);
    }

    /// @copydoc has_id
    /// Requires the caller to hold `lock_for_translation()`.
    bool unsafe_has_id(size_t e) const {
        if (!translator_.has_external(e)) {
            return false;
        }
        // Check slot is not Deleted (deferred translator cleanup).
        auto internal = translator_.get_internal(e);
        return std::atomic_ref<SlotMetadata>(const_cast<SlotMetadata&>(status_[internal]))
                   .load(std::memory_order_acquire) == SlotMetadata::Valid;
    }

    ///
    /// @brief Get the external ID mapped to be `i`.
    ///
    /// @param i The internal ID to translate to an external ID.
    ///
    /// Requires that mapping for `i` exists. Otherwise, all bets are off.
    ///
    size_t translate_internal_id(Idx i) const {
        std::shared_lock lock{*translator_mutex_};
        return unsafe_translate_internal_id(i);
    }

    /// @copydoc translate_internal_id
    /// Requires the caller to hold `lock_for_translation()`.
    size_t unsafe_translate_internal_id(Idx i) const {
        // Use get_external_or to handle concurrent consolidate erasing entries.
        // If the entry was erased, return the internal ID as-is (stale result).
        return translator_.get_external_or(i, static_cast<size_t>(i));
    }

    ///
    /// @brief Call the functor with all external IDs in the index.
    ///
    /// @param f A functor with an overloaded ``operator()(size_t)`` method. Called on
    ///     each external ID in the index.
    ///
    /// The translator lock is held for the whole traversal, so `f` must not call back into
    /// a translation method that takes the lock itself, and must not mutate the index.
    ///
    template <typename F> void on_ids(F&& f) const {
        std::shared_lock lock{*translator_mutex_};
        unsafe_on_ids(SVS_FWD(f));
    }

    /// @copydoc on_ids
    /// Requires the caller to hold `lock_for_translation()`.
    template <typename F> void unsafe_on_ids(F&& f) const {
        // Skip entries whose slot is Deleted (deferred translator cleanup).
        for (auto pair : translator_) {
            auto internal = pair.second;
            if (std::atomic_ref<SlotMetadata>(const_cast<SlotMetadata&>(status_[internal]))
                    .load(std::memory_order_acquire) == SlotMetadata::Valid) {
                f(pair.first);
            }
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
    size_t size() const { return num_valid_->load(std::memory_order_acquire); }

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
        std::shared_lock lock{*translator_mutex_};
        threads::parallel_for(
            threadpool_,
            threads::StaticPartition{getsize<0>(ids)},
            [&](const auto is, uint64_t /*tid*/) {
                for (auto i : is) {
                    for (size_t j = 0, jmax = getsize<1>(ids); j < jmax; ++j) {
                        auto internal = lib::narrow_cast<Idx>(ids.at(i, j));
                        ids.at(i, j) = unsafe_translate_internal_id(internal);
                    }
                }
            }
        );
    }

    ///
    /// @brief Get the raw data for external id `e`.
    ///
    auto get_datum(size_t e) const {
        // Lock order: compact_mutex_ then translator_mutex_ (global order).
        std::shared_lock compact_lock{*compact_mutex_};
        std::shared_lock lock{*translator_mutex_};
        if (!translator_.has_external(e)) {
            throw ANNEXCEPTION("External ID {} not found in index!", e);
        }
        return data_.get_datum(translator_.get_internal(e));
    }

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

    /// @brief RAII reader lock guarding data_/graph_ against compact()'s shrink
    /// (which frees segments). Used by BatchIterator::next() to protect the
    /// greedy traversal — mirrors the shared lock taken by search(). Growth by
    /// add_points needs no lock (grow-stable SegmentedVector storage).
    ///
    /// Acquire this only around graph traversal, and release it before
    /// acquiring lock_for_translation(): the two must never be held nested in
    /// the compact->translator order reversed, which would invert the global
    /// lock order (compact -> translator) and deadlock against compact.
    [[nodiscard]] std::shared_lock<std::shared_mutex> lock_for_search() const {
        return std::shared_lock<std::shared_mutex>(*compact_mutex_);
    }

    /// @brief RAII reader lock guarding translator_ against erase/remap by
    /// consolidate/compact. Used by BatchIterator::next() to protect
    /// internal->external ID translation.
    [[nodiscard]] std::shared_lock<std::shared_mutex> lock_for_translation() const {
        return std::shared_lock<std::shared_mutex>(*translator_mutex_);
    }

    auto greedy_search_closure(
        GreedySearchPrefetchParameters prefetch_parameters,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) const {
        return [&, prefetch_parameters](
                   const auto& query, auto& accessor, auto& distance, auto& buffer
               ) {
            // Perform the greedy search using the provided resources.
            concurrent::greedy_search(
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
    void search(
        const Query& query,
        scratchspace_type& scratch,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) const {
        // Hold compact_mutex_ shared so compact()'s shrink can't free segments
        // mid-traversal. add_points growth is lock-free (grow-stable storage).
        std::shared_lock compact_lock{*compact_mutex_};
        extensions::single_search(
            data_,
            scratch.buffer,
            scratch.scratch,
            query,
            greedy_search_closure(scratch.prefetch_parameters, cancel),
            *this
        );
    }

    template <typename I, data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<I> results,
        const Queries& queries,
        const search_parameters_type& sp,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        {
            // compact_mutex_ shared: blocks compact()'s segment-freeing shrink
            // during the traversal. Released before translate_to_external() takes
            // translator_mutex_ to keep the compact->translator lock order.
            std::shared_lock compact_lock{*compact_mutex_};
            threads::parallel_for(
                threadpool_,
                threads::StaticPartition{queries.size()},
                [&](const auto is, uint64_t SVS_UNUSED(tid)) {
                    size_t num_neighbors = results.n_neighbors();
                    auto buffer = search_buffer_type{
                        sp.buffer_config_, distance::comparator(distance_)};

                    auto prefetch_parameters = GreedySearchPrefetchParameters{
                        sp.prefetch_lookahead_, sp.prefetch_step_};

                    // Legalize search buffer for this search.
                    if (buffer.target_capacity() < num_neighbors) {
                        buffer.change_maxsize(num_neighbors);
                    }
                    auto scratch =
                        extensions::per_thread_batch_search_setup(data_, distance_);

                    extensions::per_thread_batch_search(
                        data_,
                        buffer,
                        scratch,
                        queries,
                        results,
                        threads::UnitRange{is},
                        greedy_search_closure(prefetch_parameters, cancel),
                        *this,
                        cancel
                    );
                }
            );
        }

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
        auto temp_index = temporary_flat_index(
            data_, distance_, threads::ThreadPoolReferenceWrapper(threadpool_)
        );
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
        threads::parallel_for(
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
    /// deleted nodes) occurring in the new adjacency lists.
    ///
    template <std::integral I> void clear_lists(const std::vector<I>& local_ids) {
        threads::parallel_for(
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
    //
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

        // Reserve slot ownership against compact(). Held for the entire call,
        // including the lock-free Phase 2-4 below; compact() takes this
        // exclusive and so will block until every in-flight add finishes.
        std::shared_lock compact_lock{*compact_mutex_};

        // Phase 1: reserve slots (Empty->Pending). Pending means
        // "reserved by an in-flight add"
        std::vector<size_t> slots{};
        slots.reserve(num_points);

        // Pre-allocation tail mark, restored on rollback so tail slots we consumed
        // are not stranded above first_empty_.
        size_t first_empty_before = first_empty_->load(std::memory_order_acquire);

        if (reuse_empty) {
            size_t s = first_reusable_->load(std::memory_order_acquire);
            for (; s < first_empty_before && slots.size() < num_points; ++s) {
                SlotMetadata expected = SlotMetadata::Empty;
                if (std::atomic_ref<SlotMetadata>(status_[s])
                        .compare_exchange_strong(
                            expected,
                            SlotMetadata::Pending,
                            std::memory_order_acq_rel,
                            std::memory_order_relaxed
                        )) {
                    slots.push_back(s);
                }
            }
            detail::atomic_max(*first_reusable_, s);
        }

        if (slots.size() < num_points) {
            std::lock_guard lock{*slot_alloc_mutex_};

            size_t s = first_empty_->load(std::memory_order_relaxed);
            size_t smax = status_.size();
            for (; s < smax && slots.size() < num_points; ++s) {
                if (status_[s] == SlotMetadata::Empty) {
                    std::atomic_ref<SlotMetadata>(status_[s])
                        .store(SlotMetadata::Pending, std::memory_order_release);
                    slots.push_back(s);
                }
            }

            if (slots.size() < num_points) {
                size_t needed = num_points - slots.size();
                size_t current_size = data_.size();
                size_t new_size = current_size + needed;
                data_.resize(new_size);
                graph_.unsafe_resize(new_size);
                status_.resize(new_size, SlotMetadata::Empty);
                for (size_t i = current_size; i < new_size; ++i) {
                    std::atomic_ref<SlotMetadata>(status_[i])
                        .store(SlotMetadata::Pending, std::memory_order_release);
                    slots.push_back(i);
                }
                s = new_size;
            }

            detail::atomic_max(*first_empty_, s);
        }
        assert(slots.size() == num_points);

        // Phase 2: Publish the id translation under translator_mutex_ exclusive
        // A Pending slot belongs to an in-flight adder and must
        // not be treated as stale — that would clobber the other adder's mapping.
        //
        // replace_stale_and_insert throws if any external id already
        // maps to a live slot.
        try {
            std::lock_guard lock{*translator_mutex_};
            translator_
                .replace_stale_and_insert(external_ids, slots, [this](auto internal) {
                    return std::atomic_ref<SlotMetadata>(
                               const_cast<SlotMetadata&>(status_[internal])
                           )
                               .load(std::memory_order_acquire) == SlotMetadata::Deleted;
                });
        } catch (...) {
            // Release the reserved slots back to Empty.
            for (auto s : slots) {
                std::atomic_ref<SlotMetadata>(status_[s])
                    .store(SlotMetadata::Empty, std::memory_order_release);
            }
            detail::atomic_min(*first_empty_, first_empty_before);
            if (!slots.empty()) {
                detail::atomic_min(*first_reusable_, slots.front());
            }
            throw;
        }

        // Phase 3: Lock-free data copy and adjacency clearing.
        // Slots are Pending: invisible to search (ValidBuilder filters),
        // reserved against other writers (Empty-slot scan skips Pending).
        copy_points(points, slots);
        clear_lists(slots);

        // Phase 4: Graph construction — runs without lock.
        // VamanaBuilder::construct() is thread-safe via per-node spinlock+seqlock.
        // note: VamanaBuilder constructor asserts graph_.n_nodes() == data_.size().
        // Both are grown together under the lock above, so this is always consistent.
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
            graph_,
            data_,
            distance_,
            parameters,
            threadpool_,
            prefetch_parameters,
            logger_,
            logging::Level::Trace};
        builder.construct(alpha_, entry_point(), slots, logging::Level::Trace, logger_);

        // Mark added entries as valid (unique slots per thread, no lock needed).
        for (const auto& i : slots) {
            std::atomic_ref<SlotMetadata>(status_[i])
                .store(SlotMetadata::Valid, std::memory_order_release);
        }
        num_valid_->fetch_add(slots.size(), std::memory_order_acq_rel);

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
    template <typename T> size_t delete_entries(const T& ids) {
        std::shared_lock compact_lock{*compact_mutex_};
        std::shared_lock lock{*translator_mutex_};
        size_t deleted = 0;
        for (auto i : ids) {
            if (!translator_.has_external(i)) {
                continue; // Already deleted + consolidated, or never existed.
            }
            auto internal = translator_.get_internal(i);
            if (is_deleted(internal)) {
                continue; // Already soft-deleted, translator entry not yet consolidated.
            }
            delete_entry(internal);
            ++deleted;
        }
        // Don't erase translator entries here — concurrent search may still
        // need them for translate_to_external(). Cleanup happens in
        // consolidate()/compact() when deleted slots become empty.
        return deleted;
    }

    void delete_entry(size_t i) {
        auto& meta = getindex(status_, i);
        auto ref = std::atomic_ref<SlotMetadata>(meta);
        // CAS Valid → Deleted. If the slot is Pending (concurrent adder still
        // in phase 2), wait for the adder to promote it to Valid before we
        // can soft-delete; otherwise the delete would be silently lost. Only
        // the thread that successfully transitions decrements num_valid_;
        // double-deletes silently no-op.
        for (;;) {
            SlotMetadata expected = SlotMetadata::Valid;
            if (ref.compare_exchange_strong(
                    expected,
                    SlotMetadata::Deleted,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed
                )) {
                num_valid_->fetch_sub(1, std::memory_order_acq_rel);
                return;
            }
            if (expected != SlotMetadata::Pending) {
                // Already Deleted or Empty — no-op.
                return;
            }
            // Pending: adder's Pending → Valid store is imminent; spin.
            svs::detail::pause();
        }
    }

    bool is_deleted(size_t i) const {
        // True only for slots that have been soft-deleted. Pending (in-flight
        // add) and Empty are NOT deleted: consolidate must not prune them out
        // of other nodes' adjacency lists, and search already filters
        // non-Valid slots via ValidBuilder.
        return std::atomic_ref<SlotMetadata>(const_cast<SlotMetadata&>(status_[i]))
                   .load(std::memory_order_acquire) == SlotMetadata::Deleted;
    }

    Idx entry_point() const {
        assert(entry_point_.size() == 1);
        return entry_point_[0];
    }

    ///
    /// @brief Return all internal IDs whose slot is Valid (live).
    ///
    /// Used by compact() to pick the surviving set. Pending slots (in-flight
    /// adds) are excluded — compact is only safe to run when the caller has
    /// ensured no Pending slots exist (compact holds translator_mutex_
    /// exclusive, which prevents a new add from entering phase 1, but an add
    /// that reached phase 2 before compact grabbed the lock may still be
    /// publishing status Pending → Valid; the compact caller must quiesce
    /// these adds first).
    std::vector<Idx> nonmissing_indices() const {
        auto indices = std::vector<Idx>();
        indices.reserve(size());
        for (size_t i = 0, imax = status_.size(); i < imax; ++i) {
            if (std::atomic_ref<SlotMetadata>(const_cast<SlotMetadata&>(status_[i]))
                    .load(std::memory_order_acquire) == SlotMetadata::Valid) {
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
        std::lock_guard compact_lock{*compact_mutex_};

        // Consolidate first, under the same exclusive lock. This folds any
        // outstanding soft-deletes into the graph.
        consolidate_locked();
        compact_locked(batch_size);
    }

    // Body of compact() with no compact_mutex_ locking. The caller MUST hold
    // compact_mutex_ exclusive
    void compact_locked(Idx batch_size = 1'000) {
        // The graph is fully remapped below; every reverse-edge entry would be stale.
        // Suppress recording during the remap and rebuild from the final graph at the end.
        if (auto* re = graph_.reverse_edges()) {
            re->set_recording(false);
        }

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
            // Edges to non-Valid (Deleted) slots are dropped — those slots
            // do not survive compaction, so the edge would dangle.
            threads::parallel_for(
                threadpool_,
                threads::StaticPartition(this_batch),
                [&](const auto& batch_ids, uint64_t /*tid*/) {
                    std::vector<Idx> buffer{};
                    for (auto batch_id : batch_ids) {
                        auto new_id = batch_to_new_id_map[batch_id];
                        auto old_id = new_to_old_id_map[new_id];

                        const auto& list = graph_.get_node(old_id);
                        buffer.clear();
                        buffer.reserve(list.size());

                        for (auto neighbor_old : list) {
                            auto it = old_to_new_id_map.find(neighbor_old);
                            if (it != old_to_new_id_map.end()) {
                                buffer.push_back(it->second);
                            }
                        }

                        temp_graph.replace_node(batch_id, buffer);
                    }
                }
            );

            // Copy the entries in the temporary graph to the original graph.
            threads::parallel_for(
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
        {
            std::lock_guard lock{*translator_mutex_};
            // Shrink the graph and data. compact_mutex_ is held exclusive for the
            // whole compact(), so all in-flight readers have drained — freeing
            // trailing segments here cannot dangle a concurrent search.
            graph_.unsafe_resize(max_index);
            data_.resize(max_index);
            first_empty_->store(max_index, std::memory_order_release);
            first_reusable_->store(max_index, std::memory_order_release);

            // Compact metadata and ID remapping.
            for (size_t new_id = 0; new_id < max_index; ++new_id) {
                auto old_id = getindex(new_to_old_id_map, new_id);
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

            // Update entry points. If an entry point is no longer present
            // (e.g. it was Deleted prior to compact), fall back to internal
            // ID 0 — by construction max_index > 0 implies a survivor.
            for (auto& ep : entry_point_) {
                auto it = old_to_new_id_map.find(ep);
                if (it != old_to_new_id_map.end()) {
                    ep = it->second;
                } else {
                    assert(max_index > 0);
                    ep = 0;
                }
            }
        }

        // Re-derive the reverse-edge index from the fully remapped graph.
        if (auto* re = graph_.reverse_edges()) {
            re->set_recording(true);
        }
        graph_.rebuild_reverse_edges(threadpool_);
    }

    ///// Threading Interface

    /// @brief Return the current number of threads used for search.
    ///
    /// @sa set_num_threads
    size_t get_num_threads() const { return threadpool_.size(); }

    void set_threadpool(threads::ThreadPoolHandle threadpool) {
        threadpool_ = std::move(threadpool);
    }

    ///
    /// @brief Destroy the original thread pool and set to the provided one.
    ///
    /// @param threadpool An acceptable thread pool.
    ///
    /// @copydoc threadpool_requirements
    ///
    template <threads::ThreadPool Pool>
    void set_threadpool(Pool threadpool)
        requires(!std::is_same_v<Pool, threads::ThreadPoolHandle>)
    {
        set_threadpool(threads::ThreadPoolHandle(std::move(threadpool)));
    }

    ///
    /// @brief Return the current thread pool handle.
    ///
    threads::ThreadPoolHandle& get_threadpool_handle() { return threadpool_; }

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
        std::shared_lock compact_lock{*compact_mutex_};
        consolidate_locked();
    }

    // Full consolidation: gather the Deleted set (one cheap byte scan of status_) then
    // delegate. Concurrent deletes arriving after the scan are picked up next round.
    void consolidate_locked() {
        tsl::robin_set<Idx> deleted{};
        for (size_t i = 0, imax = status_.size(); i < imax; ++i) {
            if (std::atomic_ref<SlotMetadata>(status_[i]).load(std::memory_order_acquire) ==
                SlotMetadata::Deleted) {
                deleted.insert(lib::narrow_cast<Idx>(i));
            }
        }
        consolidate_locked(deleted);
    }

    // Body of consolidate()/consolidate(ids) with no compact_mutex_ locking. The caller
    // holds compact_mutex_. `deleted` holds the internal slots to prune out of the graph
    // and reclaim; those slots must currently be SlotMetadata::Deleted. Consolidation and
    // cleanup both range over exactly `deleted`, so both are O(|deleted|), not O(N).
    void consolidate_locked(const tsl::robin_set<Idx>& deleted) {
        auto should_remove = [&](size_t i) {
            return deleted.contains(lib::narrow_cast<Idx>(i));
        };

        // Entry-point candidacy: a replacement must be live (not soft-deleted) and
        // not itself about to be removed.
        std::function<bool(size_t)> valid = [&](size_t i) {
            return !should_remove(i) && !this->is_deleted(i);
        };

        // Determine if the entry point is being removed.
        // If so - we need to pick a new one.
        assert(entry_point_.size() == 1);
        auto entry_point = entry_point_[0];
        if (should_remove(entry_point)) {
            svs::logging::debug(logger_, "Replacing entry point.");
            auto new_entry_point =
                extensions::compute_entry_point(data_, threadpool_, valid);
            svs::logging::debug(logger_, "New point: {}", new_entry_point);
            assert(valid(new_entry_point));
            entry_point_[0] = new_entry_point;
        }

        // Perform graph consolidation over the in-neighbors of `deleted`, discovered via
        // the reverse-edge index.
        concurrent::consolidate(
            graph_,
            data_,
            threadpool_,
            prune_to_,
            max_candidates_,
            alpha_,
            distance_,
            deleted,
            should_remove
        );

        // After consolidation - clean up the removed slots under lock. O(|deleted|).
        {
            std::lock_guard lock{*translator_mutex_};
            // Erase translator entries for removed slots (deferred from delete_entries).
            // Skip entries already absent — add_points with replace_stale_and_insert
            // may have reassigned the external ID and erased the stale reverse entry.
            std::vector<size_t> deleted_internal_ids;
            for (auto i : deleted) {
                if (status_[i] == SlotMetadata::Deleted && translator_.has_internal(i)) {
                    deleted_internal_ids.push_back(i);
                }
            }
            if (!deleted_internal_ids.empty()) {
                translator_.delete_internal(deleted_internal_ids, false);
            }
            // Set removed `Deleted` slots to `Empty`
            size_t min_freed = std::numeric_limits<size_t>::max();
            for (auto i : deleted) {
                if (status_[i] == SlotMetadata::Deleted) {
                    std::atomic_ref<SlotMetadata>(status_[i])
                        .store(SlotMetadata::Empty, std::memory_order_release);
                    min_freed = std::min(min_freed, static_cast<size_t>(i));
                }
            }
            if (min_freed != std::numeric_limits<size_t>::max()) {
                detail::atomic_min(*first_reusable_, min_freed);
            }
        }
    }

    ///
    /// @brief Consolidate only the soft-deleted entries listed in `ids`.
    ///
    /// * Listed IDs to be removed must have been previously soft-deleted via
    ///   `delete_entries`. IDs that are not currently soft-deleted (never existed,
    ///   already consolidated, or still Valid/Pending) are skipped.
    ///
    /// * For each consolidated ID, no live node retains an edge to it, its
    ///   translator entry is erased, and its slot is set to `Empty`.
    ///
    /// @returns The number of listed IDs that were consolidated.
    ///
    template <typename T> size_t consolidate(const T& ids) {
        std::shared_lock compact_lock{*compact_mutex_};

        // Collect the internal slots of the listed, already-soft-deleted IDs.
        tsl::robin_set<Idx> targets{};
        {
            std::shared_lock lock{*translator_mutex_};
            for (auto i : ids) {
                if (!translator_.has_external(i)) {
                    continue; // Already consolidated, or never existed.
                }
                auto internal = translator_.get_internal(i);
                if (!is_deleted(internal)) {
                    continue; // Not soft-deleted — nothing to consolidate.
                }
                targets.insert(lib::narrow_cast<Idx>(internal));
            }
        }

        if (targets.empty()) {
            return 0;
        }
        consolidate_locked(targets);
        return targets.size();
    }

    ///// Saving

    VamanaIndexParameters parameters() const {
        return {
            entry_point_.front(),
            {alpha_,
             graph_.max_degree(),
             get_construction_window_size(),
             get_max_candidates(),
             prune_to_,
             get_full_search_history()},
            get_search_parameters()};
    }

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
                return lib::SaveTable(
                    "vamana_dynamic_auxiliary_parameters",
                    save_version,
                    {
                        {"name", lib::save(name())},
                        {"parameters", lib::save(parameters(), ctx)},
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

    void save(std::ostream& os) {
        // Post-consolidation, all entries should be "valid".
        // Therefore, we don't need to save the slot metadata.
        consolidate();
        compact();

        lib::begin_serialization(os);
        auto save_table = lib::SaveTable(
            "vamana_dynamic_auxiliary_parameters",
            save_version,
            {
                {"name", lib::save(name())},
                {"parameters", lib::save(parameters())},
                {"translation", lib::detail::exit_hook(translator_.metadata())},
            }
        );
        lib::save_to_stream(save_table, os);
        translator_.save(os);

        // Save the dataset.
        lib::save_to_stream(data_, os);
        // Save the graph.
        lib::save_to_stream(graph_, os);
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
            do_search,
            logger_
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

        // Lock order: compact_mutex_ then translator_mutex_ (global order).
        // compact_mutex_ shared guards data_/graph_ against compact()'s shrink;
        // translator_mutex_ shared guards the ID translation reads below.
        std::shared_lock compact_lock{*compact_mutex_};
        std::shared_lock lock{*translator_mutex_};

        // Bounds checking.
        for (size_t i = 0; i < ids_size; ++i) {
            I id = ids[i]; // inbounds by loop bounds.
            if (!unsafe_has_id(id)) {
                throw ANNEXCEPTION("ID {} with value {} is out of bounds!", i, id);
            }
        }

        // Prerequisites checked - proceed with the operation.
        // TODO: Communicate the requested decompression type to the backend dataset to
        // allow more fine-grained specialization?
        auto threaded_function = [&](auto is, uint64_t SVS_UNUSED(tid)) {
            auto accessor = extensions::reconstruct_accessor(data_);
            for (auto i : is) {
                auto id = unsafe_translate_external_id(ids[i]);
                dst.set_datum(i, accessor(data_, id));
            }
        };
        threads::parallel_for(
            threadpool_, threads::StaticPartition{ids_size}, threaded_function
        );
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
                case SlotMetadata::Pending: {
                    // In-flight add: edges may be only partially published.
                    // Treat as not-yet-live for consistency checking.
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
                        concurrent::name(metadata),
                        j,
                        count
                    );
                }
                count++;
            }
        }
    }

    ///// Distance

    /// @brief Compute the distance between an external vector and a vector in the index.
    template <typename ExternalId, typename Query>
    double get_distance(const ExternalId& external_id, const Query& query) const {
        // Lock order: compact_mutex_ then translator_mutex_ (global order).
        // compact_mutex_ shared guards data_ against compact()'s shrink;
        // translator_mutex_ shared guards the ID translation read.
        std::shared_lock compact_lock{*compact_mutex_};
        std::shared_lock lock{*translator_mutex_};

        // Check if the external ID exists
        if (!unsafe_has_id(external_id)) {
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
        auto internal_id = unsafe_translate_external_id(external_id);

        // Call extension for distance computation
        return extensions::get_distance_ext(data_, distance_, internal_id, query);
    }

    ///
    /// @brief Top up an under-filled search buffer with a linear scan.
    ///
    /// When many vectors have been deleted the graph can become sparsely connected and
    /// the greedy search may return fewer than ``target_window()`` valid neighbors. This
    /// supplements the buffer by scanning the ID translation.
    ///
    /// **Precondition:** the caller holds ``compact_mutex_`` shared (``search()`` does).
    /// This is why the generic ``extensions::check_and_supplement_search_buffer`` is not
    /// used here: it reaches back through ``index.get_distance()``, which takes
    /// ``compact_mutex_`` shared a *second* time on the same thread. Recursive shared
    /// acquisition of a ``std::shared_mutex`` is undefined behavior and deadlocks against
    /// a queued writer. It also iterates a snapshot from ``external_ids()``, which reads
    /// ``translator_`` unlocked. Both are avoided by doing the scan here, under
    /// ``translator_mutex_`` shared (preserving the global compact -> translator order)
    /// and skipping slots that are no longer ``Valid``.
    ///
    /// Called from the ``extensions::single_search`` override below.
    ///
    template <typename SearchBuffer, typename Query>
    void supplement_search_buffer(SearchBuffer& search_buffer, const Query& query) const {
        if (search_buffer.valid() >= search_buffer.target_window() ||
            search_buffer.valid() >= size()) {
            return;
        }
        search_buffer.sort();

        // translator_mutex_ shared: guards both the traversal of ``translator_`` and the
        // internal IDs it yields against erasure by consolidate()/compact().
        std::shared_lock translator_lock{*translator_mutex_};
        auto builder = internal_search_builder();
        for (auto pair : translator_) {
            auto internal_id = pair.second;
            // Skip slots pending deferred translator cleanup, reserved by an in-flight
            // add_points, or otherwise not live.
            if (std::atomic_ref<SlotMetadata>(const_cast<SlotMetadata&>(status_[internal_id]
                ))
                    .load(std::memory_order_acquire) != SlotMetadata::Valid) {
                continue;
            }
            auto dist = extensions::get_distance_ext(data_, distance_, internal_id, query);
            search_buffer.insert(builder(internal_id, dist));
            if (search_buffer.valid() >= search_buffer.target_window()) {
                break;
            }
        }
    }

    template <typename QueryType>
    auto make_batch_iterator(
        std::span<const QueryType> query,
        size_t extra_search_buffer_capacity = svs::UNSIGNED_INTEGER_PLACEHOLDER
    ) const {
        return BatchIterator(*this, query, extra_search_buffer_capacity);
    }
};

///
/// @brief Hides an index's id translation from the search extensions.
///
/// Every ``extensions::single_search`` implementation tops the search buffer up itself, by
/// calling the generic ``extensions::check_and_supplement_search_buffer`` under
/// ``if constexpr (Index::needs_id_translation)``. That routine is unsafe for this index --
/// see ``MutableVamanaIndex::supplement_search_buffer`` -- and it is a plain function, not a
/// customization point, so it cannot be overridden. Passing this wrapper in place of the
/// index makes that branch disappear, leaving the top-up to the index itself.
///
/// The implementations touch nothing else on the index, so nothing else needs forwarding.
///
template <typename Index> struct SupplementSuppressed {
    static constexpr bool needs_id_translation = false;
};

///
/// @brief ``extensions::single_search`` for the concurrent mutable index.
///
/// Runs the dataset's own search implementation and then tops the buffer up through the
/// index (see ``MutableVamanaIndex::supplement_search_buffer``). More specialized than the
/// generic overload on the index parameter, so it wins partial ordering for this index type.
///
/// The dispatch back through ``extensions::single_search`` is what makes compressed datasets
/// work: LVQ and LeanVec each provide their own implementation, keyed on the *dataset*, which
/// unpacks the scratch space (a tuple of distance functors, for LeanVec) and reranks. Doing
/// the search inline here instead would shadow those with a plain uncompressed search, and
/// then fail to compile because the tuple is not a distance functor. Recursion is not a
/// concern: the wrapper is not a ``MutableVamanaIndex``, so this overload does not match the
/// inner call.
///
template <
    typename Data,
    typename SearchBuffer,
    typename Scratch,
    typename Query,
    typename Search,
    typename Graph,
    typename IndexData,
    typename Dist>
SVS_FORCE_INLINE void svs_invoke(
    svs::tag_t<vamana::extensions::single_search>,
    const Data& dataset,
    SearchBuffer& search_buffer,
    Scratch& scratch,
    const Query& query,
    const Search& search,
    const MutableVamanaIndex<Graph, IndexData, Dist>& index,
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    using index_type = MutableVamanaIndex<Graph, IndexData, Dist>;
    vamana::extensions::single_search(
        dataset,
        search_buffer,
        scratch,
        query,
        search,
        SupplementSuppressed<index_type>{},
        cancel
    );

    // Check if request to cancel the search
    if (cancel()) {
        return;
    }
    index.supplement_search_buffer(search_buffer, query);
}

///// Deduction Guides.
// Guide for building.
template <typename Data, typename Dist, typename ExternalIds>
MutableVamanaIndex(const VamanaBuildParameters&, Data, const ExternalIds&, Dist, size_t)
    -> MutableVamanaIndex<graphs::SimpleBlockedGraph<uint32_t>, Data, Dist>;

template <typename Data, typename Dist, typename ExternalIds, threads::ThreadPool Pool>
MutableVamanaIndex(const VamanaBuildParameters&, Data, const ExternalIds&, Dist, Pool)
    -> MutableVamanaIndex<graphs::SimpleBlockedGraph<uint32_t>, Data, Dist>;

// Guide with logging
template <typename Data, typename Dist, typename ExternalIds, threads::ThreadPool Pool>
MutableVamanaIndex(
    const VamanaBuildParameters&,
    Data,
    const ExternalIds&,
    Dist,
    Pool,
    svs::logging::logger_ptr
) -> MutableVamanaIndex<graphs::SimpleBlockedGraph<uint32_t>, Data, Dist>;

template <typename Data, typename Dist, typename ExternalIds>
MutableVamanaIndex(
    const VamanaBuildParameters&,
    Data,
    const ExternalIds&,
    Dist,
    size_t,
    svs::logging::logger_ptr
) -> MutableVamanaIndex<graphs::SimpleBlockedGraph<uint32_t>, Data, Dist>;
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
template <
    typename GraphLoader,
    typename DataLoader,
    typename Distance,
    typename ThreadPoolProto>
auto auto_dynamic_assemble(
    const std::filesystem::path& config_path,
    GraphLoader&& graph_loader,
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
    //     // If loading from the static index, then the table we receive is itself the
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
        std::move(threadpool),
        std::move(logger)};
}

template <
    typename LazyGraphLoader,
    typename LazyDataLoader,
    typename Distance,
    typename ThreadPoolProto>
auto auto_dynamic_assemble(
    std::istream& is,
    LazyGraphLoader graph_loader,
    LazyDataLoader data_loader,
    Distance distance,
    ThreadPoolProto threadpool_proto,
    bool SVS_UNUSED(debug_load_from_static) = false,
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    // Read the combined TOML (parameters + translation)
    // and the translator binary data.
    auto table = lib::detail::read_metadata(is);

    auto parameters = lib::load<VamanaIndexParameters>(
        table.template cast<toml::table>().at("parameters").template cast<toml::table>()
    );

    auto translation =
        table.template cast<toml::table>().at("translation").template cast<toml::table>();

    auto translator = IDTranslator::load(translation, is);

    auto data = data_loader();
    auto graph = graph_loader();

    auto datasize = data.size();
    auto graphsize = graph.n_nodes();
    if (datasize != graphsize) {
        throw ANNEXCEPTION(
            "Reloaded data has {} nodes while the graph has {} nodes!", datasize, graphsize
        );
    }

    auto translator_size = translator.size();
    if (translator_size != datasize) {
        throw ANNEXCEPTION(
            "Translator has {} IDs but should have {}", translator_size, datasize
        );
    }

    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    return MutableVamanaIndex{
        parameters,
        std::move(data),
        std::move(graph),
        std::move(distance),
        std::move(translator),
        std::move(threadpool),
        std::move(logger)};
}

} // namespace svs::index::vamana::concurrent
