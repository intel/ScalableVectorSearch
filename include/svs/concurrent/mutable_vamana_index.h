/*
 * Copyright 2026 Intel Corporation
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

#include "svs/concurrent/graph.h"
#include "svs/concurrent/graph_view.h"
#include "svs/concurrent/greedy_search.h"
#include "svs/lib/concurrency/writer_priority_mutex.h"

#include "svs/core/translation.h"
#include "svs/index/vamana/consolidate.h"
#include "svs/index/vamana/dynamic_index.h" // SlotMetadata
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/index.h"
#include "svs/index/vamana/iterator.h"
#include "svs/index/vamana/search_params.h"
#include "svs/index/vamana/vamana_build.h"
#include "svs/lib/concurrency/readwrite_protected.h"
#include "svs/lib/threads.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace svs::concurrent {

using svs::index::vamana::SlotMetadata;

///
/// @brief Neighbor builder that filters deleted slots using atomic status reads.
///
/// Equivalent to ``svs::index::vamana::ValidBuilder`` except that the status byte is read
/// with an atomic load, because a concurrent writer may be flipping it from ``Valid`` to
/// ``Deleted`` while a search is in flight.
///
class AtomicValidBuilder {
  public:
    explicit AtomicValidBuilder(const uint8_t* status)
        : status_{status} {}

    template <typename I>
    svs::PredicatedSearchNeighbor<I> operator()(I i, float distance) const {
        auto raw = std::atomic_ref<uint8_t>(const_cast<uint8_t&>(status_[i]))
                       .load(std::memory_order_relaxed);
        bool invalid = static_cast<SlotMetadata>(raw) == SlotMetadata::Deleted;
        return svs::PredicatedSearchNeighbor<I>(i, distance, !invalid);
    }

  private:
    const uint8_t* status_;
};

///
/// @brief A dynamic Vamana index whose searches do not block on insertions or deletions.
///
/// This is a *new index type*, not a modification of
/// ``svs::index::vamana::MutableVamanaIndex``. It is **purely additive**: it lives in its
/// own namespace and modifies no existing header. ``VamanaBuilder``, ``prune``,
/// ``consolidate``, the search-extension hooks, the search buffers and the graph concepts
/// are all consumed unmodified. The static Vamana index and the existing mutable index are
/// therefore completely unaffected by this feature -- they pay no extra atomic load, no
/// extra indirection and no extra byte per node.
///
/// ## Synchronization model
///
/// | Operation | Search blocked? | Mechanism |
/// |---|---|---|
/// | ``search`` | -- | shared lock on ``structure_mutex_`` + per-node seqlock reads |
/// | ``add_points`` (no growth) | no | ``writer_mutex_`` only |
/// | ``add_points`` (growth) | briefly | exclusive ``structure_mutex_`` around the resize |
/// | ``delete_entries`` | no | ``writer_mutex_`` + atomic status store |
/// | ``consolidate`` / ``compact`` | yes | exclusive ``structure_mutex_`` |
///
/// Three locks, with a strict acquisition order of ``writer_mutex_`` ->
/// ``structure_mutex_`` -> ``translator_mutex_``:
///
/// * ``writer_mutex_`` serializes whole mutating *operations* against each other. Note
///   this is coarser than "one writer thread": ``VamanaBuilder`` still mutates the graph
///   with the full thread pool from inside ``add_points``. That is safe because the
///   builder already guarantees *at most one writer per node* (it holds per-vertex
///   ``SpinLock``s while adding reverse edges, and partitions nodes disjointly across
///   threads elsewhere), which is exactly the precondition ``SeqLockCounter`` requires.
/// * ``structure_mutex_`` is held *shared* for the duration of a search and *exclusive*
///   only when the container structure changes under readers' feet. The graph is
///   grow-stable so it needs no such protection, but ``svs::data::BlockedData`` holds its
///   block descriptors in a ``std::vector`` that reallocates on growth, so capacity
///   growth and compaction must exclude readers. Steady-state edge rewiring -- the bulk
///   of insertion work -- does not.
/// * ``translator_mutex_`` guards the external<->internal ID maps, whose hash tables
///   rehash on insert. Searches take it shared only to convert their final k results.
///
/// The alternative to the exclusive-on-growth window is to make the dataset itself
/// grow-stable, which means changing ``svs::data::SimpleData<..., Blocked<...>>`` and so
/// adding an indirection to ``get_datum`` for *every* index type in the library. This
/// prototype deliberately keeps that cost out of the shared data path.
///
template <std::unsigned_integral Index, typename Data, typename Dist>
class MutableVamanaIndex {
  public:
    // Traits
    static constexpr bool supports_insertions = true;
    static constexpr bool supports_deletions = true;
    static constexpr bool supports_saving = false; // Not implemented in the prototype.
    static constexpr bool needs_id_translation = true;

    /// @brief Placeholder written into a result slot whose vector was deleted mid-search.
    ///
    /// Concurrent deletion is legal here, so a search can legitimately select a neighbor
    /// that is retired before its ID is translated back. Results carrying this value must
    /// be dropped by the caller. ``svs::index::vamana::MutableVamanaIndex`` has no analogue
    /// because it forbids concurrent writers outright.
    static constexpr size_t invalid_external_id = std::numeric_limits<size_t>::max();

    // Type aliases -- deliberately mirroring svs::index::vamana::MutableVamanaIndex so that
    // call sites can swap between the two.
    using Idx = Index;
    using internal_id_type = Idx;
    using external_id_type = size_t;
    using value_type = typename Data::value_type;
    using const_value_type = typename Data::const_value_type;
    static constexpr size_t extent = Data::extent;

    using distance_type = Dist;
    using search_buffer_type =
        svs::index::vamana::MutableBuffer<Idx, svs::distance::compare_t<Dist>>;

    using graph_type = SeqLockGraph<Idx>;
    using data_type = Data;
    using entry_point_type = std::vector<Idx>;
    using search_parameters_type = svs::index::vamana::VamanaSearchParameters;
    using inner_scratch_type = svs::tag_t<
        svs::index::vamana::extensions::single_search_setup>::result_t<Data, Dist>;
    using scratchspace_type =
        svs::index::vamana::SearchScratchspace<search_buffer_type, inner_scratch_type>;

    ///
    /// @brief Construct from a pre-built graph and dataset.
    ///
    /// Signature matches ``svs::index::vamana::MutableVamanaIndex``'s corresponding
    /// constructor so that integrators (e.g. VecSim) can select between the two with a type
    /// alias.
    ///
    template <typename ExternalIds, typename ThreadPoolProto>
    MutableVamanaIndex(
        graph_type graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        const ExternalIds& external_ids,
        ThreadPoolProto threadpool_proto,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{entry_point}
        , status_(data_.size(), static_cast<uint8_t>(SlotMetadata::Valid))
        , first_empty_{data_.size()}
        , translator_{}
        , distance_{std::move(distance_function)}
        , threadpool_{svs::threads::as_threadpool(std::move(threadpool_proto))}
        , search_parameters_{svs::index::vamana::construct_default_search_parameters(data_)}
        , construction_window_size_{2 * graph_.max_degree()}
        , max_candidates_{750}
        , prune_to_{graph_.max_degree()}
        , logger_{std::move(logger)} {
        translator_.insert(external_ids, svs::threads::UnitRange<Idx>(0, data_.size()));
    }

    ///// Accessors

    svs::logging::logger_ptr get_logger() const { return logger_; }
    size_t dimensions() const { return data_.dimensions(); }
    const Data& view_data() const { return data_; }
    const graph_type& view_graph() const { return graph_; }
    size_t max_degree() const { return graph_.max_degree(); }

    float get_alpha() const { return alpha_; }
    void set_alpha(float alpha) { alpha_ = alpha; }
    size_t get_construction_window_size() const { return construction_window_size_; }
    void set_construction_window_size(size_t s) { construction_window_size_ = s; }
    size_t get_max_candidates() const { return max_candidates_; }
    void set_max_candidates(size_t n) { max_candidates_ = n; }
    size_t get_prune_to() const { return prune_to_; }
    void set_prune_to(size_t n) { prune_to_ = n; }
    bool get_full_search_history() const { return use_full_search_history_; }
    void set_full_search_history(bool b) { use_full_search_history_ = b; }

    search_parameters_type get_search_parameters() const {
        return search_parameters_.get();
    }
    void set_search_parameters(const search_parameters_type& sp) {
        search_parameters_.set(sp);
    }

    void reset_performance_parameters() {
        auto sp = get_search_parameters();
        auto pp = svs::index::vamana::extensions::estimate_prefetch_parameters(data_);
        sp.prefetch_lookahead_ = pp.lookahead;
        sp.prefetch_step_ = pp.step;
        set_search_parameters(sp);
    }

    size_t get_num_threads() const { return threadpool_.size(); }
    void set_threadpool(svs::threads::ThreadPoolHandle pool) {
        threadpool_ = std::move(pool);
    }

    Dist distance_function() const { return svs::threads::shallow_copy(distance_); }

    ///// ID translation
    //
    // All of these take the translator lock shared: a concurrent ``add_points`` may be
    // rehashing the underlying maps.

    Idx translate_external_id(size_t e) const {
        std::shared_lock lock{translator_mutex_};
        return translator_.get_internal(e);
    }
    size_t translate_internal_id(Idx i) const {
        std::shared_lock lock{translator_mutex_};
        return translator_.get_external(i);
    }
    /// @brief As ``translate_internal_id``, but yields ``invalid_external_id`` instead of
    /// throwing if the slot was retired by a concurrent ``delete_entries``.
    size_t try_translate_internal_id(Idx i) const {
        std::shared_lock lock{translator_mutex_};
        return translator_.has_internal(i) ? translator_.get_external(i)
                                           : invalid_external_id;
    }
    bool has_id(size_t e) const {
        std::shared_lock lock{translator_mutex_};
        return translator_.has_external(e);
    }
    /// @brief Number of valid (non-deleted) entries.
    size_t size() const {
        std::shared_lock lock{translator_mutex_};
        return translator_.size();
    }

    template <typename F> void on_ids(F&& f) const {
        std::shared_lock lock{translator_mutex_};
        for (auto pair : translator_) {
            f(pair.first);
        }
    }

    std::vector<size_t> external_ids() const {
        std::vector<size_t> ids{};
        on_ids([&ids](size_t id) { ids.push_back(id); });
        return ids;
    }

    auto get_datum(size_t e) const { return data_.get_datum(translate_external_id(e)); }

    bool is_deleted(size_t i) const { return load_status(i) != SlotMetadata::Valid; }

    /// @brief Bytes reserved by the graph, data and dynamic metadata.
    size_t get_memory_usage() const {
        return graph_.bytes_reserved() +
               data_.size() * data_.dimensions() * sizeof(value_type) +
               status_.capacity() * sizeof(uint8_t);
    }

    ///// Search

    AtomicValidBuilder internal_search_builder() const {
        return AtomicValidBuilder{status_.data()};
    }

    scratchspace_type scratchspace(const search_parameters_type& sp) const {
        return scratchspace_type{
            search_buffer_type{sp.buffer_config_, svs::distance::comparator(distance_)},
            svs::index::vamana::extensions::single_search_setup(data_, distance_),
            svs::index::vamana::GreedySearchPrefetchParameters{
                sp.prefetch_lookahead_, sp.prefetch_step_}};
    }
    scratchspace_type scratchspace() const { return scratchspace(get_search_parameters()); }

    ///
    /// @brief The search closure handed to the extension hooks.
    ///
    /// Identical to ``svs::index::vamana::MutableVamanaIndex::greedy_search_closure`` apart
    /// from calling
    /// ``svs::concurrent::seqlock_greedy_search`` instead of
    /// ``svs::index::vamana::greedy_search``.
    ///
    auto greedy_search_closure(
        svs::index::vamana::GreedySearchPrefetchParameters prefetch_parameters,
        const svs::lib::DefaultPredicate& cancel =
            svs::lib::Returns(svs::lib::Const<false>())
    ) const {
        return [&, prefetch_parameters](
                   const auto& query, auto& accessor, auto& distance, auto& buffer
               ) {
            seqlock_greedy_search(
                graph_,
                data_,
                accessor,
                query,
                distance,
                buffer,
                EntryPointInitializer<Idx>{svs::lib::as_const_span(entry_point_)},
                internal_search_builder(),
                prefetch_parameters,
                cancel
            );
            buffer.cleanup();
        };
    }

    /// @brief Single-query search into a caller-provided scratch space.
    template <typename Query>
    void search(
        const Query& query,
        scratchspace_type& scratch,
        const svs::lib::DefaultPredicate& cancel =
            svs::lib::Returns(svs::lib::Const<false>())
    ) const {
        // Shared: excludes capacity growth and GC, admits concurrent searches and
        // concurrent steady-state insertions/deletions.
        std::shared_lock structure_lock{structure_mutex_};
        svs::index::vamana::extensions::single_search(
            data_,
            scratch.buffer,
            scratch.scratch,
            query,
            greedy_search_closure(scratch.prefetch_parameters, cancel),
            *this
        );
    }

    /// @brief Batch search over ``queries``, writing external IDs into ``results``.
    template <typename I, svs::data::ImmutableMemoryDataset Queries>
    void search(
        svs::QueryResultView<I> results,
        const Queries& queries,
        const search_parameters_type& sp,
        const svs::lib::DefaultPredicate& cancel =
            svs::lib::Returns(svs::lib::Const<false>())
    ) {
        {
            std::shared_lock structure_lock{structure_mutex_};
            svs::threads::parallel_for(
                threadpool_,
                svs::threads::StaticPartition{queries.size()},
                [&](const auto is, uint64_t /*tid*/) {
                    size_t num_neighbors = results.n_neighbors();
                    auto buffer = search_buffer_type{
                        sp.buffer_config_, svs::distance::comparator(distance_)};
                    auto prefetch_parameters =
                        svs::index::vamana::GreedySearchPrefetchParameters{
                            sp.prefetch_lookahead_, sp.prefetch_step_};
                    if (buffer.target_capacity() < num_neighbors) {
                        buffer.change_maxsize(num_neighbors);
                    }
                    auto scratch =
                        svs::index::vamana::extensions::per_thread_batch_search_setup(
                            data_, distance_
                        );
                    svs::index::vamana::extensions::per_thread_batch_search(
                        data_,
                        buffer,
                        scratch,
                        queries,
                        results,
                        svs::threads::UnitRange{is},
                        greedy_search_closure(prefetch_parameters, cancel),
                        *this,
                        cancel
                    );
                }
            );
        }

        if (cancel()) {
            return;
        }
        translate_to_external(results.indices());
    }

    /// @brief Distance between the vector stored for ``external_id`` and ``query``.
    template <typename ExternalId, typename Query>
    double get_distance(const ExternalId& external_id, const Query& query) const {
        if (!has_id(external_id)) {
            throw ANNEXCEPTION(
                "ID {} is out of bounds for index of size {}!", external_id, size()
            );
        }
        if (query.size() != dimensions()) {
            throw ANNEXCEPTION(
                "Incompatible dimensions. Query has {} while the index expects {}.",
                query.size(),
                dimensions()
            );
        }
        std::shared_lock structure_lock{structure_mutex_};
        auto internal_id = translate_external_id(external_id);
        return svs::index::vamana::extensions::get_distance_ext(
            data_, distance_, internal_id, query
        );
    }

    ///// Mutation

    ///
    /// @brief Insert ``points`` under the given external IDs.
    ///
    /// Searches run concurrently throughout, except for a brief exclusive window if the
    /// dataset must grow to make room.
    ///
    template <svs::data::ImmutableMemoryDataset Points, class ExternalIds>
    std::vector<size_t> add_points(
        const Points& points, const ExternalIds& external_ids, bool reuse_empty = false
    ) {
        const size_t num_points = points.size();
        if (num_points != external_ids.size()) {
            throw ANNEXCEPTION(
                "Number of points ({}) not equal to the number of external ids ({})!",
                num_points,
                external_ids.size()
            );
        }

        std::lock_guard writer_lock{writer_mutex_};

        // Gather reusable slots. Only this thread mutates ``status_``'s structure, so a
        // plain scan is fine.
        std::vector<size_t> slots{};
        slots.reserve(num_points);
        for (size_t s = reuse_empty ? 0 : first_empty_, smax = status_.size();
             s < smax && slots.size() < num_points;
             ++s) {
            if (load_status(s) == SlotMetadata::Empty) {
                slots.push_back(s);
            }
        }

        if (slots.size() < num_points) {
            const size_t needed = num_points - slots.size();
            const size_t current_size = data_.size();
            const size_t new_size = current_size + needed;

            // The only place searches are excluded during insertion: ``BlockedData``'s
            // block-descriptor vector may reallocate. The graph and the status array are
            // resized here too so the window covers all of them at once.
            {
                std::unique_lock structure_lock{structure_mutex_};
                data_.resize(new_size);
                graph_.unsafe_resize(new_size);
                status_.resize(new_size, static_cast<uint8_t>(SlotMetadata::Empty));
            }

            for (size_t s = current_size; s < new_size; ++s) {
                slots.push_back(s);
            }
        }
        assert(slots.size() == num_points);

        // Publish the ID mapping before any edges point at the new slots, so a searcher
        // that reaches a new node can always translate it back to an external ID.
        {
            std::unique_lock translator_lock{translator_mutex_};
            translator_.insert(external_ids, slots);
        }

        // Write the vectors before wiring the nodes in: no in-edges exist yet, so the
        // data is fully visible by the time a searcher can reach these slots.
        svs::threads::parallel_for(
            threadpool_,
            svs::threads::StaticPartition{slots.size()},
            [&](auto is, uint64_t /*tid*/) {
                for (auto i : is) {
                    data_.set_datum(slots[i], points.get_datum(i));
                }
            }
        );
        for (auto slot : slots) {
            graph_.clear_node(static_cast<Idx>(slot));
        }

        // Wire up the new nodes using the *unmodified* upstream builder. It mutates the
        // graph with the whole thread pool, but never two threads on one node, which is
        // all the per-node seqlocks require.
        auto parameters = svs::index::vamana::VamanaBuildParameters{
            alpha_,
            graph_.max_degree(),
            construction_window_size_,
            max_candidates_,
            prune_to_,
            use_full_search_history_};

        auto sp = get_search_parameters();
        auto prefetch_parameters = svs::index::vamana::GreedySearchPrefetchParameters{
            sp.prefetch_lookahead_, sp.prefetch_step_};

        auto builder = svs::index::vamana::VamanaBuilder{
            graph_,
            data_,
            distance_,
            parameters,
            threadpool_,
            prefetch_parameters,
            logger_,
            svs::logging::Level::Trace};
        builder.construct(
            alpha_, entry_point(), slots, svs::logging::Level::Trace, logger_
        );

        for (auto slot : slots) {
            store_status(slot, SlotMetadata::Valid);
        }
        if (!slots.empty()) {
            first_empty_ = std::max(first_empty_, slots.back() + 1);
        }
        return slots;
    }

    ///
    /// @brief Soft-delete the given external IDs.
    ///
    /// Never blocks searches: flipping a status byte is a single atomic store, and
    /// in-flight searches simply stop returning the affected slots.
    ///
    template <typename T> size_t delete_entries(const T& ids) {
        std::lock_guard writer_lock{writer_mutex_};
        {
            std::shared_lock translator_lock{translator_mutex_};
            translator_.check_external_exist(ids.begin(), ids.end());
        }
        for (auto i : ids) {
            Idx internal;
            {
                std::shared_lock translator_lock{translator_mutex_};
                internal = translator_.get_internal(i);
            }
            store_status(internal, SlotMetadata::Deleted);
        }
        {
            std::unique_lock translator_lock{translator_mutex_};
            translator_.delete_external(ids);
        }
        return ids.size();
    }

    ///
    /// @brief Remove deleted entries from the graph's adjacency lists.
    ///
    /// Stop-the-world: takes ``structure_mutex_`` exclusively. Reuses the *unmodified*
    /// upstream ``svs::index::vamana::consolidate``, which is generic over the graph type
    /// -- possible only because this prototype left the graph concepts alone.
    ///
    void consolidate() {
        std::lock_guard writer_lock{writer_mutex_};
        std::unique_lock structure_lock{structure_mutex_};

        auto check_is_deleted = [&](size_t i) { return this->is_deleted(i); };
        std::function<bool(size_t)> valid = [&](size_t i) { return !this->is_deleted(i); };

        // Replace the entry point if it was deleted.
        if (load_status(entry_point_[0]) == SlotMetadata::Deleted) {
            auto new_entry_point = svs::index::vamana::extensions::compute_entry_point(
                data_, threadpool_, valid
            );
            entry_point_[0] = new_entry_point;
        }

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

        for (size_t i = 0, imax = status_.size(); i < imax; ++i) {
            if (load_status(i) == SlotMetadata::Deleted) {
                store_status(i, SlotMetadata::Empty);
            }
        }
    }

    ///
    /// @brief Squeeze out empty slots so IDs are dense again.
    ///
    /// Stop-the-world. Builds a fresh graph rather than remapping in place -- simpler and
    /// safe, at the cost of transiently holding two graphs.
    ///
    void compact(size_t batch_size = 1'000'000) {
        std::lock_guard writer_lock{writer_mutex_};
        std::unique_lock structure_lock{structure_mutex_};

        // new_to_old[new_id] == old_id
        std::vector<Idx> new_to_old{};
        for (size_t i = 0, imax = status_.size(); i < imax; ++i) {
            if (load_status(i) != SlotMetadata::Empty) {
                new_to_old.push_back(static_cast<Idx>(i));
            }
        }
        const size_t new_size = new_to_old.size();
        if (new_size == status_.size()) {
            return; // Already dense.
        }

        auto old_to_new = tsl::robin_map<Idx, Idx>{};
        for (Idx new_id = 0; new_id < static_cast<Idx>(new_size); ++new_id) {
            old_to_new.insert({new_to_old[new_id], new_id});
        }

        // Remap adjacency into a fresh graph. Edges into now-empty slots are dropped;
        // consolidate() should have removed them already, but be defensive.
        auto compacted = graph_type{new_size, graph_.max_degree()};
        std::vector<Idx> buffer{};
        for (Idx new_id = 0; new_id < static_cast<Idx>(new_size); ++new_id) {
            buffer.clear();
            for (auto old_neighbor : graph_.get_node(new_to_old[new_id])) {
                auto found = old_to_new.find(old_neighbor);
                if (found != old_to_new.end()) {
                    buffer.push_back(found->second);
                }
            }
            compacted.replace_node(new_id, buffer);
        }
        graph_ = std::move(compacted);

        data_.compact(svs::lib::as_const_span(new_to_old), threadpool_, batch_size);
        data_.resize(new_size);

        // Remap metadata and the translator.
        {
            std::unique_lock translator_lock{translator_mutex_};
            std::vector<uint8_t> new_status(new_size);
            for (size_t new_id = 0; new_id < new_size; ++new_id) {
                auto old_id = new_to_old[new_id];
                new_status[new_id] = static_cast<uint8_t>(load_status(old_id));
                if (static_cast<SlotMetadata>(new_status[new_id]) == SlotMetadata::Valid &&
                    old_id != static_cast<Idx>(new_id)) {
                    translator_.remap_internal_id(old_id, static_cast<Idx>(new_id));
                }
            }
            status_ = std::move(new_status);
        }
        first_empty_ = new_size;

        for (auto& ep : entry_point_) {
            ep = old_to_new.at(ep);
        }
    }

    Idx entry_point() const { return entry_point_[0]; }

    ///
    /// @brief Hand the raw graph, data, distance and entry points to ``f``.
    ///
    /// Same contract as
    /// ``svs::index::vamana::MutableVamanaIndex::experimental_escape_hatch``, with one
    /// difference that matters: the graph passed to ``f`` is a ``SeqLockGraphView``, not
    /// the
    /// ``SeqLockGraph`` itself. ``f`` therefore gets an ordinary immutable graph whose
    /// adjacency lists are certified snapshots, and needs no awareness of concurrency --
    /// which is what lets upstream's ``BatchIterator`` traverse this index unmodified.
    ///
    /// The structure lock is held shared for the duration of the callback, so capacity
    /// growth and compaction cannot run underneath it.
    ///
    template <typename F> void experimental_escape_hatch(F&& f) const {
        std::shared_lock structure_lock{structure_mutex_};
        auto view = SeqLockGraphView<Idx>{graph_};
        std::invoke(
            SVS_FWD(f), view, data_, distance_, svs::lib::as_const_span(entry_point_)
        );
    }

    ///
    /// @brief Batch (range-query) iterator over ``query``.
    ///
    /// This is upstream's ``BatchIterator``, verbatim, made safe by ``SeqLockGraphView``
    /// rather than by editing it. Note what it does *not* promise: the iterator caches a
    /// search buffer across ``next()`` calls, so a vector inserted or deleted between two
    /// batches may be missed or repeated. That is a semantic question about what a
    /// long-lived cursor over a mutating index should mean, and it is not something a lock
    /// discipline can answer -- it needs a product decision. What is guaranteed is memory
    /// safety and per-batch internal consistency.
    ///
    template <typename QueryType>
    auto make_batch_iterator(
        std::span<const QueryType> query,
        size_t extra_search_buffer_capacity = svs::UNSIGNED_INTEGER_PLACEHOLDER
    ) const {
        return svs::index::vamana::BatchIterator(
            *this, query, extra_search_buffer_capacity
        );
    }

    static std::string name() { return "concurrent vamana index"; }

  private:
    SlotMetadata load_status(size_t i) const {
        return static_cast<SlotMetadata>(
            std::atomic_ref<uint8_t>(const_cast<uint8_t&>(status_[i]))
                .load(std::memory_order_relaxed)
        );
    }
    void store_status(size_t i, SlotMetadata m) {
        std::atomic_ref<uint8_t>(status_[i])
            .store(static_cast<uint8_t>(m), std::memory_order_relaxed);
    }

    // Signature mirrors ``svs::index::vamana::MutableVamanaIndex::translate_to_external``:
    // the argument is the
    // ``DenseArray`` view handed back by ``QueryResultView::indices()``, rewritten in
    // place.
    template <typename Dims, typename Base>
    void translate_to_external(svs::DenseArray<size_t, Dims, Base>& ids) {
        std::shared_lock translator_lock{translator_mutex_};
        svs::threads::parallel_for(
            threadpool_,
            svs::threads::StaticPartition{svs::getsize<0>(ids)},
            [&](auto is, uint64_t /*tid*/) {
                for (auto i : is) {
                    for (size_t j = 0, jmax = svs::getsize<1>(ids); j < jmax; ++j) {
                        auto internal = svs::lib::narrow_cast<Idx>(ids.at(i, j));
                        // A concurrent ``delete_entries`` can retire a slot after the
                        // search selected it but before we get here, dropping its
                        // translator entry. Upstream can simply call ``get_external`` and
                        // let it throw, because no writer can run during its search. Here
                        // the honest answer is "this neighbor was deleted mid-flight", so
                        // report the sentinel and let the caller drop it. Throwing would
                        // turn a benign, expected race into a failed query.
                        ids.at(i, j) = translator_.has_internal(internal)
                                           ? translator_.get_external(internal)
                                           : invalid_external_id;
                    }
                }
            }
        );
    }

    ///// Members

    graph_type graph_;
    data_type data_;
    entry_point_type entry_point_;
    // Flat, so the search hot path pays one indexed load. Resized only under an exclusive
    // ``structure_mutex_``; individual bytes are read/written atomically.
    std::vector<uint8_t> status_;
    size_t first_empty_ = 0;
    svs::IDTranslator translator_;

    distance_type distance_;
    svs::threads::ThreadPoolHandle threadpool_;
    svs::lib::ReadWriteProtected<search_parameters_type> search_parameters_;

    // Configuration
    size_t construction_window_size_;
    size_t max_candidates_;
    size_t prune_to_;
    float alpha_ = 1.2;
    bool use_full_search_history_ = true;

    svs::logging::logger_ptr logger_;

    ///// Synchronization. Acquire in this order.
    mutable std::mutex writer_mutex_;
    // Both must be writer-preferring, not ``std::shared_mutex``. Searches hold these
    // shared for their whole duration, so under continuous query load a reader-preferring
    // lock leaves the writer parked forever. See ``writer_priority_mutex.h``.
    mutable svs::WriterPriorityMutex structure_mutex_;
    mutable svs::WriterPriorityMutex translator_mutex_;
};

} // namespace svs::concurrent
