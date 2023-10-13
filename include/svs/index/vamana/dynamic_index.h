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
#include "svs/core/medioid.h"
#include "svs/core/query_result.h"
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

template <SlotMetadata Metadata> inline std::string name();
template <> inline std::string name<SlotMetadata::Empty>() { return "Empty"; }
template <> inline std::string name<SlotMetadata::Valid>() { return "Valid"; }
template <> inline std::string name<SlotMetadata::Deleted>() { return "Deleted"; }

// clang-format off
inline std::string name(SlotMetadata metadata) {
    #define SVS_SWITCH_RETURN(x) case x: { return name<x>(); }
    switch (metadata) {
        SVS_SWITCH_RETURN(SlotMetadata::Empty)
        SVS_SWITCH_RETURN(SlotMetadata::Valid)
        SVS_SWITCH_RETURN(SlotMetadata::Deleted)
    }
    #undef SVS_SWITCH_RETURN
    // make GCC happy
    return "unreachable";
}
// clang-format on

class SkipBuilder {
  public:
    SkipBuilder(const std::vector<SlotMetadata>& status)
        : status_{status} {}

    template <typename I>
    constexpr SkippableSearchNeighbor<I> operator()(I i, float distance) const {
        bool skipped = getindex(status_, i) == SlotMetadata::Deleted;
        // This neighbor should be skipped if the metadata corresponding to the given index
        // marks this slot as deleted.
        return SkippableSearchNeighbor<I>(i, distance, skipped);
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

    // Type Aliases
    using Idx = typename Graph::index_type;
    using value_type = typename Data::value_type;
    using const_value_type = typename Data::const_value_type;
    static constexpr size_t extent = Data::extent;

    using distance_type = Dist;
    using search_buffer_type = MutableBuffer<Idx, distance::compare_t<Dist>>;

    using graph_type = Graph;
    using data_type = Data;
    using entry_point_type = std::vector<Idx>;

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
    search_buffer_type search_buffer_prototype_;
    threads::NativeThreadPool threadpool_;

    // Configurations
    size_t construction_window_size_;
    size_t max_candidates_;
    size_t prune_to_;
    float alpha_ = 1.2;
    bool use_full_search_history_ = true;

    // Methods
  public:
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
        , search_buffer_prototype_{}
        , threadpool_{std::move(threadpool)}
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
        : graph_(Graph{parameters.graph_max_degree, data.size()})
        , data_(std::move(data))
        , entry_point_{}
        , status_(data_.size(), SlotMetadata::Valid)
        , translator_()
        , distance_(std::move(distance_function))
        , search_buffer_prototype_()
        , threadpool_(num_threads)
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
        auto builder = VamanaBuilder(graph_, data_, distance_, parameters, threadpool_);
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
        const VamanaConfigParameters& config,
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
        , search_buffer_prototype_{config.search_window_size}
        , threadpool_{std::move(threadpool)}
        , construction_window_size_{config.construction_window_size}
        , max_candidates_{config.max_candidates}
        , prune_to_{config.prune_to}
        , alpha_{config.alpha}
        , use_full_search_history_{config.use_full_search_history} {}

    ///// Accessors

    ///
    /// @brief Get the alpha value used for pruning while mutating the graph.
    ///
    /// @see set_alpha, get_construction_window_size, set_construction_window_size
    ///
    float get_alpha() const { return alpha_; }

    ///
    /// @brief Set the alpha value used for pruning while mutating the graph.
    ///
    /// @see get_alpha, get_construction_window_size, set_construction_window_size
    ///
    void set_alpha(float alpha) { alpha_ = alpha; }

    size_t get_max_candidates() const { return max_candidates_; }
    void set_max_candidates(size_t max_candidates) { max_candidates_ = max_candidates; }

    void set_full_search_history(bool enable) { use_full_search_history_ = enable; }
    bool get_full_search_history() const { return use_full_search_history_; }

    ///
    /// @brief Get the window size used the mutating the graph.
    ///
    /// @see set_construction_window_size, get_alpha, set_alpha
    ///
    size_t get_construction_window_size() const { return construction_window_size_; }

    ///
    /// @brief Set the window size used the mutating the graph.
    ///
    /// @see get_construction_window_size, get_alpha, set_alpha
    ///
    void set_construction_window_size(size_t window_size) {
        construction_window_size_ = window_size;
    }

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

    auto greedy_search_closure() const {
        return [&](const auto& query, const auto& accessor, auto& distance, auto& buffer) {
            // Perform the greedy search using the provided resources.
            greedy_search(
                graph_,
                data_,
                accessor,
                query,
                distance,
                buffer,
                entry_point_,
                SkipBuilder{status_}
            );
            // Take a pass over the search buffer to remove any deleted elements that
            // might remain.
            buffer.cleanup();
        };
    }

    ///
    /// @brief Return the ``num_neighbors`` approximate nearest neighbors to ``queries``.
    ///
    /// @param queries The queries to use.
    /// @param num_neighbors The number of neighbors to return per query.
    ///
    /// @return A query result with one row of IDs and distances for each query.
    ///
    template <data::ImmutableMemoryDataset Queries>
    QueryResult<size_t> search(const Queries& queries, size_t num_neighbors) {
        QueryResult<size_t> result{queries.size(), num_neighbors};
        search(queries.cview(), num_neighbors, result.view());
        return result;
    }

    template <typename QueryType, typename I>
    void search(
        data::ConstSimpleDataView<QueryType> queries,
        size_t num_neighbors,
        QueryResultView<I> result
    ) {
        SkipBuilder builder{status_};
        threads::run(
            threadpool_,
            threads::StaticPartition{queries.size()},
            [&](const auto is, uint64_t SVS_UNUSED(tid)) {
                auto buffer = threads::shallow_copy(search_buffer_prototype_);
                if (buffer.capacity() < num_neighbors) {
                    buffer.change_maxsize(num_neighbors);
                }
                auto scratch = extensions::per_thread_batch_search_setup(data_, distance_);

                extensions::per_thread_batch_search(
                    data_,
                    buffer,
                    scratch,
                    queries,
                    result,
                    threads::UnitRange{is},
                    greedy_search_closure()
                );
            }
        );

        // After the search procedure, the indices in `results` are internal.
        // Perform one more pass to convert these to external ids.
        translate_to_external(result.indices());
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

        VamanaBuilder builder{graph_, data_, distance_, parameters, threadpool_};
        builder.construct(alpha_, entry_point(), slots, false);
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
        // In the returned data structure, an entry `j` at index index `i` means that the
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
        data_.compact(new_to_old_id_map, threadpool_, batch_size);

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
    void set_search_window_size(size_t search_window_size) {
        search_buffer_prototype_.change_maxsize(search_window_size);
    }

    size_t get_search_window_size() const { return search_buffer_prototype_.target(); }

    void consolidate() {
        auto check_is_deleted = [&](size_t i) { return this->is_deleted(i); };
        auto valid = [&](size_t i) { return !(this->is_deleted(i)); };

        // Determine if the entry point is deleted.
        // If so - we need to pick a new one.
        assert(entry_point_.size() == 1);
        auto entry_point = entry_point_[0];
        if (status_.at(entry_point) == SlotMetadata::Deleted) {
            fmt::print("Replacing entry point! ... ");
            auto new_entry_point =
                extensions::compute_entry_point(data_, threadpool_, valid);
            fmt::print(" New point: {}\n", new_entry_point);
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

    ///// Visited Set Interface
    // TODO: Enable?
    void enable_visited_set() {}
    void disable_visited_set() {}
    bool visited_set_enabled() const { return false; }

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
                auto parameters = VamanaConfigParameters{
                    graph_.max_degree(),
                    entry_point_.front(),
                    alpha_,
                    get_max_candidates(),
                    get_construction_window_size(),
                    prune_to_,
                    get_full_search_history(),
                    get_search_window_size(),
                    visited_set_enabled()};

                return lib::SaveTable(
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

// Assembly
template <lib::LazyInvocable<> GraphLoader, typename DataLoader, typename Distance>
auto auto_dynamic_assemble(
    const std::filesystem::path& config_path,
    const GraphLoader& graph_loader,
    const DataLoader& data_loader,
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
    auto data = svs::detail::dispatch_load(data_loader, threadpool);

    // Load the graph.
    auto graph = graph_loader();

    // Make sure the data and the graph have the same size.
    auto datasize = data.size();
    auto graphsize = graph.n_nodes();
    if (datasize != graphsize) {
        throw ANNEXCEPTION(
            "Reloaded data has {} nodes while the graph has {} nodes!", datasize, graphsize
        );
    }

    // Unload the ID translator and config parameters.
    auto reloader = lib::LoadOverride{[&](const toml::table& table,
                                          const lib::LoadContext& ctx,
                                          const lib::Version& SVS_UNUSED(version)) {
        // If loading from the static index, then the table we recieve is itself the
        // parameters table.
        //
        // There will also be no index translation, so we use the identity translation
        // since the internal and external IDs for the static index are the samen.
        if (debug_load_from_static) {
            return std::make_tuple(
                lib::load<VamanaConfigParameters>(table, ctx),
                IDTranslator(IDTranslator::Identity(datasize))
            );
        } else {
            return std::make_tuple(
                lib::load_at<VamanaConfigParameters>(table, "parameters", ctx),
                lib::load_at<IDTranslator>(table, "translation", ctx)
            );
        }
    }};
    auto [parameters, translator] = lib::load_from_disk(reloader, config_path);

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
