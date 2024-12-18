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

// svs
#include "svs/core/data.h"
#include "svs/core/graph.h"
#include "svs/core/loading.h"
#include "svs/core/medioid.h"
#include "svs/core/query_result.h"
#include "svs/core/recall.h"
#include "svs/index/vamana/calibrate.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/search_buffer.h"
#include "svs/index/vamana/search_params.h"
#include "svs/index/vamana/vamana_build.h"
#include "svs/lib/concurrency/readwrite_protected.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"

// stl
#include <algorithm>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace svs::index::vamana {

struct VamanaIndexParameters {
    // Members
  public:
    // Computed Parameters
    size_t entry_point;
    // Construction Parameters
    VamanaBuildParameters build_parameters;
    // Search Parameters
    VamanaSearchParameters search_parameters;

  public:
    VamanaIndexParameters(
        size_t entry_point_,
        const VamanaBuildParameters& build_parameters_,
        const VamanaSearchParameters& search_parameters_
    )
        : entry_point{entry_point_}
        , build_parameters{build_parameters_}
        , search_parameters{search_parameters_} {}

    static constexpr std::string_view legacy_name = "vamana config parameters";
    static constexpr std::string_view name = "vamana index parameters";
    /// Change notes:
    ///
    /// v0.0.1 - Added the "use_full_search_history" option.
    ///     Loading from older versions default this to "true"
    /// v0.0.2 - Added the "prune_to" parameter option.
    ///     Loading from older versions default this to "graph_max_degree"
    /// v0.0.3 - Refactored to split out build parameters and search parameters into their
    ///     own pieces.
    ///
    ///     Compatible with all previous versions.
    static constexpr lib::Version save_version = lib::Version(0, 0, 3);
    static constexpr std::string_view serialization_schema = "vamana_index_parameters";

    // Save and Reload.
    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE(name),
             SVS_LIST_SAVE(entry_point),
             SVS_LIST_SAVE(build_parameters),
             SVS_LIST_SAVE(search_parameters)}
        );
    }

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == serialization_schema && version <= save_version;
    }

    static VamanaIndexParameters load_legacy(const lib::ContextFreeLoadTable& table) {
        svs::logging::warn(
            "Loading a legacy IndexParameters class. Please consider resaving this "
            "index to update the save version and prevent future breaking!\n"
        );

        if (table.version() > lib::Version{0, 0, 2}) {
            throw ANNEXCEPTION("Incompatible legacy version!");
        }

        auto this_name = lib::load_at<std::string>(table, "name");
        if (this_name != legacy_name) {
            throw ANNEXCEPTION(
                "Name mismatch! Got {}, expected {}!", this_name, legacy_name
            );
        }

        // Default "use_full_search_history" to "true" because v0.0.0 did not implement
        // it.
        bool use_full_search_history = true;
        if (table.contains("use_full_search_history")) {
            use_full_search_history = SVS_LOAD_MEMBER_AT(table, use_full_search_history);
        }

        size_t graph_max_degree = lib::load_at<size_t>(table, "max_out_degree");
        size_t prune_to = graph_max_degree;
        if (table.contains("prune_to")) {
            prune_to = lib::load_at<size_t>(table, "prune_to");
        }

        return VamanaIndexParameters{
            lib::load_at<size_t>(table, "entry_point"),
            VamanaBuildParameters{
                lib::load_at<float>(table, "alpha"),
                graph_max_degree,
                lib::load_at<size_t>(table, "construction_window_size"),
                lib::load_at<size_t>(table, "max_candidates"),
                prune_to,
                use_full_search_history},
            VamanaSearchParameters{
                SearchBufferConfig{
                    lib::load_at<size_t>(table, "default_search_window_size")},
                lib::load_at<bool>(table, "visited_set"),
                4,
                1}};
    }

    static VamanaIndexParameters load(const lib::ContextFreeLoadTable& table) {
        // Legacy load path.
        auto version = table.version();
        if (version <= lib::Version(0, 0, 2)) {
            return load_legacy(table);
        }

        auto this_name = lib::load_at<std::string>(table, "name");
        if (this_name != name) {
            throw ANNEXCEPTION("Name mismatch! Got {}, expected {}!", this_name, name);
        }

        return VamanaIndexParameters{
            SVS_LOAD_MEMBER_AT(table, entry_point),
            SVS_LOAD_MEMBER_AT(table, build_parameters),
            SVS_LOAD_MEMBER_AT(table, search_parameters),
        };
    }

    friend bool
    operator==(const VamanaIndexParameters&, const VamanaIndexParameters&) = default;
};

///
/// @brief Search scratchspace used by the Vamana index.
///
/// These can be pre-allocated and passed to the index when performing externally
/// threaded searches to reduce allocations.
///
/// **NOTE**: The members ``buffer``, ``scratch``, and ``prefetch_parameters`` are part of
/// the public API for this class. Users are free to access and manipulate these objects.
/// However, doing so incorrectly can yield undefined-behavior.
///
/// Acceptable uses are as follows:
/// * Changing the max capacity of the buffer (search window size).
/// * Enabling or disabling the visited set.
/// * After a search is performed, the entries of the buffer can be read, modified, and
///   deleted.
///
template <typename Buffer, typename Scratch> struct SearchScratchspace {
  public:
    // Members
    Buffer buffer;
    Scratch scratch;
    GreedySearchPrefetchParameters prefetch_parameters;

    // Constructors
    SearchScratchspace(
        Buffer buffer_, Scratch scratch_, GreedySearchPrefetchParameters prefetch_parameters
    )
        : buffer{std::move(buffer_)}
        , scratch{std::move(scratch_)}
        , prefetch_parameters{prefetch_parameters} {}

    // Apply the provided search parameters to the existing scratchspace.
    // In particular, the underlying buffer is modified according to the following rules:
    //
    // * If the new buffer capacity ``N`` is *less* than the old capacity ``O``, than the
    //   first ``N`` elements in the buffer will remain unchanged.
    //
    //   Further, the size (in terms of number of contained elements) of the underlying
    //   buffer will be the minimum of the previous size and the new capacity.
    //
    // * If the new buffer caspacity ``N`` is *greater* then the old capacity ``O``, then
    //   the first ``O`` elements in the buffer will remain unchanged with the contents of
    //   the remaining ``N - 0`` elements being undefined.
    //
    //   The size (in terms of number of contained elements) of the underlying buffer will
    //   be the previous size.
    SearchScratchspace& apply(const vamana::VamanaSearchParameters& p) {
        buffer.change_maxsize(p.buffer_config_);
        buffer.configure_visited_set(p.search_buffer_visited_set_);
        prefetch_parameters = {p.prefetch_lookahead_, p.prefetch_step_};
        return *this;
    }
};

// Construct the default search parameters for this index.
template <data::ImmutableMemoryDataset Data>
VamanaSearchParameters construct_default_search_parameters(const Data& data) {
    auto parameters = VamanaSearchParameters();
    auto default_prefetching = extensions::estimate_prefetch_parameters(data);
    parameters.prefetch_lookahead(default_prefetching.lookahead);
    parameters.prefetch_step(default_prefetching.step);
    return parameters;
}

///
/// @brief Implementation of the static Vamana index.
///
/// @tparam Graph The full type of the graph being used to conduct searches.
/// @tparam Data The full type of the dataset being indexed.
/// @tparam Dist The distance functor used to compare queries with elements of the
///     dataset.
/// @tparam PostOp The cleanup operation that is performed after the initial graph
///     search. This may be included to perform operations like reranking for quantized
///     datasets.
///
/// The mid-level implementation of the static Vamana graph-based index.
///
/// We can't enforce any constraints on `Dist` because at construction time, we don't
/// necessarily know what the query type is going to be (since it need not be the same
/// as the dataset elements).
///
/// Checking of the full concept is done when the `VamanaIndex` is embedded inside some
/// kind of searcher (and in the test suite).
///
template <
    graphs::ImmutableMemoryGraph Graph,
    data::ImmutableMemoryDataset Data,
    typename Dist>
class VamanaIndex {
  public:
    static constexpr bool supports_insertions = false;
    static constexpr bool supports_deletions = false;
    static constexpr bool supports_saving = true;
    static constexpr bool needs_id_translation = false;

    ///// Type Aliases

    /// The integer type used to encode entries in the graph.
    using Idx = typename Graph::index_type; // Todo (MH): deprecate?
    /// The integer type used internally for IDs.
    using internal_id_type = Idx;
    /// The integer type used for external IDs.
    using external_id_type = Idx;
    /// The type of entries in the dataset.
    using value_type = typename Data::value_type;
    /// The type of constant entries in the dataset.
    using const_value_type = typename Data::const_value_type;
    /// The static dimensionality of the dataset.
    static constexpr size_t extent = Data::extent;
    /// Type of the distance functor.
    using distance_type = Dist;
    using search_buffer_type = SearchBuffer<Idx, distance::compare_t<Dist>>;
    /// Type of the graph.
    using graph_type = Graph;
    /// Type of the dataset.
    using data_type = Data;
    using entry_point_type = std::vector<Idx>;
    /// The type of the configurable search parameters.
    using search_parameters_type = VamanaSearchParameters;

    // Members
  private:
    graph_type graph_;
    data_type data_;
    entry_point_type entry_point_;
    // Base distance type.
    distance_type distance_;
    // Thread-pool for batch queries.
    threads::NativeThreadPool threadpool_;
    // Search Parameters (protected for multiple readers and writers).
    lib::ReadWriteProtected<VamanaSearchParameters> default_search_parameters_{};
    // Construction parameters
    VamanaBuildParameters build_parameters_{};

  public:
    // This is because some datasets may not yet support single-searching, which is required
    // by the BatchIterator.
    SVS_TEMPORARY_DISABLE_SINGLE_SEARCH static constexpr bool
    temporary_disable_batch_iterator() {
        return extensions::temporary_disable_single_search<data_type>();
    }

    /// The type of the search resource used for external threading.
    using inner_scratch_type =
        svs::tag_t<extensions::single_search_setup>::result_t<Data, Dist>;
    using scratchspace_type = SearchScratchspace<search_buffer_type, inner_scratch_type>;

    /// @brief Return a copy of the primary distance functor used by the index.
    distance_type get_distance() const { return distance_; }

    ///
    /// @brief Construct a VamanaIndex from constituent parts.
    ///
    /// @param graph An existing graph over ``data`` that has been previously
    /// constructed.
    /// @param data The dataset being indexed.
    /// @param entry_point The entry-point into the graph to begin searches.
    /// @param distance_function The distance function used to compare queries and
    ///     elements of the dataset.
    /// @param threadpool The threadpool to use to conduct searches.
    ///
    /// This is a lower-level function that is meant to take a collection of
    /// instantiated components and assemble the final index. For a more "hands-free"
    /// approach, see the factory methods.
    ///
    /// **Preconditions:**
    ///
    /// * `graph.n_nodes() == data.size()`: Graph and data should have the same number
    /// of entries.
    ///
    /// @sa auto_assemble
    ///
    VamanaIndex(
        Graph graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        threads::NativeThreadPool threadpool
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{entry_point}
        , distance_{std::move(distance_function)}
        , threadpool_{std::move(threadpool)}
        , default_search_parameters_{construct_default_search_parameters(data_)} {}

    VamanaIndex(
        Graph graph, Data data, Idx entry_point, Dist distance_function, size_t num_threads
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{entry_point}
        , distance_{std::move(distance_function)}
        , threadpool_{num_threads}
        , default_search_parameters_{construct_default_search_parameters(data_)} {}

    ///
    /// @brief Build a VamanaIndex over the given dataset.
    ///
    /// @param parameters The build parameters used to construct the graph.
    /// @param graph An unpopulated graph with the same size as ``data``. This graph
    ///     will be populated as part of the constructor operation.
    /// @param data The dataset to be indexed indexed.
    /// @param entry_point The entry-point into the graph to begin searches.
    /// @param distance_function The distance function used to compare queries and
    ///     elements of the dataset.
    /// @param threadpool The threadpool to use to conduct searches.
    ///
    /// This is a lower-level function that is meant to take a dataset and construct
    /// the graph-based index over the dataset. For a more "hands-free" approach, see
    /// the factory methods.
    ///
    /// **Preconditions:**
    ///
    /// * `graph.n_nodes() == data.size()`: Graph and data should have the same number of
    ///     entries.
    ///
    /// @sa auto_build
    ///
    VamanaIndex(
        const VamanaBuildParameters& parameters,
        Graph graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        threads::NativeThreadPool threadpool
    )
        : VamanaIndex{
              std::move(graph),
              std::move(data),
              entry_point,
              std::move(distance_function),
              std::move(threadpool)} {
        if (graph_.n_nodes() != data_.size()) {
            throw ANNEXCEPTION("Wrong sizes!");
        }

        build_parameters_ = parameters;
        auto builder = VamanaBuilder(
            graph_,
            data_,
            distance_,
            parameters,
            threadpool_,
            extensions::estimate_prefetch_parameters(data_)
        );

        builder.construct(1.0F, entry_point_[0]);
        builder.construct(parameters.alpha, entry_point_[0]);
    }

    /// @brief Apply the given configuration parameters to the index.
    void apply(const VamanaIndexParameters& parameters) {
        entry_point_.clear();
        entry_point_.push_back(parameters.entry_point);

        build_parameters_ = parameters.build_parameters;
        set_search_parameters(parameters.search_parameters);
    }

    /// @brief Return scratch space resources for external threading.
    scratchspace_type scratchspace(const VamanaSearchParameters& sp) const {
        return scratchspace_type{
            search_buffer_type(
                sp.buffer_config_,
                distance::comparator(distance_),
                sp.search_buffer_visited_set_
            ),
            extensions::single_search_setup(data_, distance_),
            {sp.prefetch_lookahead_, sp.prefetch_step_}};
    }

    /// @brief Return scratch-space resources for external threading with default parameters
    ///
    /// The default parameters can be accessed using ``get_search_parameters`` and
    /// ``set_search_parameters``.
    scratchspace_type scratchspace() const { return scratchspace(get_search_parameters()); }

    // Return a `greedy_search` compatible builder for this index.
    // This is an internal method, mostly used to help implement the batch iterator.
    static NeighborBuilder internal_search_builder() { return NeighborBuilder(); }

    auto greedy_search_closure(
        GreedySearchPrefetchParameters prefetch_parameters,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) const {
        return [&, prefetch_parameters](
                   const auto& query, auto& accessor, auto& distance, auto& buffer
               ) {
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
        };
    }

    ///
    /// @brief Perform a nearest neighbor search for query using the provided scratch
    /// space.
    ///
    /// Operations performed:
    /// * Graph search to obtain k-nearest neighbors.
    /// * Search result reranking (if needed).
    ///
    /// Results will be present in the data structures contained inside ``scratch``.
    /// Extraction should pull out the search buffer for extra post-processing.
    ///
    /// **Note**: It is the caller's responsibility to ensure that the scratch space has
    /// been initialized properly to return the requested number of neighbors.
    ///
    template <typename Query>
    void search(
        const Query& query,
        scratchspace_type& scratch,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) const {
        extensions::single_search(
            data_,
            scratch.buffer,
            scratch.scratch,
            query,
            greedy_search_closure(scratch.prefetch_parameters, cancel)
        );
    }

    ///
    /// @brief Fill the result with the ``num_neighbors`` nearest neighbors for each
    /// query.
    ///
    /// @tparam I The integer type used to encode neighbors in the QueryResult.
    /// @tparam QueryType The element of the queries.
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
    ///
    /// Perform a multi-threaded graph search over the index, overwriting the contents
    /// of ``result`` with the results of search.
    ///
    /// After the initial graph search, a post-operation defined by the ``PostOp`` type
    /// parameter will be conducted which may conduct refinement on the candidates.
    ///
    /// If the current value of ``get_search_window_size()`` is less than
    /// ``num_neighbors``, it will temporarily be set to ``num_neighbors``.
    ///
    /// **Preconditions:**
    ///
    /// The following pre-conditions must hold. Otherwise, the behavior is undefined.
    /// - ``result.n_queries() == queries.size()``
    /// - ``result.n_neighbors() == num_neighbors``.
    /// - The value type of ``queries`` is compatible with the value type of the index
    ///     dataset with respect to the stored distance functor.
    ///
    template <typename I, data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<I> result,
        const Queries& queries,
        const search_parameters_type& search_parameters,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        threads::run(
            threadpool_,
            threads::StaticPartition{queries.size()},
            [&](const auto is, uint64_t SVS_UNUSED(tid)) {
                // The number of neighbors to store in the result.
                size_t num_neighbors = result.n_neighbors();

                // Allocate scratchspace according to the provided search parameters.
                auto search_buffer = search_buffer_type{
                    SearchBufferConfig(search_parameters.buffer_config_),
                    distance::comparator(distance_),
                    search_parameters.search_buffer_visited_set_};

                auto prefetch_parameters = GreedySearchPrefetchParameters{
                    search_parameters.prefetch_lookahead_,
                    search_parameters.prefetch_step_};

                // Increase the search window size if the defaults are not suitable for the
                // requested number of neighbors.
                if (search_buffer.capacity() < num_neighbors) {
                    search_buffer.change_maxsize(SearchBufferConfig{num_neighbors});
                }

                // Pre-allocate scratch space needed by the dataset implementation.
                auto scratch = extensions::per_thread_batch_search_setup(data_, distance_);

                // Perform a search over the batch of queries.
                extensions::per_thread_batch_search(
                    data_,
                    search_buffer,
                    scratch,
                    queries,
                    result,
                    threads::UnitRange{is},
                    greedy_search_closure(prefetch_parameters, cancel),
                    cancel
                );
            }
        );
    }

    // TODO (Mark): Make descriptions better.
    std::string name() const { return "VamanaIndex"; }

    ///// Dataset Interface

    /// @brief Return the number of vectors in the index.
    size_t size() const { return data_.size(); }

    /// @brief Return the logical number aLf dimensions of the indexed vectors.
    size_t dimensions() const { return data_.dimensions(); }

    /// @brief Reconstruct vectors.
    ///
    /// Reconstruct each vector indexed by an external ID and store the results into
    /// ``dst``.
    ///
    /// Preconditions:
    /// - ``ids.size() == svs::getsize<0>(dst)``: Each ID has a corresponding entry in the
    ///     destination array.
    /// - ``0 <= i < size()`` for all ``i`` in ``ids``: Each ID is in-bounds.
    /// - ``svs::getsize<1>(dst) == dimensions()``: The space allocated for each vector in
    ///     ``dst`` is correct.
    ///
    /// An exception will be thrown if any of these pre-conditions does not hold.
    /// If such an exception is thrown, the argument ``dst`` will be left unmodified.
    template <std::unsigned_integral I, svs::Arithmetic T>
    void reconstruct_at(data::SimpleDataView<T> dst, std::span<const I> ids) {
        // Check pre-conditions.
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
        const size_t sz = size();
        for (size_t i = 0; i < ids_size; ++i) {
            I id = ids[i]; // inbounds by loop bounds.
            if (id >= sz) {
                throw ANNEXCEPTION("ID {} with value {} is out of bounds!", i, id);
            }
        }

        // Prerequisites checked - proceed with the operation.
        // TODO: Communicate the requested decompression type to the backend dataset to
        // allow more fine-grained specialization?
        auto threaded_function = [&](auto is, uint64_t SVS_UNUSED(tid)) {
            auto accessor = extensions::reconstruct_accessor(data_);
            for (auto i : is) {
                auto id = ids[i];
                dst.set_datum(i, accessor(data_, id));
            }
        };
        threads::run(threadpool_, threads::StaticPartition{ids_size}, threaded_function);
    }

    ///// Threading Interface

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

    ///
    /// @brief Obtain the underlying threadpool.
    ///
    /// N.B.: Jobs may be run on the thread pool but it may not be resized safely.
    ///
    threads::NativeThreadPool& borrow_threadpool() { return threadpool_; }

    ///// Search Parameter Setting

    ///
    /// @brief Return the current search parameters stored by the index.
    ///
    /// These parameters include
    ///
    VamanaSearchParameters get_search_parameters() const {
        return default_search_parameters_.get();
    }

    void populate_search_parameters(VamanaSearchParameters& parameters) const {
        parameters = get_search_parameters();
    }

    void set_search_parameters(const VamanaSearchParameters& parameters) {
        default_search_parameters_.set(parameters);
    }

    ///
    /// @brief Reset performance parameters to their default values for this index.
    ///
    /// Parameters affected are only those that modify throughput on a given architecture.
    /// Accuracy results should not change as a side-effect of calling this function.
    ///
    void reset_performance_parameters() {
        auto p = get_search_parameters();
        auto prefetch_parameters = extensions::estimate_prefetch_parameters(data_);
        p.prefetch_lookahead(prefetch_parameters.lookahead);
        p.prefetch_step(prefetch_parameters.step);
        set_search_parameters(p);
    }

    ///// Parameter manipulation.

    /// @brief Get the value of ``alpha`` used during graph construction.
    float get_alpha() const { return build_parameters_.alpha; }
    /// @brief Set the value of ``alpha`` to be used during graph construction.
    void set_alpha(float alpha) { build_parameters_.alpha = alpha; }

    /// @brief Get the ``graph_max_degree`` that was used for graph construction.
    size_t get_graph_max_degree() const { return graph_.max_degree(); }

    /// @brief Get the max candidate pool size that was used for graph construction.
    size_t get_max_candidates() const { return build_parameters_.max_candidate_pool_size; }
    /// @brief Set the max candidate pool size to be used for graph construction.
    void set_max_candidates(size_t max_candidates) {
        build_parameters_.max_candidate_pool_size = max_candidates;
    }

    /// @brief Get the prune_to value that was used for graph construction.
    size_t get_prune_to() const { return build_parameters_.prune_to; }
    /// @brief Set the prune_to value to be used for graph construction.
    void set_prune_to(size_t prune_to) { build_parameters_.prune_to = prune_to; }

    /// @brief Get the search window size that was used for graph construction.
    size_t get_construction_window_size() const { return build_parameters_.window_size; }
    /// @brief Set the search window size to be used for graph construction.
    void set_construction_window_size(size_t construction_window_size) {
        build_parameters_.window_size = construction_window_size;
    }

    /// @brief Return whether the full search history is being used for index
    /// construction.
    bool get_full_search_history() const {
        return build_parameters_.use_full_search_history;
    }
    /// @brief Enable using the full search history for candidate generation while
    /// building.
    void set_full_search_history(bool enable) {
        build_parameters_.use_full_search_history = enable;
    }

    ///// Saving

    ///
    /// @brief Save the whole index to disk to enable reloading in the future.
    ///
    /// @param config_directory Directory where the index metadata will be saved.
    ///     The contents of the metadata include the configured search window size,
    ///     entry point etc.
    /// @param graph_directory Directory where the graph will be saved.
    /// @param data_directory Directory where the vector dataset will be saved.
    ///
    /// In general, the save locations must be directories since each data structure
    /// (config, graph, and data) may require an arbitrary number of auxiliary files in
    /// order to by completely saved to disk. The given directories must be different.
    ///
    /// Each directory may be created as a side-effect of this method call provided that
    /// the parent directory exists. Data in existing directories will be overwritten
    /// without warning.
    ///
    /// The choice to use a multi-directory structure is to enable design-space
    /// exploration with different data quantization techniques or graph
    /// implementations. That is, the save format for the different components is
    /// designed to be orthogonal to allow mixing and matching of different types upon
    /// reloading.
    ///
    void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& graph_directory,
        const std::filesystem::path& data_directory
    ) const {
        // Construct and save runtime parameters.
        auto parameters = VamanaIndexParameters{
            entry_point_.front(), build_parameters_, get_search_parameters()};

        // Config
        lib::save_to_disk(parameters, config_directory);
        // Data
        lib::save_to_disk(data_, data_directory);
        // Graph
        lib::save_to_disk(graph_, graph_directory);
    }

    ///// Calibration

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

    ///// Experimental Methods

    /// Invoke the provided callable with constant references to the contained graph, data,
    /// and entry points.
    ///
    /// This function is meant to provide a means for implementing experimental algorithms
    /// on the contained data structures.
    template <typename F> void experimental_escape_hatch(F&& f) const {
        std::invoke(SVS_FWD(f), graph_, data_, distance_, lib::as_const_span(entry_point_));
    }
};

// Shared documentation for assembly methods.

///
/// @brief Entry point for loading a Vamana graph-index from disk.
///
/// @param parameters The parameters to use for graph construction.
/// @param data_proto A dispatch loadable class yielding a dataset.
/// @param distance The distance **functor** to use to compare queries with elements of
///     the dataset.
/// @param threadpool_proto Precursor for the thread pool to use. Can either be a
///     threadpool instance of an integer specifying the number of threads to use.
/// @param graph_allocator The allocator to use for the graph data structure.
///
template <
    typename DataProto,
    typename Distance,
    typename ThreadpoolProto,
    typename Allocator = HugepageAllocator<uint32_t>>
auto auto_build(
    const VamanaBuildParameters& parameters,
    DataProto data_proto,
    Distance distance,
    ThreadpoolProto threadpool_proto,
    const Allocator& graph_allocator = {}
) {
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data = svs::detail::dispatch_load(std::move(data_proto), threadpool);
    auto entry_point = extensions::compute_entry_point(data, threadpool);

    // Default graph.
    auto graph = default_graph(data.size(), parameters.graph_max_degree, graph_allocator);
    using I = typename decltype(graph)::index_type;
    return VamanaIndex{
        parameters,
        std::move(graph),
        std::move(data),
        lib::narrow<I>(entry_point),
        std::move(distance),
        std::move(threadpool)};
}

///
/// @brief Entry point for loading a Vamana graph-index from disk.
///
/// @param config_path The directory where the index configuration file resides.
/// @param graph_loader A ``svs::GraphLoader`` for loading the graph.
/// @param data_proto A dispatch loadable class yielding a dataset.
/// @param distance The distance **functor** to use to compare queries with elements of
///        the dataset.
/// @param threadpool_proto Precursor for the thread pool to use. Can either be a
///        threadpool instance of an integer specifying the number of threads to use.
///
/// This method provides much of the heavy lifting for instantiating a Vamana index from
/// a collection of files on disk (or perhaps a mix-and-match of existing data in-memory
/// and on disk).
///
/// Refer to the examples for use of this interface.
///
template <
    typename GraphProto,
    typename DataProto,
    typename Distance,
    typename ThreadPoolProto>
auto auto_assemble(
    const std::filesystem::path& config_path,
    GraphProto graph_loader,
    DataProto data_proto,
    Distance distance,
    ThreadPoolProto threadpool_proto
) {
    auto threadpool = threads::as_threadpool(threadpool_proto);
    auto data = svs::detail::dispatch_load(std::move(data_proto), threadpool);
    auto graph = svs::detail::dispatch_load(std::move(graph_loader), threadpool);

    // Extract the index type of the provided graph.
    using I = typename decltype(graph)::index_type;
    auto index = VamanaIndex{
        std::move(graph), std::move(data), I{}, std::move(distance), std::move(threadpool)};

    auto config = lib::load_from_disk<VamanaIndexParameters>(config_path);
    index.apply(config);
    return index;
}
} // namespace svs::index::vamana
