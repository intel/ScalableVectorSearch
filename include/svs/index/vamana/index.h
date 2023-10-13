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

// Include the flat index to spin-up exhaustive searches on demand.
#include "svs/index/flat/flat.h"

// svs
#include "svs/core/data.h"
#include "svs/core/graph.h"
#include "svs/core/loading.h"
#include "svs/core/medioid.h"
#include "svs/core/query_result.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/search_buffer.h"
#include "svs/index/vamana/vamana_build.h"
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

struct VamanaConfigParameters {
    static constexpr std::string_view name = "vamana config parameters";
    /// Change notes:
    ///
    /// v0.0.1 - Added the "use_full_search_history" option.
    ///     Loading from older versions default this to "true"
    /// v0.0.2 - Added the "prune_to" parameter option.
    ///     Loading from older versions default this to "graph_max_degree"
    static constexpr lib::Version save_version = lib::Version(0, 0, 2);

    // Save and Reload.
    lib::SaveTable save() const {
        return lib::SaveTable(
            save_version,
            {
                SVS_LIST_SAVE(name),
                {"max_out_degree", lib::save(graph_max_degree)},
                SVS_LIST_SAVE(entry_point),
                SVS_LIST_SAVE(alpha),
                SVS_LIST_SAVE(max_candidates),
                SVS_LIST_SAVE(construction_window_size),
                SVS_LIST_SAVE(prune_to),
                {"default_search_window_size", lib::save(search_window_size)},
                SVS_LIST_SAVE(visited_set),
                SVS_LIST_SAVE(use_full_search_history),
            }
        );
    }

    static VamanaConfigParameters
    load(const toml::table& table, const lib::Version& version) {
        if (version > lib::Version(0, 0, 2)) {
            throw ANNEXCEPTION("Version mismatch!");
        }

        auto this_name = lib::load_at<std::string>(table, "name");
        if (this_name != name) {
            throw ANNEXCEPTION("Name mismatch! Got {}, expected {}!", this_name, name);
        }

        // Default "use_full_search_history" to "true" because v0.0.0 did not implement it.
        bool use_full_search_history = true;
        if (table.contains("use_full_search_history")) {
            use_full_search_history = SVS_LOAD_MEMBER_AT(table, use_full_search_history);
        }

        size_t graph_max_degree = lib::load_at<size_t>(table, "max_out_degree");
        size_t prune_to = graph_max_degree;
        if (table.contains("prune_to")) {
            prune_to = lib::load_at<size_t>(table, "prune_to");
        }

        return VamanaConfigParameters{
            graph_max_degree,
            SVS_LOAD_MEMBER_AT(table, entry_point),
            SVS_LOAD_MEMBER_AT(table, alpha),
            SVS_LOAD_MEMBER_AT(table, max_candidates),
            SVS_LOAD_MEMBER_AT(table, construction_window_size),
            prune_to,
            use_full_search_history,
            lib::load_at<size_t>(table, "default_search_window_size"),
            SVS_LOAD_MEMBER_AT(table, visited_set)};
    }

    friend bool
    operator==(const VamanaConfigParameters&, const VamanaConfigParameters&) = default;

    // Members
  public:
    // general parameters
    size_t graph_max_degree;
    size_t entry_point;
    // construction parameters
    float alpha;
    size_t max_candidates;
    size_t construction_window_size;
    size_t prune_to;
    bool use_full_search_history;
    // runtime parameters
    size_t search_window_size;
    bool visited_set;
};

///
/// @brief Search scratchspace used by the Vamana index.
///
/// These can be pre-allocated and passed to the index when performing externally
/// threaded searches to reduce allocations.
///
/// **NOTE**: The members ``buffer`` and ``scratch`` are part of the public API for
/// this class. Users are free to access and manipulate these objects. However, doing so
/// improperly can yield undefined-behavior.
///
/// Acceptable uses are as follows:
/// * Changing the max capacity of the buffer (search window size).
/// * Enabling or disabling the visited set.
/// * After a search is performed, the entries of the buffer can be read, modified, and
///   deleted.
///
template <typename Buffer, typename Scratch> struct SearchScratchspace {
  public:
  public:
    // Members
    Buffer buffer;
    Scratch scratch;

  public:
    // Constructors
    SearchScratchspace(Buffer buffer_, Scratch scratch_)
        : buffer{std::move(buffer_)}
        , scratch{std::move(scratch_)} {}
};

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

    ///// Type Aliases

    /// The integer type used to encode entries in the graph.
    using Idx = typename Graph::index_type;
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
    // Members
  private:
    graph_type graph_;
    data_type data_;
    entry_point_type entry_point_;

    // Base distance type.
    distance_type distance_;

    // "prototype" because it is copy-constructed by each thread when it begins working
    // on a batch of queries.
    search_buffer_type search_buffer_prototype_ = {};
    threads::NativeThreadPool threadpool_;

    // Construction parameters
    float alpha_ = 0.0;
    size_t max_candidates_ = 1'000;
    size_t construction_window_size_ = 0;
    size_t prune_to_ = 0;
    bool use_full_search_history_ = true;

    // Methods
  public:
    /// The type of the search resource used for external threading.
    using inner_scratch_type =
        svs::tag_t<extensions::single_search_setup>::result_t<Data, Dist>;
    using scratchspace_type = SearchScratchspace<search_buffer_type, inner_scratch_type>;

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
    /// of
    ///     entries.
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
        , search_buffer_prototype_{}
        , threadpool_{std::move(threadpool)} {}

    VamanaIndex(
        Graph graph, Data data, Idx entry_point, Dist distance_function, size_t num_threads
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{entry_point}
        , distance_{std::move(distance_function)}
        , search_buffer_prototype_{}
        , threadpool_{num_threads} {}

    ///
    /// @brief Build a VamanaIndex over the given dataset.
    ///
    /// @param parameters The build parameters used to construct the graph.
    /// @param graph An unpopulated graph with the same size as ``data``. This graph
    /// will
    ///     BE POPULATED AS PART OF THE CONSTRUCTOR OPERATIon.
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
    /// * `graph.n_nodes() == data.size()`: Graph and data should have the same number
    /// of
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

        alpha_ = parameters.alpha;
        max_candidates_ = parameters.max_candidate_pool_size;
        construction_window_size_ = parameters.window_size;
        use_full_search_history_ = parameters.use_full_search_history;
        prune_to_ = parameters.prune_to;

        auto builder = VamanaBuilder(graph_, data_, distance_, parameters, threadpool_);
        builder.construct(1.0F, entry_point_[0]);
        builder.construct(parameters.alpha, entry_point_[0]);
    }

    /// @brief Apply the given configuration parameters to the index.
    void apply(const VamanaConfigParameters& parameters) {
        entry_point_.clear();
        entry_point_.push_back(parameters.entry_point);

        set_alpha(parameters.alpha);
        set_max_candidates(parameters.max_candidates);
        set_construction_window_size(parameters.construction_window_size);
        set_search_window_size(parameters.search_window_size);
        prune_to_ = parameters.prune_to;
        set_full_search_history(parameters.use_full_search_history);
        parameters.visited_set ? enable_visited_set() : disable_visited_set();
    }

    /// @brief Return scratch space resources for external threading.
    scratchspace_type scratchspace() const {
        return SearchScratchspace(
            threads::shallow_copy(search_buffer_prototype_),
            extensions::single_search_setup(data_, distance_)
        );
    }

    auto greedy_search_closure() const {
        return [&](const auto& query, const auto& accessor, auto& distance, auto& buffer) {
            greedy_search(graph_, data_, accessor, query, distance, buffer, entry_point_);
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
    void search(const Query& query, scratchspace_type& scratch) const {
        extensions::single_search(
            data_, scratch.buffer, scratch.scratch, query, greedy_search_closure()
        );
    }

    ///
    /// @brief Return the ``num_neighbors`` approximate nearest neighbors to each query.
    ///
    /// @tparam Queries The full type of the queries.
    ///
    /// @param queries The queries. Each entry will be processed.
    /// @param num_neighbors The number of approximate nearest neighbors to return for
    ///     each query.
    ///
    /// @returns A QueryResult containing ``queries.size()`` entries with position-wise
    ///     correspondence to the queries. Row `i` in the result corresponds to the
    ///     neighbors for the `i`th query. Neighbors within each row are ordered from
    ///     nearest to furthest.
    ///
    /// Internally, this method calls the mutating version of search. See the
    /// documentation of that method for more details.
    ///
    template <data::ImmutableMemoryDataset Queries>
    QueryResult<size_t> search(const Queries& queries, size_t num_neighbors) {
        QueryResult<size_t> result{queries.size(), num_neighbors};
        search(queries, num_neighbors, result.view());
        return result;
    }

    ///
    /// @brief Fill the result with the ``num_neighbors`` nearest neighbors for each
    /// query.
    ///
    /// @tparam QueryType The element of the queries.
    /// @tparam I The integer type used to encode neighbors in the QueryResult.
    ///
    /// @param queries A dense collection of queries in R^n.
    /// @param num_neighbors The number of approximate nearest neighbors to return.
    /// @param result The result data structure to populate.
    ///     Row `i` in the result corresponds to the neighbors for the `i`th query.
    ///     Neighbors within each row are ordered from nearest to furthest.
    ///
    /// Perform a multi-threaded graph search over the index, overwriting the contents
    /// of
    /// ``result`` with the results of search.
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
    template <data::ImmutableMemoryDataset Queries, typename I>
    void search(const Queries& queries, size_t num_neighbors, QueryResultView<I> result) {
        threads::run(
            threadpool_,
            threads::StaticPartition{queries.size()},
            [&](const auto is, uint64_t SVS_UNUSED(tid)) {
                auto search_buffer = threads::shallow_copy(search_buffer_prototype_);
                if (search_buffer.capacity() < num_neighbors) {
                    search_buffer.change_maxsize(num_neighbors);
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
                    greedy_search_closure()
                );
            }
        );
    }

    // TODO (Mark): Make descriptions better.
    std::string name() const { return "VamanaIndex"; }

    ///// Dataset Interface

    /// @brief Return the number of vectors in the index.
    size_t size() const { return data_.size(); }

    /// @brief Return the logical number of dimensions of the indexed vectors.
    size_t dimensions() const { return data_.dimensions(); }

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

    ///// Window Interface

    ///
    /// @brief Set the search window size used during graph search.
    ///
    /// @param search_window_size The new search window size to use.
    ///
    /// The window size provides a parameter by which accuracy can be traded for
    /// performance. Setting the search window higher will generally yield more accurate
    /// results but the search will take longer.
    ///
    /// @sa get_search_window_size
    ///
    void set_search_window_size(size_t search_window_size) {
        search_buffer_prototype_.change_maxsize(search_window_size);
    }

    ///
    /// @brief Return the current search window size
    ///
    /// @sa set_search_window_size
    ///
    size_t get_search_window_size() const { return search_buffer_prototype_.capacity(); }

    ///// Visited Set Interface
    void enable_visited_set() { search_buffer_prototype_.enable_visited_set(); }
    void disable_visited_set() { search_buffer_prototype_.disable_visited_set(); }
    bool visited_set_enabled() const {
        return search_buffer_prototype_.visited_set_enabled();
    }

    ///// Parameter manipulation.

    /// @brief Return the value of ``alpha`` used during graph construction.
    float get_alpha() const { return alpha_; }
    void set_alpha(float alpha) { alpha_ = alpha; }

    /// @brief Return the max candidate pool size that was used for graph construction.
    size_t get_max_candidates() const { return max_candidates_; }
    void set_max_candidates(size_t max_candidates) { max_candidates_ = max_candidates; }

    /// @brief Return the search window size that was used for graph construction.
    size_t get_construction_window_size() const { return construction_window_size_; }
    void set_construction_window_size(size_t construction_window_size) {
        construction_window_size_ = construction_window_size;
    }

    ///
    /// @brief Enable using the full search history for candidate generation while
    /// building.
    ///
    void set_full_search_history(bool enable) { use_full_search_history_ = enable; }

    /// @brief Return whether the full search history is being used for index
    /// construction.
    bool get_full_search_history() const { return use_full_search_history_; }

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
        // Config
        lib::save_to_disk(parameters, config_directory);
        // Data
        lib::save_to_disk(data_, data_directory);
        // Graph
        lib::save_to_disk(graph_, graph_directory);
    }
};

// Shared documentation for assembly methods.

///
/// @class hidden_vamana_auto_doc
///
/// data_loader
/// ===========
///
/// The data loader should be any object loadable via ``svs::detail::dispatch_load``
/// returning a Vamana compatible dataset. Concrete examples include:
///
/// * An instance of ``VectorDataLoader``.
/// * An LVQ loader: ``svs::quantization::lvq::LVQLoader``.
/// * An implementation of ``svs::data::ImmutableMemoryDataset`` (passed by value).
///

///
/// @brief Entry point for loading a Vamana graph-index from disk.
///
/// @param parameters The parameters to use for graph construction.
/// @param data_proto Data prototype. See expanded notes.
/// @param distance The distance **functor** to use to compare queries with elements of
///     the dataset.
/// @param threadpool_proto Precursor for the thread pool to use. Can either be a
///     threadpool instance of an integer specifying the number of threads to use.
/// @param graph_allocator The allocator to use for the graph data structure.
///
/// @copydoc hidden_vamana_auto_doc
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
    const Allocator& graph_allocator
) {
    auto threadpool = threads::as_threadpool(threadpool_proto);
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
/// @param data_proto Data prototype. See expanded notes.
/// @param distance The distance **functor** to use to compare queries with elements of
///        the dataset.
/// @param threadpool_proto Precursor for the thread pool to use. Can either be a
///        threadpool instance of an integer specifying the number of threads to use.
///
/// This method provides much of the heavy lifting for instantiating a Vamana index from
/// a collection of files on disk (or perhaps a mix-and-match of existing data in-memory
/// and on disk).
///
/// @copydoc hidden_vamana_auto_doc
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

    auto config = lib::load_from_disk<VamanaConfigParameters>(config_path);
    index.apply(config);
    return index;
}
} // namespace svs::index::vamana
