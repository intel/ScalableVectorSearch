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

// vector quantization
#include "svs/quantization/lvq/lvq.h"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/core/medioid.h"
#include "svs/core/query_result.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/search_buffer.h"
#include "svs/index/vamana/vamana_build.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"
#include "svs/lib/traits.h"

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

///
/// @brief No operation following graph search.
///
/// Following the main search routine, we have an opportunity to modify the contents
/// of the search buffer.
///
/// For example, if a quantization technique is used, we may wish to introduce a reranking
/// step that has access to the uncompressed dataset to more accurately recompute distances
/// between returned entities in the `SearchBuffer`.
///
/// By default, no such post-operation occurs and that is described by this struct.
///
/// All defined VamanaIndex post-operations should define `operator()` like the template
/// below.
///
struct NoPostOp {
    ///
    /// @brief Feed-through post-operation.
    ///
    /// @param search_buffer The search buffer following graph search. This function is free
    ///     to modify the state of the search buffer if desired.
    /// @param dataset The primary dataset stored in the index.
    /// @param query The current query.
    /// @param distance The distance functor. It can be assumed that
    ///     ``distance::maybe_fix_argument`` has already been called with the current query.
    /// Stub function that simply returns the search buffer unmodified.
    template <typename SearchBuffer, typename Dataset, typename Query, typename Distance>
    SearchBuffer& operator()(
        SearchBuffer& search_buffer,
        const Dataset& SVS_UNUSED(dataset),
        const Query& SVS_UNUSED(query),
        Distance& SVS_UNUSED(distance)
    ) {
        return search_buffer;
    }

    ///// IO
    static constexpr std::string_view name = "NoPostOp";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveType save(const lib::SaveContext& /*ctx*/) const {
        return lib::SaveType(toml::table({{"name", name}}), save_version);
    }

    static NoPostOp load(
        const toml::table& table,
        const lib::LoadContext& /*ctx*/,
        const lib::Version& version
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }

        auto this_name = get(table, "name").value();
        if (this_name != name) {
            throw ANNEXCEPTION(
                "Trying to load NoPostOp but found ", this_name, " instead!"
            );
        }
        return NoPostOp();
    }
};

struct VamanaConfigParameters {
    static constexpr std::string_view name = "vamana config parameters";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    // Save and Reload.
    lib::SaveType save(const lib::SaveContext& /*ctx*/) const {
        auto table = toml::table(
            {{"name", name},
             {"max_out_degree", prepare(graph_max_degree)},
             {"entry_point", prepare(entry_point)},
             {"alpha", prepare(alpha)},
             {"max_candidates", prepare(max_candidates)},
             {"construction_window_size", prepare(construction_window_size)},
             {"default_search_window_size", prepare(search_window_size)},
             {"visited_set", visited_set}}
        );
        return std::make_pair(std::move(table), save_version);
    }

    static VamanaConfigParameters load(
        const toml::table& table,
        const lib::LoadContext& SVS_UNUSED(ctx),
        const lib::Version& /*version*/
    ) {
        auto this_name = get(table, "name").value();
        if (this_name != name) {
            throw ANNEXCEPTION(
                fmt::format("Name mismatch! Got {}, expected {}!", this_name, name)
            );
        }

        return VamanaConfigParameters{
            get<size_t>(table, "max_out_degree"),
            get<size_t>(table, "entry_point"),
            get<float>(table, "alpha"),
            get<size_t>(table, "max_candidates"),
            get<size_t>(table, "construction_window_size"),
            get<size_t>(table, "default_search_window_size"),
            get<bool>(table, "visited_set")};
    }

    // Members
  public:
    // general parameters
    size_t graph_max_degree;
    size_t entry_point;
    // construction parameters
    float alpha;
    size_t max_candidates;
    size_t construction_window_size;
    // runtime parameters
    size_t search_window_size;
    bool visited_set;
};

///// Saving

// TODO: We probably need to have a more formal method of dispatching to the correct
// saving patterns based on the provenance of the original data.
//
// This would require propagating semantically *what kind* of dataset we're handling
// in the index, but the complexity may be worth it. Otherwise, I'm not sure how we're
// going to reliably generalize saving.
struct UnknownSaver {
    static constexpr bool is_vamana_saver = true;
    static constexpr bool saving_enabled = false;
};

struct UncompressedSaver {
    static constexpr bool is_vamana_saver = true;
    static constexpr bool saving_enabled = true;

    template <data::ImmutableMemoryDataset Data, typename Distance>
    void save_dataset(
        const std::filesystem::path& dir,
        const Data& data,
        const Distance& SVS_UNUSED(distance),
        NoPostOp SVS_UNUSED(post_op)
    ) const {
        lib::save(data, dir);
    }
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
    typename Dist,
    typename PostOp = NoPostOp,
    typename Saver = UnknownSaver>
class VamanaIndex {
  public:
    static constexpr bool supports_insertions = false;
    static constexpr bool supports_deletions = false;
    static constexpr bool supports_saving = Saver::saving_enabled;

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

    // Thread local data structures.
    distance::BroadcastDistance<Dist> distance_;

    // "prototype" because it is copy-constructed by each thread when it begins working on
    // a batch of queries.
    search_buffer_type search_buffer_prototype_ = {};
    threads::NativeThreadPool threadpool_;
    PostOp post_op_;
    [[no_unique_address]] Saver saver_ = Saver{};

    // Construction parameters
    float alpha_ = 0.0;
    size_t max_candidates_ = 1'000;
    size_t construction_window_size_ = 0;

    // Methods
  public:
    ///
    /// @brief Construct a VamanaIndex from constituent parts.
    ///
    /// @param graph An existing graph over ``data`` that has been previously constructed.
    /// @param data The dataset being indexed.
    /// @param entry_point The entry-point into the graph to begin searches.
    /// @param distance_function The distance function used to compare queries and
    ///     elements of the dataset.
    /// @param threadpool The threadpool to use to conduct searches.
    /// @param post_op Cleanup operations to occur after the graph search stage for any
    ///     reranking or similar procedure.
    /// @param saver The helper class used to save the dataset.
    ///
    /// This is a lower-level function that is meant to take a collection of instantiated
    /// components and assemble the final index. For a more "hands-free" approach, see
    /// the factory methods.
    ///
    /// **Preconditions:**
    ///
    /// * `graph.n_nodes() == data.size()`: Graph and data should have the same number of
    ///     entries.
    ///
    /// @sa auto_assemble
    ///
    VamanaIndex(
        Graph graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        threads::NativeThreadPool threadpool,
        PostOp post_op = PostOp{},
        Saver saver = Saver{}
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{entry_point}
        , distance_{std::move(distance_function), threadpool.size()}
        , search_buffer_prototype_{}
        , threadpool_{std::move(threadpool)}
        , post_op_{std::move(post_op)}
        , saver_{std::move(saver)} {}

    VamanaIndex(
        Graph graph,
        Data data,
        Idx entry_point,
        Dist distance_function,
        size_t num_threads,
        Saver saver = Saver{}
    )
        : graph_{std::move(graph)}
        , data_{std::move(data)}
        , entry_point_{entry_point}
        , distance_{std::move(distance_function), num_threads}
        , search_buffer_prototype_{}
        , threadpool_{num_threads}
        , saver_{std::move(saver)} {}

    ///
    /// @brief Build a VamanaIndex over the given dataset.
    ///
    /// @param parameters The build parameters used to construct the graph.
    /// @param graph An unpopulated graph with the same size as ``data``. This graph will
    ///     BE POPULATED AS PART OF THE CONSTRUCTOR OPERATIon.
    /// @param data The dataset to be indexed indexed.
    /// @param entry_point The entry-point into the graph to begin searches.
    /// @param distance_function The distance function used to compare queries and
    ///     elements of the dataset.
    /// @param threadpool The threadpool to use to conduct searches.
    /// @param saver The helper class used to save the dataset.
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
        threads::NativeThreadPool threadpool,
        Saver saver = Saver{}
    )
        : VamanaIndex{
              std::move(graph),
              std::move(data),
              entry_point,
              std::move(distance_function),
              std::move(threadpool),
              PostOp(),
              std::move(saver)} {
        if (graph_.n_nodes() != data_.size()) {
            throw ANNEXCEPTION("Wrong sizes!");
        }

        alpha_ = parameters.alpha;
        max_candidates_ = parameters.max_candidate_pool_size;
        construction_window_size_ = parameters.window_size;

        auto builder = VamanaBuilder(graph_, data_, distance_[0], parameters, threadpool_);
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
        parameters.visited_set ? enable_visited_set() : disable_visited_set();
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
    ///
    /// Internally, this method calls the mutating version of search. See the documentation
    /// of that method for more details.
    ///
    template <data::ImmutableMemoryDataset Queries>
    QueryResult<size_t> search(const Queries& queries, size_t num_neighbors) {
        QueryResult<size_t> result{queries.size(), num_neighbors};
        search(queries, num_neighbors, result.view());
        return result;
    }

    ///
    /// @brief Fill the result with the ``num_neighbors`` nearest neighbors for each query.
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
    /// Perform a multi-threaded graph search over the index, overwriting the contents of
    /// ``result`` with the results of search.
    ///
    /// After the initial graph search, a post-operation defined by the ``PostOp`` type
    /// parameter will be conducted which may conduct refinement on the candidates.
    ///
    /// If the current value of ``get_search_window_size()`` is less than ``num_neighbors``,
    /// it will temporarily be set to ``num_neighbors``.
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
            [&](const auto is, uint64_t tid) {
                auto buffer = threads::shallow_copy(search_buffer_prototype_);
                auto& distance = distance_[tid];

                // TODO: Use iterators for returning neighbors.
                //
                // Perform a sanity check on the search buffer.
                // If the buffer is too small, we need to set it to a minimum size to
                // avoid segfaults when extracting the neighbors.
                if (buffer.capacity() < num_neighbors) {
                    buffer.change_maxsize(num_neighbors);
                }

                for (auto i : is) {
                    const auto& query = queries.get_datum(i);

                    // Perform the greedy search.
                    // Results from the search will be present in `buffer`.
                    greedy_search(graph_, data_, query, distance, buffer, entry_point_);

                    // Copy back results.
                    post_op_(buffer, data_, query, distance);
                    for (size_t j = 0; j < num_neighbors; ++j) {
                        const auto& neighbor = buffer[j];
                        result.index(i, j) = neighbor.id();
                        result.distance(i, j) = neighbor.distance();
                    }
                }
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
        distance_.resize(num_threads);
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
    /// The choice to use a multi-directory structure is to enable design-space exploration
    /// with different data quantization techniques or graph implementations.
    /// That is, the save format for the different components is designed to be orthogonal
    /// to allow mixing and matching of different types upon reloading.
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
            get_search_window_size(),
            visited_set_enabled()};
        lib::save(parameters, config_directory);
        // Data
        saver_.save_dataset(data_directory, data_, distance_[0], post_op_);
        // Graph
        lib::save(graph_, graph_directory);
    }
};

///// Loading / Constructing

///
/// @brief Forward an existing dataset.
///
template <data::ImmutableMemoryDataset Data, typename Distance, threads::ThreadPool Pool>
std::tuple<Data&, Distance, NoPostOp, size_t, UncompressedSaver> load_dataset(
    NoopLoaderTag SVS_UNUSED(tag), Data& data, Distance distance, Pool& threadpool
) {
    size_t entry_point = utils::find_medioid(data, threadpool);
    return std::tuple<Data&, Distance, NoPostOp, size_t, UncompressedSaver>(
        data, std::move(distance), NoPostOp(), entry_point, UncompressedSaver()
    );
}

///
/// @brief Load a standard dataset.
///
template <typename T, size_t Extent, typename Distance, threads::ThreadPool Pool>
std::tuple<DefaultDataset<T, Extent>, Distance, NoPostOp, size_t, UncompressedSaver>
load_dataset(
    VectorDataLoaderTag SVS_UNUSED(tag),
    const VectorDataLoader<T, Extent>& loader,
    Distance distance,
    Pool& threadpool
) {
    auto data = loader.load();
    size_t medioid_index = utils::find_medioid(data, threadpool);
    return std::make_tuple(
        std::move(data), std::move(distance), NoPostOp(), medioid_index, UncompressedSaver()
    );
}

///// Bridge to vector quantization
template <size_t Bits, size_t Extent> class ResidualReranker {
  public:
    using dataset_type =
        quantization::lvq::CompressedDataset<quantization::lvq::Signed, Bits, Extent>;

    explicit ResidualReranker(dataset_type residuals)
        : residuals_{std::move(residuals)} {}

    ///
    /// The reranking operation.
    /// Preconditons:
    /// * `maybe_fix_argument` must already be applied to the distance function.
    ///
    template <typename SearchBuffer, typename Dataset, typename Query, typename Distance>
    SearchBuffer& operator()(
        SearchBuffer& search_buffer,
        const Dataset& primary_dataset,
        const Query& query,
        Distance& rerank_distance
    ) {
        for (size_t i = 0; i < search_buffer.size(); ++i) {
            auto& neighbor = search_buffer[i];
            auto id = neighbor.id();

            // Get the pieces needed to reconstruct the higher precision data point.
            const auto& primary = primary_dataset.get_datum(id);
            const auto& residual = residuals_.get_datum(id);
            auto stitched = quantization::lvq::combine(primary, residual);
            auto new_distance = distance::compute(rerank_distance, query, stitched);
            neighbor.set_distance(new_distance);
        }
        // Resort the search buffer now that we've updated the distances.
        search_buffer.sort();
        return search_buffer;
    }

    ///
    /// @brief Return a constant reference to the residual dataset.
    ///
    /// This method is largely intended to allow saving of the reranking dataset.
    ///
    const dataset_type& underlying() const { return residuals_; }

  private:
    dataset_type residuals_;
};

///
/// @brief One level vector compression strategies require no reranking.
///
inline NoPostOp wrap_reranker(quantization::lvq::NoResidual /*unused*/) {
    return NoPostOp();
}

inline quantization::lvq::NoResidual unwrap_reranker(NoPostOp /*unused*/) {
    return quantization::lvq::NoResidual();
}

///
/// @brief Use the residual dataset to instantiate a post-search reranker.
///
/// @param residual The residual encoded dataset.
///
template <size_t Bits, size_t Extent>
ResidualReranker<Bits, Extent> wrap_reranker(
    quantization::lvq::CompressedDataset<quantization::lvq::Signed, Bits, Extent> residual
) {
    return ResidualReranker<Bits, Extent>(std::move(residual));
}

template <size_t Bits, size_t Extent>
const quantization::lvq::CompressedDataset<quantization::lvq::Signed, Bits, Extent>&
unwrap_reranker(const ResidualReranker<Bits, Extent>& reranker) {
    return reranker.underlying();
}

struct LVQSaver {
    static constexpr bool is_vamana_saver = true;
    static constexpr bool saving_enabled = true;

    template <data::ImmutableMemoryDataset Primary, typename Distance, typename PostOp>
    void save_dataset(
        const std::filesystem::path& dir,
        const Primary& data,
        const Distance& distance,
        const PostOp& post_op
    ) const {
        auto temp = quantization::lvq::DistanceMainResidualRef{
            distance, data, unwrap_reranker(post_op)};
        lib::save(temp, dir);
    }
};

template <typename LVQLoader, typename Distance, threads::ThreadPool Pool>
auto load_dataset(
    quantization::lvq::CompressorTag SVS_UNUSED(tag),
    const LVQLoader& loader,
    const Distance& distance,
    Pool& threadpool
) {
    // TODO: Propagate threat pools around to avoid creating new threads all the time.
    auto bundle = loader.load(distance, threadpool.size());

    // Compute the index of the approximate medioid
    size_t medioid_index = utils::find_medioid(
        bundle.main,
        threadpool,
        lib::ReturnsTrueType(),           /*predicate*/
        quantization::lvq::Decompressor() /*element-wise map*/
    );

    return std::make_tuple(
        std::move(bundle.main),
        std::move(bundle.distance),
        wrap_reranker(std::move(bundle.residual)),
        medioid_index,
        LVQSaver()
    );
}

///
/// @brief Resolve a filename as an entrypoint using the ``load_entry_point`` method.
///
/// @param path The file path to the metadata file containing the entry point.
/// @param type The type of the entry point integer.
///
/// Read the metadata file given by ``path`` and return the parsed entry point.
///
template <std::integral I>
I resolve_entry_point(const std::string& path, meta::Type<I> SVS_UNUSED(type)) {
    return load_entry_point<I>(path);
}

///
/// @brief Foward the given entry point.
///
/// @param entry_point Externally supplied entry point.
/// @param type The desired type for the entry point.
///
/// Assume that the entry point has been given externally and simply forward the entry
/// point.
///
/// Can throw a narrowing error if the passed entry point is not losslessly convertible
/// to the desired type.
///
template <std::integral U, std::integral I>
I resolve_entry_point(U entry_point, meta::Type<I> SVS_UNUSED(type)) {
    return lib::narrow<I>(entry_point);
}

// Shared documentation for assembly methods.

///
/// @class hidden_vamana_auto_assemble_doc
///
/// data_loader
/// ===========
///
/// The data loader should be an instance of one of the classes below.
///
/// * An instance of ``svs::VectorDataLoader``.
/// * An LVQ loader: ``svs::quantization::lvq::OneLevelWithBias`` or
///   ``svs::quantization::lvq::TwoLevelWithBias``.
/// * An implementation of ``svs::data::ImmutableMemoryDataset`` (passed by value).
///

///
/// @class hidden_vamana_auto_build_doc
///
/// data_loader
/// ===========
///
/// The data loader should be an instance of one of the classes below.
///
/// * An instance of ``VectorDataLoader``.
/// * An LVQ loader: ``svs::quantization::lvq::OneLevelWithBias``.
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
/// @copydoc hidden_vamana_auto_build_doc
///
template <
    typename DataProto,
    typename Distance,
    typename ThreadpoolProto,
    typename Allocator = HugepageAllocator>
auto auto_build(
    const VamanaBuildParameters& parameters,
    DataProto data_proto,
    Distance distance,
    ThreadpoolProto threadpool_proto,
    const Allocator& graph_allocator
) {
    auto threadpool = threads::as_threadpool(threadpool_proto);
    auto [data, final_distance, post_op, entry_point, saver] =
        load_dataset(lib::loader_tag<DataProto>, data_proto, distance, threadpool);

    // TODO: Limitation for now. Can't build indexes that require post-ops.
    static_assert(
        std::is_same_v<std::decay_t<decltype(post_op)>, NoPostOp>,
        "Cannot yet build indexes that require post operations!"
    );

    // Default graph.
    auto graph = default_graph(data.size(), parameters.graph_max_degree, graph_allocator);
    using I = typename decltype(graph)::index_type;
    return VamanaIndex{
        parameters,
        std::move(graph),
        std::move(data),
        lib::narrow<I>(entry_point),
        std::move(final_distance),
        std::move(threadpool),
        std::move(saver)};
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
/// @copydoc hidden_vamana_auto_assemble_doc
///
/// Refer to the examples for use of this interface.
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
    auto [data, final_distance, post_op, computed_entry_point, saver] =
        load_dataset(lib::loader_tag<DataProto>, data_proto, distance, threadpool);

    auto graph = graph_loader.load();
    // Extract the index type of the provided graph.
    using I = typename decltype(graph)::index_type;
    auto index = VamanaIndex{
        std::move(graph),
        std::move(data),
        lib::narrow<I>(computed_entry_point),
        std::move(final_distance),
        std::move(threadpool),
        std::move(post_op),
        std::move(saver)};

    auto config = lib::load<VamanaConfigParameters>(config_path);
    index.apply(config);
    return index;
}
} // namespace svs::index::vamana
