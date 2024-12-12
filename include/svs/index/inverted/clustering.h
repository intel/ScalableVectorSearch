/*
 * Copyright 2024 Intel Corporation
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
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"

#include "svs/lib/timing.h"

#include "svs/concepts/data.h"
#include "svs/core/data/view.h"
#include "svs/core/logging.h"
#include "svs/index/flat/flat.h"
#include "svs/index/inverted/common.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/prune.h"

#include "svs/index/index.h"
#include "svs/index/inverted/extensions.h"

// third-party
#include "tsl/robin_set.h"

// stl
#include <random>
#include <vector>

namespace svs::index::inverted {

struct ClusteringParameters {
  public:
    /// The percent of the original dataset to use for centroids.
    lib::Percent percent_centroids_ = lib::Percent(0.10);
    /// Pruning parameter for assignment expansion.
    double epsilon_ = 0.05;
    /// The maximum number of replicas allowed for each dataset element.
    size_t max_replicas_ = 8;
    /// The maximum cluster size allowed.
    ///
    /// Setting to the default value of ``0`` effectively
    /// disables this setting.
    ///
    /// The clustering algorithm will fail in a very niche circumstance where all elements
    /// in a cluster are the only copies of those element in the entire database and the
    /// size of that cluster exceeds the maximum size.
    size_t max_cluster_size_ = 0;
    /// Random seed to use for initialization.
    uint64_t seed_ = 0xc0ffee;
    /// Dataset batchsize to use when clustering.
    size_t batchsize_ = 100'000;
    /// The search window size to use the index.
    size_t search_window_size_ = 50;
    /// The number of intermediate results to return from index search.
    size_t num_intermediate_results_ = 20;
    /// Refinement Alpha
    double refinement_alpha_ = 1.0;

  public:
    ClusteringParameters() = default;
    ClusteringParameters(
        lib::Percent percent_centroids,
        double epsilon,
        size_t max_replicas,
        size_t max_cluster_size,
        uint64_t seed,
        size_t batchsize,
        size_t search_window_size,
        size_t num_intermediate_results,
        double refinement_alpha
    )
        : percent_centroids_{percent_centroids}
        , epsilon_{epsilon}
        , max_replicas_{max_replicas}
        , max_cluster_size_{max_cluster_size}
        , seed_{seed}
        , batchsize_{batchsize}
        , search_window_size_{search_window_size}
        , num_intermediate_results_{num_intermediate_results}
        , refinement_alpha_{refinement_alpha} {}

    // Chain setters to help with construction.
    SVS_CHAIN_SETTER_(ClusteringParameters, percent_centroids);
    SVS_CHAIN_SETTER_(ClusteringParameters, epsilon);
    SVS_CHAIN_SETTER_(ClusteringParameters, max_replicas);
    SVS_CHAIN_SETTER_(ClusteringParameters, max_cluster_size);
    SVS_CHAIN_SETTER_(ClusteringParameters, seed);
    SVS_CHAIN_SETTER_(ClusteringParameters, batchsize);
    SVS_CHAIN_SETTER_(ClusteringParameters, search_window_size);
    SVS_CHAIN_SETTER_(ClusteringParameters, num_intermediate_results);
    SVS_CHAIN_SETTER_(ClusteringParameters, refinement_alpha);

    // Comparison
    friend constexpr bool
    operator==(const ClusteringParameters&, const ClusteringParameters&) = default;

    // Saving
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "clustering_parameters";
    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {
                SVS_LIST_SAVE_(percent_centroids),
                SVS_LIST_SAVE_(epsilon),
                SVS_LIST_SAVE_(max_replicas),
                SVS_LIST_SAVE_(max_cluster_size),
                {"seed", lib::save(lib::FullUnsigned(seed_))},
                SVS_LIST_SAVE_(batchsize),
                SVS_LIST_SAVE_(search_window_size),
                SVS_LIST_SAVE_(num_intermediate_results),
                SVS_LIST_SAVE_(refinement_alpha),
            }
        );
    }

    static ClusteringParameters load(const lib::ContextFreeLoadTable& table) {
        return ClusteringParameters(
            SVS_LOAD_MEMBER_AT_(table, percent_centroids),
            SVS_LOAD_MEMBER_AT_(table, epsilon),
            SVS_LOAD_MEMBER_AT_(table, max_replicas),
            SVS_LOAD_MEMBER_AT_(table, max_cluster_size),
            lib::load_at<lib::FullUnsigned>(table, "seed"),
            SVS_LOAD_MEMBER_AT_(table, batchsize),
            SVS_LOAD_MEMBER_AT_(table, search_window_size),
            SVS_LOAD_MEMBER_AT_(table, num_intermediate_results),
            SVS_LOAD_MEMBER_AT_(table, refinement_alpha)
        );
    }
};

/// Randomly select centroids
///
/// Implementation notes:
/// * Uses uniform sampling without replacement to get the centroid IDs using rejection if
///   an ID is already sampled. May not be appropriate if the percent of centroids to
///   select is high (>30%).
template <std::integral I = uint32_t>
std::vector<I, lib::Allocator<I>> randomly_select_centroids(
    size_t data_size,
    size_t num_centroids,
    uint64_t seed,
    // const ClusteringParameters& params,
    lib::Type<I> SVS_UNUSED(integer_type) = {}
) {
    assert(num_centroids <= data_size);
    auto rng = std::mt19937_64(seed);
    // N.B.: uniform_int_distribution is inclusive.
    auto distribution = std::uniform_int_distribution<I>(0, lib::narrow<I>(data_size - 1));
    auto ids = tsl::robin_set<I>();

    // Keep generating numbers until we've reached the target number of centroids.
    while (ids.size() < num_centroids) {
        ids.insert(distribution(rng));
    }
    auto v = std::vector<I, lib::Allocator<I>>(ids.begin(), ids.end());
    std::sort(v.begin(), v.end());
    return v;
}

template <std::integral I> struct Cluster {
  public:
    I centroid_;
    std::vector<Neighbor<I>> elements_{};

  public:
    using vector_type = std::vector<Neighbor<I>>;

    // Type Aliases
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;
    using reverse_iterator = typename vector_type::reverse_iterator;
    using const_reverse_iterator = typename vector_type::const_reverse_iterator;

  public:
    explicit Cluster(I centroid)
        : centroid_{centroid} {}

    Cluster(I centroid, std::initializer_list<Neighbor<I>> elements)
        : centroid_{centroid}
        , elements_{elements} {}

    size_t size() const { return elements_.size(); }
    I centroid() const { return centroid_; }

    const std::vector<Neighbor<I>>& elements() const { return elements_; }
    std::vector<Neighbor<I>>& elements() { return elements_; }

    // Iterators
    iterator begin() { return elements_.begin(); }
    const_iterator begin() const { return elements_.begin(); }
    const_iterator cbegin() const { return elements_.cbegin(); }

    iterator end() { return elements_.end(); }
    const_iterator end() const { return elements_.cend(); }
    const_iterator cend() const { return elements_.cend(); }

    reverse_iterator rbegin() { return elements_.rbegin(); }
    const_reverse_iterator rbegin() const { return elements_.rbegin(); }
    const_reverse_iterator crbegin() const { return elements_.crbegin(); }

    reverse_iterator rend() { return elements_.rend(); }
    const_reverse_iterator rend() const { return elements_.rend(); }
    const_reverse_iterator crend() const { return elements_.crend(); }

    void push_back(Neighbor<I> id) { elements_.push_back(id); }

    template <typename Cmp> void sort(Cmp cmp = {}) {
        std::sort(elements_.begin(), elements_.end(), TotalOrder(cmp));
    }

    // Serializing and Deserializing.
    size_t serialize(std::ostream& stream) const {
        size_t bytes = lib::write_binary(stream, centroid_);
        bytes += lib::write_binary(stream, size());
        bytes += lib::write_binary(stream, elements_);
        return bytes;
    }

    static Cluster deserialize(std::istream& stream) {
        I centroid = lib::read_binary<I>(stream);
        size_t size = lib::read_binary<size_t>(stream);
        Cluster cluster(centroid);
        cluster.elements_.resize(size);
        lib::read_binary(stream, cluster.elements_);
        return cluster;
    }

    friend bool operator==(const Cluster& x, const Cluster& y) {
        if (x.centroid() != y.centroid()) {
            return false;
        }
        if (x.size() != y.size()) {
            return false;
        }
        return std::equal(x.cbegin(), x.cend(), y.cbegin(), NeighborEqual());
    }
};

struct ClusteringStats {
  public:
    size_t min_size_ = std::numeric_limits<size_t>::max();
    size_t max_size_ = std::numeric_limits<size_t>::min();
    size_t empty_clusters_ = 0;
    size_t num_clusters_ = 0;
    size_t num_leaves_ = 0;
    double mean_size_ = 0;
    double std_size_ = 0;

  public:
    template <typename Iter, typename Projection>
    ClusteringStats(Iter begin, Iter end, Projection&& proj) {
        for (auto it = begin; it != end; ++it) {
            const auto& list = proj(*it);
            num_clusters_++;
            auto these_leaves = list.size();
            num_leaves_ += these_leaves;
            min_size_ = std::min(min_size_, these_leaves);
            max_size_ = std::max(max_size_, these_leaves);
            if (these_leaves == 0) {
                empty_clusters_ += 1;
            }
        }
        mean_size_ = static_cast<double>(num_leaves_) / num_clusters_;

        // Compute the standard deviation
        double accum = 0;
        for (auto it = begin; it != end; ++it) {
            const auto& list = proj(*it);
            auto x = static_cast<double>(list.size()) - mean_size_;
            accum += x * x;
        }
        std_size_ = std::sqrt(accum / static_cast<double>(num_clusters_));
    }

    // static constexpr lib::Version save_version{0, 0, 0};
    // lib::SaveTable save() const {
    //     return lib::SaveTable(
    //         save_version,
    //         {
    //             SVS_LIST_SAVE_(min_size),
    //             SVS_LIST_SAVE_(max_size),
    //             SVS_LIST_SAVE_(empty_clusters),
    //             SVS_LIST_SAVE_(num_clusters),
    //             SVS_LIST_SAVE_(num_leaves),
    //             SVS_LIST_SAVE_(mean_size),
    //             SVS_LIST_SAVE_(std_size),
    //         }
    //     );
    // }

    std::vector<std::string> prepare_report() const {
        return std::vector<std::string>({
            SVS_SHOW_STRING_(min_size),
            SVS_SHOW_STRING_(max_size),
            SVS_SHOW_STRING_(empty_clusters),
            SVS_SHOW_STRING_(num_clusters),
            SVS_SHOW_STRING_(num_leaves),
            SVS_SHOW_STRING_(mean_size),
            SVS_SHOW_STRING_(std_size),
        });
    }

    [[nodiscard]] std::string report() const { return report(", "); }
    [[nodiscard]] std::string report(std::string_view separator) const {
        return fmt::format("{}", fmt::join(prepare_report(), separator));
    }
};

// forward declaration.
template <std::integral I> class Clustering;

namespace detail {

template <typename T> inline constexpr bool is_clustering = false;
template <std ::integral I> inline constexpr bool is_clustering<Clustering<I>> = true;
template <std ::integral I> inline constexpr bool is_clustering<const Clustering<I>> = true;

///
/// Invoke the callable `f` on each cluster in `clustering` in a deterministic ordering.
///
/// The call order is in increasing centroid index number.
///
template <typename C, typename F>
    requires is_clustering<C>
void for_each_cluster(C& clustering, F&& f) {
    // Parameter `C` is unconstrained to deduce constness.
    auto ids = clustering.sorted_centroids();
    for (auto id : ids) {
        f(clustering.at(id));
    }
}

} // namespace detail

template <std::integral I> class Clustering {
  public:
    // Type aliases
    using map_type = tsl::robin_map<I, Cluster<I>>;
    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;

  public:
    // Constructors

    /// @brief Construct a clustering from an iterator of ids.
    template <typename Iter>
    Clustering(Iter begin, Iter end)
        : clusters_{} {
        for (Iter it = begin; it != end; ++it) {
            auto id = lib::narrow<I>(*it);
            clusters_.emplace(id, id);
        }
    }

    /// @brief Construct an empty clustering.
    explicit Clustering()
        : clusters_{} {}

    // Methods

    /// @brief Return the cluster for node id `i`.
    const Cluster<I>& at(size_t i) const { return clusters_.at(i); }

    /// @brief Return the cluster for node id `i`.
    Cluster<I>& at(size_t i) { return clusters_.at(i); }

    /// @brief Return whether a cluster exists for node `i`.
    bool contains(size_t i) const { return clusters_.contains(i); }

    /// @brief Return a histogram counting the number of times a leaf element occurs.
    tsl::robin_map<I, uint32_t> leaf_histogram() const {
        auto histogram = tsl::robin_map<I, uint32_t>();
        for (auto& cluster : *this) {
            for (auto neighbor : cluster.second.elements()) {
                auto [itr, _] = histogram.try_emplace(neighbor.id(), 0);
                ++(itr.value());
            }
        }
        return histogram;
    }

    ///
    /// @brief Insert ``leaf`` into the cluster for ``centroid``
    ///
    /// Preconditions: Requires a cluster for ``centroid`` to exist.
    ///
    void insert(I centroid, Neighbor<I> leaf) { at(centroid).push_back(leaf); }

    /// @brief Insert a new cluster.
    void insert(Cluster<I>&& cluster) {
        auto id = cluster.centroid();
        if (contains(id)) {
            throw ANNEXCEPTION("Trying to add centroid {} more than once!", id);
        }
        clusters_.emplace(id, std::move(cluster));
    }

    /// @brief Return the number of clusters in the clustering.
    size_t size() const { return clusters_.size(); }

    /// @brief Return the total number of elements in the cluster, include centroids.
    size_t total_size() const {
        size_t sz = 0;
        for (auto& kv : *this) {
            sz += 1 + kv.second.size();
        }
        return sz;
    }

    ClusteringStats statistics() const {
        return ClusteringStats(cbegin(), cend(), [](auto pair) { return pair.second; });
    }

    template <typename Iter> std::vector<I> complement(Iter begin, Iter end) const {
        auto ids = std::vector<I>();
        for (Iter it = begin; it != end; ++it) {
            I i = *it;
            if (!contains(i)) {
                ids.push_back(i);
            }
        }
        return ids;
    }

    template <typename Range> std::vector<I> complement_range(Range&& range) const {
        return complement(range.begin(), range.end());
    }

    std::vector<I> complement(size_t max_size) const {
        return complement_range(threads::UnitRange(0, max_size));
    }

    // Iterator interface.
    iterator begin() { return clusters_.begin(); }
    const_iterator begin() const { return clusters_.begin(); }
    const_iterator cbegin() const { return clusters_.cbegin(); }

    iterator end() { return clusters_.end(); }
    const_iterator end() const { return clusters_.end(); }
    const_iterator cend() const { return clusters_.cend(); }

    // Comparison
    friend bool operator==(const Clustering&, const Clustering&) = default;

    // Sort all clusters for reliable comparison.
    template <typename Cmp> void sort_clusters(Cmp cmp = {}) {
        auto e = end();
        for (auto itr = begin(); itr != e; ++itr) {
            itr.value().sort(cmp);
        }
    }

    // Return the cluster ids in order.
    std::vector<I, lib::Allocator<I>> sorted_centroids() const {
        auto result = std::vector<I, lib::Allocator<I>>(size());
        size_t i = 0;
        for (auto& pair : *this) {
            result[i] = pair.first;
            ++i;
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    // Map over all clusters in a deterministic order.
    template <typename F> void for_each_cluster(F&& f) const {
        return detail::for_each_cluster(*this, SVS_FWD(f));
    }

    template <typename F> void for_each_cluster(F&& f) {
        return detail::for_each_cluster(*this, SVS_FWD(f));
    }

    tsl::robin_map<I, I> packed_leaf_translation() const {
        auto mapping = tsl::robin_map<I, I>();
        for_each_cluster([&](const auto& cluster) {
            for (auto neighbor : cluster) {
                // N.B.: `narrow_cast` is acceptable because we cannot have
                // `std::numeric_limits<I>::max()` elements in the map.
                mapping.insert({neighbor.id(), lib::narrow_cast<I>(mapping.size())});
            }
        });
        return mapping;
    }

    // Reduce to a max-size.
    template <typename Cmp>
    bool reduce_maxsize(size_t max_cluster_size, Cmp cmp, bool dry_run = false) {
        sort_clusters(cmp);
        return reduce_maxsize_sorted(max_cluster_size, dry_run);
    }

    bool reduce_maxsize_sorted(size_t max_cluster_size, bool dry_run = false) {
        // Make a histogram of the number of times each leaf element occurs in the cluster.
        auto histogram = leaf_histogram();

        // Now that we have a histogram, we can be ensured that we won't drop any database
        // items by accident.
        auto delete_list = std::vector<size_t>();
        for (auto kv = begin(), e = end(); kv != e; ++kv) {
            auto& cluster = kv.value();

            // Fast path
            auto& elements = cluster.elements();
            size_t num_elements = elements.size();
            if (num_elements <= max_cluster_size) {
                continue;
            }

            size_t elements_to_delete = num_elements - max_cluster_size;

            // Sort the elements from nearest to farthers.
            // Work backwards, marking indices for deletion only if deleting the element
            // will not cause it to be dropped from the database entirely.
            delete_list.clear();

            auto not_enough_deleted = [&]() {
                return delete_list.size() != elements_to_delete;
            };

            for (size_t j_ = 0; j_ < num_elements && not_enough_deleted(); ++j_) {
                // Convert from ascending to descending.
                size_t i = num_elements - j_ - 1;
                auto& neighbor = elements.at(i);
                auto& count = histogram.at(neighbor.id());
                // Don't delete last one.
                if (count == 1) {
                    continue;
                }

                delete_list.push_back(i);
                --count;
            }

            if (delete_list.size() != elements_to_delete) {
                if (dry_run) {
                    return false;
                }
                throw ANNEXCEPTION("Could not sufficiently reduce cluster!");
            }

            if (!dry_run) {
                // Since delete list is in descending order, we don't invalidate any
                // iterators.
                for (auto i : delete_list) {
                    elements.erase(elements.begin() + i);
                }
            }
        }
        return true;
    }

    // Saving and Loading.
    static constexpr lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "clustering";
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        // Serialize all clusters into an auxiliary file.
        auto fullpath = ctx.generate_name("clustering", "bin");
        size_t filesize = 0;
        {
            auto io = lib::open_write(fullpath);
            for (auto& cluster : *this) {
                filesize += cluster.second.serialize(io);
            }
        }

        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"filepath", lib::save(fullpath.filename())},
             SVS_LIST_SAVE(filesize),
             {"integer_type", lib::save(datatype_v<I>)},
             {"num_clusters", lib::save(size())}}
        );
    }

    static Clustering<I> load(const lib::LoadTable& table) {
        // Ensure we have the correct integer type when decoding.
        auto saved_integer_type = lib::load_at<DataType>(table, "integer_type");
        if (saved_integer_type != datatype_v<I>) {
            auto type = datatype_v<I>;
            throw ANNEXCEPTION(
                "Clustering was saved using {} but we're trying to reload it using {}!",
                saved_integer_type,
                type
            );
        }

        auto num_clusters = lib::load_at<size_t>(table, "num_clusters");
        auto expected_filesize = lib::load_at<size_t>(table, "filesize");
        auto clustering = Clustering<I>();
        {
            auto file = table.resolve_at("filepath");
            size_t actual_filesize = std::filesystem::file_size(file);
            if (actual_filesize != expected_filesize) {
                throw ANNEXCEPTION(
                    "Expected cluster file size to be {}. Instead, it is {}!",
                    actual_filesize,
                    expected_filesize
                );
            }

            auto io = lib::open_read(file);
            for (size_t i = 0; i < num_clusters; ++i) {
                clustering.insert(Cluster<I>::deserialize(io));
            }
        }
        return clustering;
    }

  private:
    tsl::robin_map<I, Cluster<I>> clusters_;
};

namespace detail {
// Helper struct to attach metadata to Neighbors
struct LocalID {
    size_t local_id = 0;
};
} // namespace detail

///
/// @brief Assign data points to clusters.
///
/// @param data The full dataset being clustered (including leaf elements and centroids).
/// @param parameters The parameters controlling global clustering behavior.
/// @param clustering The current ``Clustering`` record.
/// @param results Query results for (approximate) nearest neighbors of a subset of the
///     dataset over the centroids (see note below).
/// @param distance_functor The distance functor prototype to use when comparing dataset
///     elements to centroids.
/// @param ordinal_translator Functor translating the ordinal index of "queries" in
///     ``results`` to their global index in ``data``.
/// @param centroid_translator Functor translating id values *inside* ``results`` to their
///     global index in ``data``.
/// @param threadpool Auxiliary threadpool to use for parallelization.
///
/// The ``results`` argument has local indices of ``[0, results.n_queries())``
/// These need to be turned into global ids of ``data`` using ``translator``, which
/// takes a single integer id as an argument.
///
/// The distance functor will be passed to the customization point
/// @code{cpp}
/// svs_invoke(svs::tag_t<svs::index::inverted::clustering_distance>, const Data&)
/// @endcode{}
/// For modification prior to use.
///
template <
    data::ImmutableMemoryDataset Data,
    std::integral I,
    std::integral J,
    typename Distance,
    typename F,
    typename G,
    threads::ThreadPool Pool>
void post_process_neighbors(
    const Data& data,
    const ClusteringParameters& parameters,
    Clustering<I>& clustering,
    const QueryResult<J>& results,
    const Distance& distance_functor,
    const F& ordinal_translator,
    const G& centroid_translator,
    Pool& threadpool // <-- borrowed threadpool. Don't resize.
) {
    // Pre-allocate scratch data structures.
    auto locks = std::vector<SpinLock>(clustering.size());
    auto impl = [&](auto range, auto SVS_UNUSED(tid)) {
        // Unpack ``results``.
        auto& indices = results.indices();
        auto& distances = results.distances();
        auto compare = distance::comparator(distance_functor);

        auto result_buffer = std::vector<Neighbor<I, detail::LocalID>>();
        auto neighbor_buffer = std::vector<Neighbor<I, detail::LocalID>>();
        auto f = extensions::clustering_distance(data, distance_functor);

        for (auto i : range) {
            auto query_id = ordinal_translator(i);
            auto closest_distance = distances.at(i, 0);

            // If we're any further than this distance, then stop performing closure
            // assignment.
            // auto upper_bound = closest_distance * (1 + parameters.epsilon_);
            auto bound = inverted::bound_with<double>(
                closest_distance, parameters.epsilon_, distance_functor
            );

            neighbor_buffer.clear();
            for (size_t j = 0, jmax = results.n_neighbors(); j < jmax; ++j) {
                auto distance = distances.at(i, j);
                if (compare(bound, distance)) {
                    break;
                }
                auto centroid_local_id = indices.at(i, j);
                auto centroid_global_id = centroid_translator(centroid_local_id);
                neighbor_buffer.push_back(
                    {centroid_global_id, distance, detail::LocalID{centroid_local_id}}
                );
            }

            // Add 1 to `max_replicas_` ensure we always get the closest centroid.
            vamana::heuristic_prune_neighbors(
                // vamana::prune_strategy(distance_functor),
                vamana::LegacyPruneStrategy(),
                parameters.max_replicas_ + 1,
                parameters.refinement_alpha_,
                data,
                data::GetDatumAccessor{},
                f,
                std::numeric_limits<I>::max(), // Guarenteed to not be in the results.
                lib::as_const_span(neighbor_buffer),
                result_buffer
            );

            for (auto centroid : result_buffer) {
                auto guard = std::lock_guard(locks.at(centroid.local_id));
                clustering.insert(
                    centroid.id(), {lib::narrow<I>(query_id), centroid.distance()}
                );
            }
        }
    };
    threads::run(threadpool, threads::StaticPartition(results.n_queries()), impl);
}

template <typename Index, std::integral I> struct ClusteringSetup {
    Index index;
    std::vector<I, lib::Allocator<I>> centroids;
};

// Deduction guide
template <typename Index, typename I>
ClusteringSetup(Index, std::vector<I, lib::Allocator<I>>) -> ClusteringSetup<Index, I>;

///
/// @brief Build the primary graph index for an inverted index.
///
/// The primary index consists of an efficient graph-based index over a subset of the total
/// dataset.
///
/// @brief data The dataset to subsample for the primary index.
/// @brief ids The indices of `data` to be used to the subsample.
/// @brief vamana_parameters Build parameters controlling the construction of the primary
///     index.
/// @brief distance The distance functor to use for construction and queries.
/// @brief threadpool The threapool to use when building the index. This function assumes
///     ownership of the provided threadpool.
///
/// NOTE: The resulting search index will not automatically perform conversion from index
/// local IDs to global dataset IDs.
///
template <data::ImmutableMemoryDataset Data, typename Distance, std::integral I>
auto build_primary_index(
    const Data& data,
    std::span<const I> ids,
    const vamana::VamanaBuildParameters& vamana_parameters,
    const Distance& distance,
    threads::ThreadPool auto threadpool
) {
    return vamana::auto_build(
        vamana_parameters,
        lib::Lazy([&]() {
            auto view = data::make_const_view(data, ids);
            auto local_data = extensions::create_auxiliary_dataset(
                data, ids.size(), data.get_allocator()
            );
            data::copy(view, local_data);
            return local_data;
        }),
        distance,
        std::move(threadpool),
        HugepageAllocator<I>()
    );
}

template <data::ImmutableMemoryDataset Data, typename Index, std::unsigned_integral I>
Clustering<I> cluster_with(
    const Data& data,
    std::span<const I> centroid_ids,
    const ClusteringParameters& params,
    Index& primary_index
) {
    for (auto id : centroid_ids) {
        if (id >= data.size()) {
            throw ANNEXCEPTION(
                "Centroid id {} is out of bounds (maximum is {})", id, data.size()
            );
        }
    }
    auto clustering = Clustering<I>(centroid_ids.begin(), centroid_ids.end());

    primary_index.set_search_parameters(
        index::vamana::VamanaSearchParameters().buffer_config(params.search_window_size_)
    );

    auto batchsize = params.batchsize_;
    size_t start = 0;
    size_t datasize = data.size();
    auto timer = lib::Timer();
    auto logger = svs::logging::get();

    while (start < datasize) {
        size_t stop = std::min(start + batchsize, datasize);
        svs::logging::debug(logger, "Processing batch [{}, {})", start, stop);
        auto indices = clustering.complement_range(threads::UnitRange(start, stop));
        auto subdata = extensions::prepare_index_search(data, lib::as_const_span(indices));

        // Get neighbor candidates.
        auto handle = timer.push_back("Search Phase");
        auto results = svs::index::search_batch(
            primary_index, subdata, params.num_intermediate_results_
        );
        handle.finish();

        // Assign dataset elements to clusters.
        auto pp = timer.push_back("post process");
        auto local_translator = [&indices](size_t i) { return indices.at(i); };
        post_process_neighbors(
            data,
            params,
            clustering,
            results,
            primary_index.get_distance(),
            local_translator,
            [centroid_ids](size_t i) { return centroid_ids[i]; },
            primary_index.borrow_threadpool()
        );
        start = stop;
    }
    svs::logging::debug(logger, "{}", timer);
    svs::logging::debug(
        logger, "Clustering Stats: {}", clustering.statistics().report("\n")
    );

    // Post Processing
    auto max_cluster_size = params.max_cluster_size_;
    auto compare = distance::comparator(primary_index.get_distance());
    if (max_cluster_size != 0) {
        clustering.reduce_maxsize(params.max_cluster_size_, compare);
    }

    clustering.sort_clusters(compare);
    return clustering;
}

} // namespace svs::index::inverted
