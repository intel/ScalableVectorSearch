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
#include "svs/concepts/data.h"
#include "svs/index/inverted/clustering.h"
#include "svs/index/inverted/common.h"
#include "svs/index/inverted/extensions.h"
#include "svs/index/inverted/memory_build_params.h"
#include "svs/index/inverted/memory_search_params.h"

// stl
#include <concepts>

namespace svs::index::inverted {

namespace detail {
inline size_t get_number_of_centroids(size_t datasize, lib::Percent percent_centroids) {
    return lib::narrow_cast<size_t>(std::floor(datasize * percent_centroids.value()));
}
} // namespace detail

/////
///// Clustered Datasets
/////

// Idea behind the `SparseClusteredDataset`.
// Dataset elements are stored in a single-monolithic location.
// The elements in each cluster point into this dataset.
//
// This has a lower memory footprint than explicitly co-locating all leaf elements in each
// cluster, but significantly lower memory overhead.

template <std::integral I> struct SparseIDs {
    I local;
    I global;
};

template <data::MemoryDataset Data, std::integral I> class SparseClusteredDataset {
  public:
    Data data_;
    // IDs for each cluster.
    // Outer Vector: One entry for each cluster.
    // Inner Vector: One entry for each element of the cluster.
    //    - local: The index of the dataset element in the local `data_` dataset.
    //    - global: The global ID of the dataset element.
    std::vector<std::vector<SparseIDs<I>>> ids_;
    size_t prefetch_offset_ = 2;

  public:
    using index_type = I;

    template <typename Original, typename Alloc>
    SparseClusteredDataset(
        const Original& original, const Clustering<I>& clustering, const Alloc& allocator
    )
        : SparseClusteredDataset{
              original, clustering, clustering.packed_leaf_translation(), allocator} {}

    template <typename Original, typename Alloc>
    SparseClusteredDataset(
        const Original& original,
        const Clustering<I>& clustering,
        const tsl::robin_map<I, I>& global_to_local_map,
        const Alloc& allocator
    )
        : data_{extensions::create_sparse_cluster(
              original, global_to_local_map.size(), allocator
          )}
        , ids_{} {
        // Copy elements from the original dataset into the local dataset.
        for (auto [global, local] : global_to_local_map) {
            data_.set_datum(local, original.get_datum(global));
        }

        // Populate the ID translation.
        clustering.for_each_cluster([&](const auto& cluster) {
            auto& these_ids = ids_.emplace_back(cluster.size());
            size_t i = 0;
            for (auto neighbor : cluster) {
                auto global_id = neighbor.id();
                these_ids.at(i) = SparseIDs<I>{
                    .local = global_to_local_map.at(global_id), .global = global_id};
                ++i;
            }
        });
    }

    SparseClusteredDataset(Data data, std::vector<std::vector<SparseIDs<I>>> ids)
        : data_{std::move(data)}
        , ids_{std::move(ids)} {}

    template <typename Callback>
    void on_leaves(Callback&& f, size_t cluster, size_t prefetch_offset) const {
        auto& ids = ids_[cluster];
        // Begin prefetching.
        size_t clustersize = ids.size();
        size_t p = 0;

        bool prefetch_enabled = prefetch_offset != 0;
        for (size_t pmax = std::min(prefetch_offset, clustersize); p < pmax; ++p) {
            data_.prefetch(ids[p].local);
        }

        for (size_t i = 0, imax = ids.size(); i < imax; ++i) {
            if (prefetch_enabled && p < clustersize) {
                data_.prefetch(ids[p].local);
                ++p;
            }

            auto idpair = ids[i];
            f(data_.get_datum(idpair.local), idpair.global);
        }
    }

    template <typename Callback> void on_leaves(Callback&& f, size_t cluster) const {
        on_leaves(SVS_FWD(f), cluster, prefetch_offset_);
    }

    // Prefetch parameters.
    size_t get_prefetch_offset() const { return prefetch_offset_; }
    void set_prefetch_offset(size_t offset) { prefetch_offset_ = offset; }

    // ///// Saving and Loading.
    // static constexpr lib::Version save_version{0, 0, 0};
    // lib::SaveTable save(const lib::SaveContext& ctx) const {
    //     // Save the IDs to a file.
    //     auto ids_file = ctx.generate_name("cluster_ids", "bin");
    //     size_t filesize = 0;
    //     {
    //         auto io = lib::open_write(ids_file);
    //         for (auto& v : ids_) {
    //             filesize += lib::write_binary(io, v.size());
    //             filesize += lib::write_binary(io, v);
    //         }
    //     }

    //     return lib::SaveTable(
    //         save_version,
    //         {{"ids_file", lib::save(ids_file.filename())},
    //          SVS_LIST_SAVE(filesize),
    //          {"num_clusters", lib::save(ids_.size())},
    //          {"integer_type", lib::save(datatype_v<I>)},
    //          SVS_LIST_SAVE_(data, ctx)}
    //     );
    // }

    // template <typename Alloc = typename Data::allocator_type>
    // static SparseClusteredDataset load(
    //     const toml::table& table,
    //     const lib::LoadContext& ctx,
    //     const lib::Version& version,
    //     const Alloc& allocator = {}
    // ) {
    //     // Ensure we have the correct integer type when decoding.
    //     auto saved_integer_type = lib::load_at<DataType>(table, "integer_type");
    //     if (saved_integer_type != datatype_v<I>) {
    //         auto type = datatype_v<I>;
    //         auto msg = fmt::format(
    //             "Clustering was saved using {} but we're trying to reload it using {}!",
    //             saved_integer_type,
    //             type
    //         );
    //         throw ANNEXCEPTION(msg);
    //     }

    //     auto num_clusters = lib::load_at<size_t>(table, "num_clusters");
    //     auto file = ctx.get_directory();
    //     file /= lib::load_at<std::filesystem::path>(table, "ids_file");
    //     auto ids = std::vector<std::vector<SparseIDs<I>>>();
    //     {
    //         auto io = lib::open_read(file);
    //         for (size_t i = 0; i < num_clusters; ++i) {
    //             size_t cluster_size = lib::read_binary<size_t>(io);
    //             auto& v = ids.emplace_back(cluster_size);
    //             lib::read_binary(io, v);
    //         }
    //     }

    //     return SparseClusteredDataset(
    //         SVS_LOAD_MEMBER_AT(table, ctx, allocator), std::move(ids)
    //     );
    // }
};

///// DenseClusteredDataset
template <typename Data, std::integral I> struct DenseCluster {
  public:
    DenseCluster(Data data, std::vector<I> ids)
        : data_{std::move(data)}
        , ids_{std::move(ids)} {
        if (data_.size() != ids_.size()) {
            throw ANNEXCEPTION("Size mismatch!");
        }
    }

    size_t size() const { return data_.size(); }

    template <typename Callback>
    void on_leaves(Callback&& f, size_t prefetch_offset = 2) const {
        size_t p = 0;
        size_t clustersize = size();
        for (size_t pmax = std::min(prefetch_offset, clustersize); p < pmax; ++p) {
            data_.prefetch(p);
        }

        for (size_t i = 0; i < clustersize; ++i) {
            if (p < clustersize) {
                data_.prefetch(p);
                ++p;
            }
            f(data_.get_datum(i), ids_[i]);
        }
    }

  public:
    Data data_;
    std::vector<I> ids_;
};

template <data::MemoryDataset Data, std::integral I> class DenseClusteredDataset {
  public:
    // Type aliases
    using index_type = I;

    // Constructor
    template <typename Original, typename Alloc>
    DenseClusteredDataset(
        const Original& original, const Clustering<I>& clustering, const Alloc& allocator
    )
        : clusters_{} {
        clustering.for_each_cluster([&](const auto& cluster) {
            size_t cluster_size = cluster.size();
            // Create a new dense leaf for this data structure.
            auto& leaf = clusters_.emplace_back(
                extensions::create_dense_cluster(original, cluster_size, allocator),
                std::vector<I>(cluster_size)
            );
            size_t i = 0;
            for (auto neighbor : cluster) {
                auto id = neighbor.id();
                leaf.data_.set_datum(i, original.get_datum(id));
                leaf.ids_[i] = id;
                ++i;
            }
        });
    }

    template <typename Callback> void on_leaves(Callback&& f, size_t cluster) const {
        clusters_.at(cluster).on_leaves(SVS_FWD(f));
    }

    size_t get_prefetch_offset() const { return prefetch_offset_; }
    void set_prefetch_offset(size_t offset) { prefetch_offset_ = offset; }

  private:
    std::vector<DenseCluster<Data, I>> clusters_;
    size_t prefetch_offset_ = 2;
};

namespace detail {

// Associate extension customization point objects with an implementation for the in-memory
// cluster.
template <typename CPO> struct AssociatedClustering;

// sparse
template <> struct AssociatedClustering<svs::tag_t<extensions::create_sparse_cluster>> {
    template <typename Data, std::integral I> using type = SparseClusteredDataset<Data, I>;
};

// dense
template <> struct AssociatedClustering<svs::tag_t<extensions::create_dense_cluster>> {
    template <typename Data, std::integral I> using type = DenseClusteredDataset<Data, I>;
};

// Cleanup accessing the associated datatype.
template <typename CPO, typename Data, std::integral I>
using compute_dataset_type = typename AssociatedClustering<CPO>::template type<Data, I>;

// Common code path for dispatching.
template <typename CPO> struct StrategyDispatcher {
    template <data::ImmutableMemoryDataset Data, typename Alloc>
    using cpo_return_type = typename CPO::template return_type<Data, Alloc>;

    template <data::ImmutableMemoryDataset Data, std::integral I, typename Alloc>
    using dataset_type = compute_dataset_type<CPO, cpo_return_type<Data, Alloc>, I>;

    // Pack data to the selected clustered representation.
    template <data::ImmutableMemoryDataset Data, std::integral I, typename Alloc>
    dataset_type<Data, I, Alloc> operator()(
        const Data& data, const Clustering<I>& clustering, const Alloc& allocator
    ) const {
        return dataset_type<Data, I, Alloc>(data, clustering, allocator);
    }
};

template <typename T> inline constexpr bool is_strategy_dispatcher_v = false;

template <typename T>
inline constexpr bool is_strategy_dispatcher_v<StrategyDispatcher<T>> = true;

} // namespace detail

// Cleaner aliases for the associated strategy.
using SparseStrategy =
    detail::StrategyDispatcher<svs::tag_t<extensions::create_sparse_cluster>>;

using DenseStrategy =
    detail::StrategyDispatcher<svs::tag_t<extensions::create_dense_cluster>>;

template <typename T>
concept StorageStrategy = detail::is_strategy_dispatcher_v<T>;

/////
///// Memory Based Inverted Index
/////

template <typename Index, typename Cluster> class InvertedIndex {
  public:
    using index_type = typename Cluster::index_type;
    using translator_type = std::vector<index_type, lib::Allocator<index_type>>;
    using search_parameters_type = InvertedSearchParameters;

    InvertedIndex(
        Index index,
        Cluster cluster,
        translator_type index_local_to_global,
        threads::NativeThreadPool threadpool
    )
        : index_{std::move(index)}
        , cluster_{std::move(cluster)}
        , index_local_to_global_{std::move(index_local_to_global)}
        , threadpool_{std::move(threadpool)} {
        // Clear out the threadpool in the inner index - prefer to handle threading
        // ourselves.
        index_.set_num_threads(1);
    }

    ///// Threading
    static constexpr bool can_change_threads() { return true; }
    size_t get_num_threads() const { return threadpool_.size(); }
    void set_num_threads(size_t num_threads) {
        threadpool_.resize(std::max<size_t>(num_threads, 1));
    }

    size_t size() const {
        // TODO: Fix
        return 0;
    }
    size_t dimensions() const { return index_.dimensions(); }

    ///// Search Parameter Setting
    search_parameters_type get_search_parameters() const {
        return InvertedSearchParameters{
            index_.get_search_parameters(), refinement_epsilon_};
    }

    void set_search_parameters(const search_parameters_type& parameters) {
        index_.set_search_parameters(parameters.primary_parameters_);
        refinement_epsilon_ = parameters.refinement_epsilon_;
    }

    // Search
    template <typename Idx, data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<Idx> results,
        const Queries& queries,
        const search_parameters_type& search_parameters,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        threads::run(
            threadpool_,
            threads::StaticPartition(queries.size()),
            [&](auto is, auto SVS_UNUSED(tid)) {
                size_t num_neighbors = results.n_neighbors();

                // Allocate scratch space to use the externally threaded
                auto scratch = index_.scratchspace();
                if (scratch.buffer.capacity() == 0) {
                    scratch.buffer.change_maxsize(1);
                }

                // A search buffer to accumulate results of the cluster search.
                auto buffer = threads::shallow_copy(scratch.buffer);
                buffer.change_maxsize(num_neighbors);

                for (auto i : is) {
                    buffer.clear();

                    auto&& query = queries.get_datum(i);
                    // Primary Index Search
                    index_.search(query, scratch, cancel);

                    auto& d = scratch.scratch;
                    auto compare = distance::comparator(d);

                    // Cluster Search
                    auto& scratch_buffer = scratch.buffer;
                    auto cutoff_distance = inverted::bound_with<double>(
                        scratch_buffer[0].distance(),
                        search_parameters.refinement_epsilon_,
                        index_.get_distance()
                    );

                    for (size_t j = 0, jmax = scratch_buffer.size(); j < jmax; ++j) {
                        // Check if request to cancel the search
                        // TODO: this cancel may also be inside on_leaves()
                        if (cancel()) {
                            return;
                        }
                        auto candidate = scratch_buffer[j];
                        if (!compare(candidate.distance(), cutoff_distance)) {
                            break;
                        }

                        auto cluster_id = candidate.id();

                        // Compute the distance between the query and each leaf element.
                        cluster_.on_leaves(
                            [&](const auto& datum, index_type global_id) {
                                // TODO: Can we provide a better API for obtaining the
                                // distance component of the scratchspace?
                                auto distance =
                                    distance::compute(scratch.scratch, query, datum);
                                buffer.insert({global_id, distance});
                            },
                            cluster_id
                        );

                        // Add the centroid to the results.
                        buffer.insert(
                            {index_local_to_global_.at(cluster_id), candidate.distance()}
                        );
                    }

                    // Store results
                    for (size_t j = 0; j < num_neighbors; ++j) {
                        results.set(buffer[j], i, j);
                    }
                }
            }
        );
    }

    // Saving
    // N.B.: For prototyping, we don't provide an entire saving API.
    // Instead, we allow for saving of the underlying index and require that the
    // clustered portion of the dataset is loaded from a `Clustering` and an original
    // dataset.
    void save_primary_index(
        const std::filesystem::path& index_config,
        const std::filesystem::path& graph,
        const std::filesystem::path& data
    ) const {
        index_.save(index_config, graph, data);
    }

  private:
    // Tunable Parameters
    double refinement_epsilon_ = 10.0;

    // The index used for the first phase of search.
    Index index_;
    Cluster cluster_;
    translator_type index_local_to_global_;

    // Transient parameters.
    threads::NativeThreadPool threadpool_;
};

struct PickRandomly {
    template <svs::data::ImmutableMemoryDataset Data, std::integral I = uint32_t>
    std::vector<I, lib::Allocator<I>> operator()(
        const Data& data,
        const ClusteringParameters& clustering_parameters,
        size_t SVS_UNUSED(num_threads),
        lib::Type<I> SVS_UNUSED(integer_type) = {}
    ) const {
        return randomly_select_centroids(
            data.size(),
            detail::get_number_of_centroids(
                data.size(), clustering_parameters.percent_centroids_
            ),
            clustering_parameters.seed_
        );
    }
};

inline constexpr PickRandomly pick_centroids_randomly{};

struct ClusteringPostOp {
    template <std::integral I>
    void operator()(const Clustering<I>& SVS_UNUSED(clustering)) const {}
};

inline constexpr ClusteringPostOp no_clustering_post_op{};

///// Auto building
template <
    typename DataProto,
    typename Distance,
    typename ThreadpoolProto,
    StorageStrategy Strategy = SparseStrategy,
    typename CentroidPicker = PickRandomly,
    typename ClusteringOp = ClusteringPostOp>
auto auto_build(
    const inverted::InvertedBuildParameters& parameters,
    DataProto data_proto,
    Distance distance,
    ThreadpoolProto threadpool_proto,
    // Customizations
    Strategy strategy = {},
    CentroidPicker centroid_picker = {},
    ClusteringOp clustering_op = {}
) {
    // Perform clustering.
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data = svs::detail::dispatch_load(std::move(data_proto), threadpool);
    size_t num_threads = threadpool.size();

    // Select Centroids.
    auto centroids =
        centroid_picker(data, parameters.clustering_parameters_, threadpool.size());

    // Build Primary Index.
    auto index = build_primary_index(
        data,
        lib::as_const_span(centroids),
        parameters.primary_parameters_,
        distance,
        std::move(threadpool)
    );

    // Cluster the dataset with the help of the primary index.
    auto clustering = cluster_with(
        data, lib::as_const_span(centroids), parameters.clustering_parameters_, index
    );

    // Perform any post-proceseccing on the clustering.
    // (Usually means saving).
    clustering_op(clustering);

    // Put together the final pieces.
    return InvertedIndex{
        std::move(index),
        strategy(data, clustering, HugepageAllocator<std::byte>()),
        std::move(centroids),
        threads::NativeThreadPool(num_threads)};
}

///// Auto Assembling.
template <typename DataProto, typename Distance, StorageStrategy Strategy>
auto assemble_from_clustering(
    const std::filesystem::path& clustering_path,
    DataProto data_proto,
    Distance distance,
    Strategy strategy,
    const std::filesystem::path& index_config,
    const std::filesystem::path& graph,
    size_t num_threads
) {
    auto threadpool = threads::as_threadpool(num_threads);
    auto original = svs::detail::dispatch_load(std::move(data_proto), threadpool);
    auto clustering = lib::load_from_disk<Clustering<uint32_t>>(clustering_path);
    auto ids = clustering.sorted_centroids();

    // Create the prinary dataset from the original.
    auto index = index::vamana::auto_assemble(
        index_config,
        GraphLoader<uint32_t>(graph),
        lib::Lazy([&]() {
            auto view = data::make_const_view(original, lib::as_const_span(ids));
            auto local_data = extensions::create_auxiliary_dataset(
                original, ids.size(), original.get_allocator()
            );
            data::copy(view, local_data);
            return local_data;
        }),
        distance,
        1
    );

    // Create the clustering and return the final results.
    return InvertedIndex(
        std::move(index),
        strategy(original, clustering, HugepageAllocator<std::byte>()),
        std::move(ids),
        std::move(threadpool)
    );
}

} // namespace svs::index::inverted
