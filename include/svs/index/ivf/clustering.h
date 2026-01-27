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

// svs
#include "svs/concepts/data.h"
#include "svs/core/data.h"
#include "svs/core/logging.h"
#include "svs/index/ivf/extensions.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"
#include "svs/lib/timing.h"

namespace svs::index::ivf {

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
    template <typename Iter> ClusteringStats(Iter begin, Iter end) {
        for (auto it = begin; it != end; ++it) {
            const auto& list = *it;
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
            const auto& list = *it;
            auto x = static_cast<double>(list.size()) - mean_size_;
            accum += x * x;
        }
        std_size_ = std::sqrt(accum / static_cast<double>(num_clusters_));
    }

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

template <data::ImmutableMemoryDataset Data, std::integral I> class Clustering {
  public:
    Data centroids_;
    std::vector<std::vector<I>> clusters_;

  public:
    using vector_type = std::vector<I>;
    using T = typename Data::element_type;

    // Type Aliases
    using iterator = typename std::vector<vector_type>::iterator;
    using const_iterator = typename std::vector<vector_type>::const_iterator;

  public:
    Clustering() = default;

    Clustering(size_t n_clusters, size_t n_dims)
        : centroids_{n_clusters, n_dims}
        , clusters_(n_clusters) {}

    Clustering(Data centroids, std::vector<vector_type> clusters)
        : centroids_{std::move(centroids)}
        , clusters_{std::move(clusters)} {}

    size_t size() const { return clusters_.size(); }

    void check_valid(size_t cluster_id) const {
        if (cluster_id >= size()) {
            throw ANNEXCEPTION(
                "Cluster id {} can't be higher than the number of clusters!",
                cluster_id,
                size()
            );
        }
    }

    size_t size(size_t id) const {
        check_valid(id);
        return clusters_[id].size();
    }

    const vector_type& cluster(size_t id) const {
        check_valid(id);
        return clusters_[id];
    }

    vector_type& cluster(size_t id) {
        check_valid(id);
        return clusters_[id];
    }

    Data centroids() { return centroids_; }

    // Iterators
    iterator begin() { return clusters_.begin(); }
    iterator end() { return clusters_.end(); }

    const_iterator cbegin() const { return clusters_.cbegin(); }
    const_iterator cend() const { return clusters_.cend(); }

    template <typename F> void for_each_cluster(F&& f) const {
        for (size_t i = 0; i < size(); i++) {
            f(cluster(i));
        }
    }

    template <typename F, threads::ThreadPool Pool>
    void for_each_cluster_parallel(F&& f, Pool& threadpool) const {
        threads::parallel_for(
            threadpool,
            threads::StaticPartition{size()},
            [&](auto indices, auto /*tid*/) {
                for (auto i : indices) {
                    f(cluster(i), i);
                }
            }
        );
    }

    ClusteringStats statistics() const { return ClusteringStats(cbegin(), cend()); }

    // Serializing and Deserializing.
    size_t serialize_clusters(std::ostream& stream) const {
        size_t bytes = lib::write_binary(stream, size());
        for (size_t i = 0; i < size(); i++) {
            bytes += lib::write_binary(stream, size(i));
            bytes += lib::write_binary(stream, clusters_[i]);
        }
        return bytes;
    }

    static std::vector<vector_type> deserialize_clusters(std::istream& stream) {
        size_t n_clusters = lib::read_binary<size_t>(stream);
        std::vector<vector_type> clusters(n_clusters);
        for (size_t i = 0; i < n_clusters; i++) {
            size_t cluster_size = lib::read_binary<size_t>(stream);
            clusters[i].resize(cluster_size);
            lib::read_binary(stream, clusters[i]);
        }
        return clusters;
    }

    // Saving and Loading.
    static constexpr lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "IVF clustering";
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        // Serialize all clusters into an auxiliary file.
        auto fullpath = ctx.generate_name("clusters", "bin");
        size_t filesize = 0;
        {
            auto io = lib::open_write(fullpath);
            filesize += serialize_clusters(io);
        }

        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(centroids, ctx),
             {"filepath", lib::save(fullpath.filename())},
             SVS_LIST_SAVE(filesize),
             {"data_type", lib::save(datatype_v<T>)},
             {"integer_type", lib::save(datatype_v<I>)},
             {"num_clusters", lib::save(size())}}
        );
    }

    template <threads::ThreadPool Pool>
    static Clustering<Data, I> load(const lib::LoadTable& table, Pool& threadpool) {
        auto saved_data_type = lib::load_at<DataType>(table, "data_type");

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

        auto expected_filesize = lib::load_at<size_t>(table, "filesize");

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
        if (saved_data_type != datatype_v<T>) {
            auto centroids_orig =
                lib::load_at<data::SimpleData<float, Data::extent>>(table, "centroids");
            if constexpr (std::is_same_v<T, svs::Float16> || std::is_same_v<T, svs::BFloat16>) {
                auto centroids = convert_data<T>(centroids_orig, threadpool);
                return Clustering<Data, I>{centroids, deserialize_clusters(io)};
            } else {
                throw ANNEXCEPTION("Centroids datatype {} not supported!", datatype_v<T>);
            }
        }

        return Clustering<Data, I>{
            SVS_LOAD_MEMBER_AT_(table, centroids), deserialize_clusters(io)};
    }
};

template <typename Data, std::integral I> struct DenseCluster {
  public:
    using data_type = Data;
    using index_type = I;

    // Default constructor for in-place initialization
    DenseCluster() = default;

    DenseCluster(Data data, std::vector<I> ids)
        : data_{std::move(data)}
        , ids_{std::move(ids)} {
        if (data_.size() != ids_.size()) {
            throw ANNEXCEPTION("Size mismatch!");
        }
    }

    size_t size() const { return data_.size(); }

    // Support for dynamic operations
    void resize(size_t new_size) {
        data_.resize(new_size);
        ids_.resize(new_size);
    }

    template <typename Callback>
    void on_leaves(Callback&& f, size_t prefetch_offset) const {
        size_t p = 0;
        size_t clustersize = size();
        auto accessor = extensions::accessor(data_);

        for (size_t pmax = std::min(prefetch_offset, clustersize); p < pmax; ++p) {
            accessor.prefetch(data_, p);
        }

        for (size_t i = 0; i < clustersize; ++i) {
            if (p < clustersize) {
                accessor.prefetch(data_, p);
                ++p;
            }
            f(accessor(data_, i), ids_[i], i);
        }
    }

    auto get_datum(size_t id) const { return data_.get_datum(id); }
    auto get_secondary(size_t id) const { return data_.get_secondary(id); }
    auto get_global_id(size_t local_id) const { return ids_[local_id]; }
    const Data& view_cluster() const { return data_; }
    Data& view_cluster() { return data_; }

  public:
    Data data_;
    std::vector<I> ids_;
};

template <
    data::ImmutableMemoryDataset Centroids,
    std::integral I,
    data::ImmutableMemoryDataset Data>
class DenseClusteredDataset {
  public:
    // Type aliases
    using index_type = I;
    using data_type = Data;

    // Default constructor for use in load()
    DenseClusteredDataset() = default;

    // Constructor from clustering (for building from existing data)
    template <typename Original, threads::ThreadPool Pool, typename Alloc>
    DenseClusteredDataset(
        const Clustering<Centroids, I>& clustering,
        const Original& original,
        Pool& threadpool,
        const Alloc& allocator
    )
        : clusters_{} {
        clustering.for_each_cluster([&](const auto& cluster) {
            size_t cluster_size = cluster.size();
            clusters_.emplace_back(
                extensions::create_dense_cluster(original, cluster_size, allocator),
                std::vector<I>(cluster_size)
            );
        });

        clustering.for_each_cluster_parallel(
            [&](const auto& cluster, size_t cluster_id) {
                auto& leaf = clusters_[cluster_id];
                extensions::set_dense_cluster(original, leaf.data_, cluster, leaf.ids_);
            },
            threadpool
        );
    }

    // Constructor for empty clusters (for assembly/dynamic operations)
    // Note: This constructor creates empty clusters with the given dimensionality
    DenseClusteredDataset(size_t num_clusters, size_t dimensions)
        : clusters_{} {
        clusters_.reserve(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            clusters_.emplace_back(Data(0, dimensions), std::vector<I>());
        }
    }

    template <typename Callback> void on_leaves(Callback&& f, size_t cluster) const {
        clusters_.at(cluster).on_leaves(SVS_FWD(f), prefetch_offset_);
    }

    size_t get_prefetch_offset() const { return prefetch_offset_; }
    void set_prefetch_offset(size_t offset) { prefetch_offset_ = offset; }

    // Cluster access (const)
    const DenseCluster<Data, I>& operator[](size_t cluster) const {
        return clusters_[cluster];
    }

    // Cluster access (mutable) - for dynamic IVF operations
    DenseCluster<Data, I>& operator[](size_t cluster) { return clusters_[cluster]; }

    // Number of clusters
    size_t size() const { return clusters_.size(); }

    // Datum access (const)
    auto get_datum(size_t cluster, size_t id) const {
        return clusters_.at(cluster).get_datum(id);
    }
    auto get_secondary(size_t cluster, size_t id) const {
        return clusters_.at(cluster).get_secondary(id);
    }
    auto get_global_id(size_t cluster, size_t id) const {
        return clusters_.at(cluster).get_global_id(id);
    }

    // View cluster data (const)
    const Data& view_cluster(size_t cluster) const {
        return clusters_.at(cluster).view_cluster();
    }

    // View cluster data (mutable) - for dynamic IVF operations
    Data& view_cluster(size_t cluster) { return clusters_[cluster].view_cluster(); }

    // Get the dimensions of the data
    size_t dimensions() const {
        if (clusters_.empty()) {
            return 0;
        }
        return clusters_[0].data_.dimensions();
    }

    ///// Saving and Loading /////

    static constexpr lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "ivf_dense_clustered_dataset";

    /// @brief Save the DenseClusteredDataset to disk.
    ///
    /// Saves all cluster data using the existing save mechanisms for each data type
    /// (SimpleData, LVQ, LeanVec, etc.), then archives everything into a single file.
    ///
    /// File format:
    /// - clusters_archive.bin: Archive containing all cluster data directories
    /// - ids.bin: Concatenated binary IDs for all clusters
    /// - Config contains: cluster_sizes array, ids_offsets array
    ///
    /// @param ctx The save context providing directory and naming utilities
    /// @return SaveTable containing metadata for reloading
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        auto num_clusters = size();
        auto dims = dimensions();

        // Compute cluster sizes and ID offsets
        std::vector<size_t> cluster_sizes(num_clusters);
        std::vector<size_t> ids_offsets(num_clusters + 1);

        size_t ids_offset = 0;
        for (size_t i = 0; i < num_clusters; ++i) {
            cluster_sizes[i] = clusters_[i].size();
            ids_offsets[i] = ids_offset;
            ids_offset += cluster_sizes[i] * sizeof(I);
        }
        ids_offsets[num_clusters] = ids_offset;

        // Create a temporary directory for cluster data
        lib::UniqueTempDirectory tempdir{"svs_ivf_clusters_save"};
        auto clusters_temp_dir = tempdir.get() / "clusters";
        std::filesystem::create_directories(clusters_temp_dir);

        // Save each cluster's data using lib::save_to_disk
        // This uses the existing save mechanism for each data type (LVQ, LeanVec, etc.)
        for (size_t i = 0; i < num_clusters; ++i) {
            auto cluster_dir = clusters_temp_dir / fmt::format("cluster_{}", i);
            lib::save_to_disk(clusters_[i].data_, cluster_dir);
        }

        // Archive all cluster data into a single file
        auto archive_path = ctx.get_directory() / "clusters_archive.bin";
        {
            std::ofstream archive_stream(archive_path, std::ios::binary);
            if (!archive_stream) {
                throw ANNEXCEPTION("Failed to open {} for writing!", archive_path);
            }
            lib::DirectoryArchiver::pack(clusters_temp_dir, archive_stream);
        }

        // Write all cluster IDs to a single file
        auto ids_path = ctx.get_directory() / "ids.bin";
        {
            auto stream = lib::open_write(ids_path);
            for (size_t i = 0; i < num_clusters; ++i) {
                if (!clusters_[i].ids_.empty()) {
                    lib::write_binary(stream, clusters_[i].ids_);
                }
            }
        }

        // Serialize offset arrays to binary files for efficiency
        auto cluster_sizes_path = ctx.get_directory() / "cluster_sizes.bin";
        {
            auto stream = lib::open_write(cluster_sizes_path);
            lib::write_binary(stream, cluster_sizes);
        }

        auto ids_offsets_path = ctx.get_directory() / "ids_offsets.bin";
        {
            auto stream = lib::open_write(ids_offsets_path);
            lib::write_binary(stream, ids_offsets);
        }

        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"num_clusters", lib::save(num_clusters)},
             {"dimensions", lib::save(dims)},
             {"prefetch_offset", lib::save(prefetch_offset_)},
             {"index_type", lib::save(datatype_v<I>)},
             {"clusters_archive_file", lib::save(std::string("clusters_archive.bin"))},
             {"ids_file", lib::save(std::string("ids.bin"))},
             {"cluster_sizes_file", lib::save(std::string("cluster_sizes.bin"))},
             {"ids_offsets_file", lib::save(std::string("ids_offsets.bin"))},
             {"total_ids_bytes", lib::save(ids_offset)}}
        );
    }

    /// @brief Check if a saved file is compatible with this loader
    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == serialization_schema && version <= save_version;
    }

    /// @brief Load a DenseClusteredDataset from disk.
    ///
    /// Loads from the archive-based format where cluster data is stored
    /// using the native save/load mechanism for each data type.
    ///
    /// @tparam Pool Thread pool type for parallel loading
    /// @tparam Allocator Allocator type for cluster data (optional)
    /// @param table The load table containing saved metadata
    /// @param threadpool Thread pool for parallel operations (unused, kept for API
    /// consistency)
    /// @param allocator Optional allocator for cluster data. For blocked data types,
    ///     this controls the block size. If not provided, default allocator is used.
    /// @return Loaded DenseClusteredDataset
    template <threads::ThreadPool Pool, typename Allocator = typename Data::allocator_type>
    static DenseClusteredDataset load(
        const lib::LoadTable& table,
        Pool& SVS_UNUSED(threadpool),
        const Allocator& allocator = Allocator{}
    ) {
        auto num_clusters = lib::load_at<size_t>(table, "num_clusters");
        // Note: "dimensions" field is saved for validation but not used during load
        // since each cluster's data type determines its own dimensions
        [[maybe_unused]] auto dims = lib::load_at<size_t>(table, "dimensions");
        auto prefetch_offset = lib::load_at<size_t>(table, "prefetch_offset");

        // Verify index type matches
        auto saved_index_type = lib::load_at<DataType>(table, "index_type");
        if (saved_index_type != datatype_v<I>) {
            throw ANNEXCEPTION(
                "DenseClusteredDataset was saved using index type {} but we're trying to "
                "reload it using {}!",
                saved_index_type,
                datatype_v<I>
            );
        }

        auto base_dir = table.context().get_directory();

        // Load offset arrays from binary files
        std::vector<size_t> cluster_sizes(num_clusters);
        std::vector<size_t> ids_offsets(num_clusters + 1);

        {
            auto stream = lib::open_read(base_dir / "cluster_sizes.bin");
            lib::read_binary(stream, cluster_sizes);
        }
        {
            auto stream = lib::open_read(base_dir / "ids_offsets.bin");
            lib::read_binary(stream, ids_offsets);
        }

        // Create a temporary directory to unpack the clusters archive
        lib::UniqueTempDirectory tempdir{"svs_ivf_clusters_load"};
        auto clusters_temp_dir = tempdir.get() / "clusters";
        std::filesystem::create_directories(clusters_temp_dir);

        // Unpack the clusters archive
        {
            std::ifstream archive_stream(
                base_dir / "clusters_archive.bin", std::ios::binary
            );
            if (!archive_stream) {
                throw ANNEXCEPTION(
                    "Failed to open {} for reading!", base_dir / "clusters_archive.bin"
                );
            }
            lib::DirectoryArchiver::unpack(archive_stream, clusters_temp_dir);
        }

        // Create result dataset with default constructor
        DenseClusteredDataset result;
        result.prefetch_offset_ = prefetch_offset;
        result.clusters_.reserve(num_clusters);

        // Load IDs file for reading
        auto ids_stream = lib::open_read(base_dir / "ids.bin");

        // Load each cluster's data and ids together
        for (size_t i = 0; i < num_clusters; ++i) {
            // Load cluster data with provided allocator
            auto cluster_dir = clusters_temp_dir / fmt::format("cluster_{}", i);
            auto cluster_data = lib::load_from_disk<Data>(cluster_dir, allocator);

            // Load cluster IDs
            size_t cluster_size = cluster_sizes[i];
            std::vector<I> cluster_ids(cluster_size);
            if (cluster_size > 0) {
                ids_stream.seekg(static_cast<std::streamoff>(ids_offsets[i]));
                lib::read_binary(ids_stream, cluster_ids);
            }

            // Construct cluster with both data and ids
            result.clusters_.emplace_back(std::move(cluster_data), std::move(cluster_ids));
        }

        return result;
    }

  private:
    std::vector<DenseCluster<Data, I>> clusters_;
    size_t prefetch_offset_ = 8;
};



} // namespace svs::index::ivf
